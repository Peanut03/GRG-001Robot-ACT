import asyncio
import json
import os
import threading
import time
import cv2
import websockets
from datetime import datetime

# ================== 1. 硬件与存储配置 ==================
ROBOT_IP = "192.168.50.83"
WS_URL = f"ws://{ROBOT_IP}:8081/joint_state"

CAM_CONFIG = {
    "front": {"id": "/dev/video8", "folder": "front"},
    "hand": {"id": "/dev/video0", "folder": "hand"}
}

TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS

# ================== 2. 摄像头测试工具 ==================
def check_camera_indices(limit=5):
    print("\n--- 正在扫描摄像头序号 ---")
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[FOUND] ID {i}: 可正常读取图像")
            cap.release()
    print("--- 扫描结束 ---\n")

# ================== 3. 增强型相机模块（带断连检测） ==================
class AsyncCameraStream:
    def __init__(self, source, name):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.name = name
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.error_occurred = False
        self.consecutive_failures = 0
        self.max_failures = 10  # 连续10次失败视为断开

        if not self.cap.isOpened():
            print(f"[ERROR] 无法打开摄像头 {name} (ID: {source})")
            self.error_occurred = True

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                self.consecutive_failures = 0  # 重置失败计数
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures:
                    print(f"[CRITICAL] 摄像头 {self.name} 连续 {self.max_failures} 次读取失败，可能已断开！")
                    self.error_occurred = True
                    break
                time.sleep(0.01)  # 避免忙等

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def has_error(self):
        return self.error_occurred

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# ================== 4. 数据记录器 ==================
class DataRecorder:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        os.mkdir(base_dir)
        self.log_path = os.path.join(base_dir, "data_log.txt")
        self.log_file = open(self.log_path, "a", encoding="utf-8")
        self.start_time = time.perf_counter()
        self.img_counter = 1

        for cfg in CAM_CONFIG.values():
            os.makedirs(os.path.join(base_dir, cfg['folder']), exist_ok=True)

    def get_relative_ms(self):
        return int((time.perf_counter() - self.start_time) * 1000)

    def write_ros_event(self, ros_json):
        event = {
            "time": self.get_relative_ms(),
            "ros": ros_json
        }
        line = json.dumps(event, ensure_ascii=False)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def write_image_event(self, frames):
        rel_ms = self.get_relative_ms()
        img_filename = f"{self.img_counter:06d}.jpg"

        for cam_key, frame in frames.items():
            folder = CAM_CONFIG[cam_key]['folder']
            path = os.path.join(self.base_dir, folder, img_filename)
            cv2.imwrite(path, frame)

        event = {
            "time": rel_ms,
            "front": f"front_{img_filename}",
            "hand": f"hand_{img_filename}"
        }
        self.log_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.log_file.flush()
        self.img_counter += 1

    def close(self):
        self.log_file.close()

# ================== 5. 核心控制器 ==================
class RecordingController:
    def __init__(self):
        self.is_recording = False
        self.recorder = None
        self.lock = asyncio.Lock()

    async def process_ros_data(self, data):
        foot_status = data.get("footStatus", 0)

        async with self.lock:
            if foot_status == 1 and not self.is_recording:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                root_dir = f"embodiment_data_{timestamp}"
                self.recorder = DataRecorder(root_dir)
                self.is_recording = True
                print(f"\n[START] 检测到 footStatus=1，开始录制并重新计时。\n正在存储至: {root_dir}")

            elif foot_status == 0 and self.is_recording:
                self.is_recording = False
                if self.recorder:
                    self.recorder.close()
                    print(f"\n[STOP] 检测到 footStatus=0，已停止录制。\n数据已保存至: {self.recorder.base_dir}")
                    self.recorder = None

            if self.is_recording and foot_status == 1 and self.recorder:
                self.recorder.write_ros_event(data)

    async def record_images(self, frames):
        async with self.lock:
            if self.is_recording and self.recorder:
                self.recorder.write_image_event(frames)

# ================== 6. 异步任务 ==================
async def ros_listener_task(controller):
    print(f"[WS] 正在连接 {WS_URL}...")
    try:
        async with websockets.connect(WS_URL) as ws:
            print("[WS] 连接成功，等待 footStatus 变为 1 触发记录...")
            while True:
                msg = await ws.recv()
                try:
                    data = json.loads(msg)
                    if data.get("topic") == "/joint_state":
                        await controller.process_ros_data(data)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[WS ERROR] 连接断开或失败: {e}")

async def camera_capture_task(controller, cams):
    print(f"[CAM] 摄像头数据流已启动 (目标: {TARGET_FPS} FPS)，等待记录指令...")
    while True:
        loop_start = time.perf_counter()

        # 检查摄像头是否断开
        for name, cam in cams.items():
            if cam.has_error():
                raise RuntimeError(f"摄像头 '{name}' 已断开连接，请检查硬件！")

        frames = {name: cam.get_frame() for name, cam in cams.items()}

        if all(f is not None for f in frames.values()):
            await controller.record_images(frames)
        else:
            print("[WARNING] 某摄像头帧为空，可能初始化中...")

        elapsed = time.perf_counter() - loop_start
        sleep_time = FRAME_INTERVAL - elapsed
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        else:
            await asyncio.sleep(0.001)

# ================== 7. 入口函数 ==================
async def main():
    cams = {}
    for name, cfg in CAM_CONFIG.items():
        print(f"[INIT] 启动 {name} 相机 (ID: {cfg['id']})")
        cams[name] = AsyncCameraStream(cfg['id'], name)

    controller = RecordingController()
    print("\n[INFO] 系统已就绪，按下 Ctrl+C 退出程序\n")

    try:
        await asyncio.gather(
            ros_listener_task(controller),
            camera_capture_task(controller, cams)
        )
    except KeyboardInterrupt:
        print("\n[EXIT] 用户手动退出。")
    except Exception as e:
        print(f"\n[FATAL] 程序因错误终止: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for cam in cams.values():
            cam.stop()
        if controller.is_recording and controller.recorder:
            controller.recorder.close()
            print(f"[INFO] 退出前已将剩余数据安全保存至 {controller.recorder.base_dir}")

if __name__ == "__main__":
    # check_camera_indices()
    asyncio.run(main())