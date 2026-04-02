import asyncio
import json
import os
import threading
import time
import cv2
import websockets
from datetime import datetime

# ================== 1. 硬件与存储配置 ==================
ROBOT_IP = "10.1.49.5"  # 请确保 PC 与机器人处于同一网段
ROBOT_IP = "192.168.50.83"
WS_URL = f"ws://{ROBOT_IP}:8081/joint_state"


# 【重要】在这里手动指定摄像头序号
CAM_CONFIG = {
    "front": {"id": "/dev/video6", "folder": "front"},  # 通常外部相机是 1 或 2, 但是mac需要注意
    "hand": {"id": "/dev/video9", "folder": "hand"}
}

TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS


# ================== 2. 摄像头测试工具 ==================
def check_camera_indices(limit=5):
    """辅助工具：快速确认摄像头对应的 ID"""
    print("\n--- 正在扫描摄像头序号 ---")
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[FOUND] ID {i}: 可正常读取图像")
            cap.release()
    print("--- 扫描结束 ---\n")


# ================== 3. 增强型相机模块 ==================
class AsyncCameraStream:
    def __init__(self, source, name):
        self.cap = cv2.VideoCapture(source)
        # 强制设置分辨率和FPS（部分相机需要手动指定才能达到30fps）
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.name = name
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        if not self.cap.isOpened():
            print(f"[ERROR] 无法打开摄像头 {name} (ID: {source})")

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.001)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

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

        # 创建子文件夹
        for cfg in CAM_CONFIG.values():
            os.makedirs(os.path.join(base_dir, cfg['folder']), exist_ok=True)

    def get_relative_ms(self):
        return int((time.perf_counter() - self.start_time) * 1000)

    def write_ros_event(self, ros_json):
        """记录 WS 接收到的机器人数据"""
        event = {
            "time": self.get_relative_ms(),
            "ros": ros_json
        }
        line = json.dumps(event, ensure_ascii=False)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def write_image_event(self, frames):
        """记录并保存 30FPS 的图像"""
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


# ================== 5. 核心控制器与异步任务 ==================
class RecordingController:
    """管理数据记录逻辑：根据 footStatus 启停录制"""

    def __init__(self):
        self.is_recording = False
        self.recorder = None
        self.lock = asyncio.Lock()

    async def process_ros_data(self, data):
        # 默认不含 footStatus 时设为 0
        foot_status = data.get("footStatus", 0)

        async with self.lock:
            # 状态 0 -> 1，开启新的录制
            if foot_status == 1 and not self.is_recording:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                root_dir = f"embodiment_data_{timestamp}"
                self.recorder = DataRecorder(root_dir)
                self.is_recording = True
                print(f"\n[START] 检测到 footStatus=1，开始录制并重新计时。\n正在存储至: {root_dir}")

            # 状态 1 -> 0，停止录制
            elif foot_status == 0 and self.is_recording:
                self.is_recording = False
                if self.recorder:
                    self.recorder.close()
                    print(f"\n[STOP] 检测到 footStatus=0，已停止录制。\n数据已保存至: {self.recorder.base_dir}")
                    self.recorder = None

            # 当处于录制状态且脚踏踩下时，写入数据
            if self.is_recording and foot_status == 1 and self.recorder:
                self.recorder.write_ros_event(data)

    async def record_images(self, frames):
        async with self.lock:
            # 只有在 is_recording 为 True 才会调用 write_image_event，图像与时间一一对应
            if self.is_recording and self.recorder:
                self.recorder.write_image_event(frames)


async def ros_listener_task(controller):
    """负责监听 WS 端口并捕获 ROS 数据"""
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
    """强制 30FPS 采集"""
    print(f"[CAM] 摄像头数据流已启动 (目标: {TARGET_FPS} FPS)，等待记录指令...")
    while True:
        loop_start = time.perf_counter()

        frames = {name: cam.get_frame() for name, cam in cams.items()}

        if all(f is not None for f in frames.values()):
            # controller 内部会自动判断当前是否应该记录
            await controller.record_images(frames)

        elapsed = time.perf_counter() - loop_start
        sleep_time = FRAME_INTERVAL - elapsed
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        else:
            await asyncio.sleep(0.001)


# ================== 6. 入口函数 ==================
async def main():
    # 启动相机流
    cams = {}
    for name, cfg in CAM_CONFIG.items():
        print(f"[INIT] 启动 {name} 相机 (ID: {cfg['id']})")
        cams[name] = AsyncCameraStream(cfg['id'], name)

    # 实例化控制器
    controller = RecordingController()

    print("\n[INFO] 系统已就绪，按下 Ctrl+C 退出程序\n")

    try:
        await asyncio.gather(
            ros_listener_task(controller),
            camera_capture_task(controller, cams)
        )
    except KeyboardInterrupt:
        print("\n[EXIT] 用户手动退出。")
    finally:
        for cam in cams.values():
            cam.stop()

        # 安全退出时，确保最后一个正在记录的文件夹被正常关闭
        if controller.is_recording and controller.recorder:
            controller.recorder.close()
            print(f"[INFO] 退出前已将剩余数据安全保存至 {controller.recorder.base_dir}")


if __name__ == "__main__":
    # 如果你想先测试摄像头序号，请取消下面一行的注释
    # check_camera_indices()
     asyncio.run(main())

