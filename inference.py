import os
import sys
import asyncio
import time
import json
import math
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
import websockets

# --- 路径设置 ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from act_pytorch.policies.act_policy import ACTPolicy
from act_pytorch.utils.train_utils import set_seed

# --- 机器人通信配置 ---
ROBOT_IP = "10.1.49.9"
WS_PORT = "19891"
WS_URL_ARM = f"ws://{ROBOT_IP}:{WS_PORT}/control_joint"
WS_URL_HAND = f"ws://{ROBOT_IP}:{WS_PORT}/control_hand"
WS_URL_STATE = f"ws://{ROBOT_IP}:{WS_PORT}/robot_state"  # 新增状态端点

# 首帧安全等待时间（秒）
INITIAL_DELAY_SEC = 5.0

# 夹爪频率限制（5Hz → 最小间隔 200ms）
HAND_MAX_FREQ_HZ = 5.0
HAND_INTERVAL_MS = 1000.0 / HAND_MAX_FREQ_HZ

# 全局变量：存储最新机器人状态
latest_robot_qpos = [0.0] * 8

# --- 工具函数 ---
def rad_to_deg(rad):
    return math.degrees(rad)

def make_parser():
    parser = argparse.ArgumentParser(
        description="Real-time ACT inference with camera input and robot control via WebSocket.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ACT checkpoint.")
    parser.add_argument("--head_cam", type=str, default="/dev/video8", help="Head camera device index.")
    parser.add_argument("--hand_cam", type=str, default="/dev/video0", help="Hand camera device index.")
    parser.add_argument("--show_video", action="store_true", help="Show live camera feed.")
    return parser

def get_image_tensor_from_cv2(cv2_image):
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    preprocess = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(pil_image)

# --- 发送函数 ---
async def send_arm_message(ws_arm, joint_pos_rad):
    if len(joint_pos_rad) < 7:
        return
    cmds = []
    for i in range(7):
        pos_deg = rad_to_deg(joint_pos_rad[i])
        motor_cmd = {
            "motor_index": i,
            "mode": 8,
            "pos": pos_deg,
            "spd": rad_to_deg(1.305),
            "cur": 0.0
        }
        cmds.append(motor_cmd)
    message = {"type": "arm", "cmds": cmds}
    await ws_arm.send(json.dumps(message))
    print(f"📤 手臂 [Motor 0]: {cmds[0]['pos']:.2f}°")

async def send_hand_message(ws_hand, claw_value, last_sent_time_ms, current_time_ms):
    claw_status = [255] if claw_value >= 0.5 else [0]
    if (current_time_ms - last_sent_time_ms) >= HAND_INTERVAL_MS:
        message = {
            "name": ["left-0"],
            "position": claw_status,
            "velocity": [100],
            "effort": [255]
        }
        await ws_hand.send(json.dumps(message))
        print(f"🤏 夹爪指令: {claw_status[0]}")
        return current_time_ms
    else:
        return last_sent_time_ms

# --- 状态获取与监听 ---
async def fetch_initial_robot_state(ws_state):
    """启动时主动请求一次机器人当前状态"""
    try:
        await ws_state.send(json.dumps({"cmd": "get_state"}))
        response = await asyncio.wait_for(ws_state.recv(), timeout=5.0)
        data = json.loads(response)
        joint_pos = data.get("jointPos", [0.0] * 7)
        claw_status = data.get("clawStatus", [0])
        claw_value = 0.0 if claw_status[0] == 0 else 1.0
        initial_qpos = list(joint_pos) + [claw_value]
        print(f"✅ 初始机器人状态: {initial_qpos}")
        return initial_qpos
    except Exception as e:
        print(f"⚠️ 获取初始状态失败 ({e})，使用默认零位姿")
        return [0.0] * 8

async def listen_robot_state(ws_state):
    """持续监听机器人广播的状态"""
    global latest_robot_qpos
    try:
        async for message in ws_state:
            try:
                data = json.loads(message)
                joint_pos = data.get("jointPos", [0.0] * 7)
                claw_status = data.get("clawStatus", [0])
                claw_value = 0.0 if claw_status[0] == 0 else 1.0
                latest_robot_qpos = list(joint_pos) + [claw_value]
            except Exception as e:
                print(f"⚠️ 状态解析错误: {e}")
    except Exception as e:
        print(f"❌ 状态监听异常: {e}")

# --- ACT 推理类 ---
class ACTInference:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.train_args = ckpt["args"]
        self.norm_stats = ckpt["norm_stats"]

        set_seed(self.train_args.seed)
        self.policy = ACTPolicy(self.train_args).to(self.device)
        self.policy.model.load_state_dict(ckpt["model"])
        self.policy.eval()
        print(f"✅ Model loaded from {checkpoint_path}")

    def normalize_qpos(self, qpos_raw):
        qpos_np = np.array(qpos_raw)
        norm_mode = self.norm_stats.get("norm_mode", "min_max")
        if norm_mode == "min_max":
            qpos_min = self.norm_stats["qpos_min"]
            qpos_max = self.norm_stats["qpos_max"]
            qpos_norm = (qpos_np - qpos_min) / (qpos_max - qpos_min + 1e-8)
        else:
            qpos_mean = self.norm_stats["qpos_mean"]
            qpos_std = self.norm_stats["qpos_std"]
            qpos_norm = (qpos_np - qpos_mean) / qpos_std
        return qpos_norm

    def unnormalize_action(self, action_pred):
        norm_mode = self.norm_stats.get("norm_mode", "min_max")
        if norm_mode == "min_max":
            action_min = self.norm_stats["action_min"]
            action_max = self.norm_stats["action_max"]
            action_unnorm = action_pred * (action_max - action_min) + action_min
        else:
            action_mean = self.norm_stats["action_mean"]
            action_std = self.norm_stats["action_std"]
            action_unnorm = action_pred * action_std + action_mean
        return action_unnorm

    @torch.no_grad()
    def predict(self, head_img, hand_img, qpos_raw):
        head_tensor = get_image_tensor_from_cv2(head_img)
        hand_tensor = get_image_tensor_from_cv2(hand_img)
        images = torch.stack([head_tensor, hand_tensor], dim=0).unsqueeze(0).to(self.device)

        qpos_norm = self.normalize_qpos(qpos_raw)
        qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(self.device)
        if qpos_tensor.ndim == 3:
            qpos_tensor = qpos_tensor.squeeze(1)

        action_pred = self.policy(qpos_tensor, images)
        action_pred = action_pred.cpu().numpy()[0]
        action_unnorm = self.unnormalize_action(action_pred)
        return action_unnorm

# --- 主异步函数 ---
async def async_main():
    parser = make_parser()
    args = parser.parse_args()

    engine = ACTInference(args.checkpoint)

    head_cap = cv2.VideoCapture(args.head_cam)
    hand_cap = cv2.VideoCapture(args.hand_cam)

    if not head_cap.isOpened():
        print(f"❌ 无法打开头戴相机 (设备 {args.head_cam})")
        return
    if not hand_cap.isOpened():
        print(f"❌ 无法打开手部相机 (设备 {args.hand_cam})")
        head_cap.release()
        return

    print("✅ 摄像头已打开，正在连接机器人...")

    try:
        async with (
            websockets.connect(WS_URL_ARM) as ws_arm,
            websockets.connect(WS_URL_HAND) as ws_hand,
            websockets.connect(WS_URL_STATE) as ws_state
        ):
            print("✅ 所有 WebSocket 连接成功！")

            # 获取真实初始状态（关键！）
            current_qpos = await fetch_initial_robot_state(ws_state)

            # 启动后台状态监听
            state_task = asyncio.create_task(listen_robot_state(ws_state))

            frame_count = 0
            is_first_frame = True
            last_hand_sent_time_ms = -HAND_INTERVAL_MS

            while True:
                ret1, head_frame = head_cap.read()
                ret2, hand_frame = hand_cap.read()
                if not ret1 or not ret2:
                    print("⚠️ 摄像头读取失败，退出循环。")
                    break

                # 使用最新机器人状态（来自监听）
                current_qpos = latest_robot_qpos.copy()

                # 推理
                action_seq = engine.predict(head_frame, hand_frame, current_qpos)
                target_action = action_seq[0]

                current_time_ms = time.time() * 1000

                # 发送控制指令
                tasks = [send_arm_message(ws_arm, target_action[:7])]
                new_hand_time = await send_hand_message(
                    ws_hand, target_action[7], last_hand_sent_time_ms, current_time_ms
                )
                last_hand_sent_time_ms = new_hand_time
                await asyncio.gather(*tasks)

                # 首帧安全等待
                if is_first_frame:
                    print(f"\n{'='*60}")
                    print(f"⏳ 首帧动作已发送！等待机器人归位 {INITIAL_DELAY_SEC} 秒...")
                    print(f"{'='*60}\n")
                    await asyncio.sleep(INITIAL_DELAY_SEC)
                    is_first_frame = False

                frame_count += 1

                if args.show_video:
                    cv2.imshow('Head Camera', head_frame)
                    cv2.imshow('Hand Camera', hand_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 清理后台任务
            state_task.cancel()
            try:
                await state_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        print(f"❌ 运行时错误: {e}")
    finally:
        head_cap.release()
        hand_cap.release()
        if args.show_video:
            cv2.destroyAllWindows()

# --- 入口 ---
if __name__ == '__main__':
    asyncio.run(async_main())