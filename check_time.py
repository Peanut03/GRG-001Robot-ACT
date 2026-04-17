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
from concurrent.futures import ThreadPoolExecutor

# --- 路径设置 ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from act_pytorch.policies.act_policy import ACTPolicy
from act_pytorch.utils.train_utils import set_seed

# --- 机器人通信配置 ---
ROBOT_IP = "192.168.50.83"
WS_PORT = "19891"
WS_URL_ARM = f"ws://{ROBOT_IP}:{WS_PORT}/control_joint"
WS_URL_HAND = f"ws://{ROBOT_IP}:{WS_PORT}/control_hand"
WS_URL_ARM_STATUS = f"ws://{ROBOT_IP}:{WS_PORT}/arm_status"
WS_URL_HAND_STATUS = f"ws://{ROBOT_IP}:{WS_PORT}/left_hand_status"

# 首帧安全等待时间（秒）
INITIAL_DELAY_SEC = 5.0

# 夹爪频率限制（5Hz → 最小间隔 200ms）
HAND_MAX_FREQ_HZ = 5.0
HAND_INTERVAL_MS = 1000.0 / HAND_MAX_FREQ_HZ

# 关节执行完成阈值（弧度误差小于0.001）
JOINT_TOLERANCE_RAD = 0.001

# --- 统一归一化工具函数 ---
def normalize(data, stats, prefix):
    mode = stats.get("norm_mode", "mean_std")
    data = np.array(data, dtype=np.float32)
    if mode == "min_max":
        # 对应 train_utils 里的计算方式
        denom = stats[f"{prefix}_max"] - stats[f"{prefix}_min"]
        denom = np.clip(denom, 1e-2, None)  # 防止除零
        return (data - stats[f"{prefix}_min"]) / denom
    elif mode == "mixed":
        # 混合归一化：前7个自由度使用meanstd，最后一个使用minmax
        if len(data.shape) == 1:
            normalized_data = np.zeros_like(data, dtype=np.float32)
            # 前7个维度使用meanstd
            for i in range(min(7, len(data))):
                normalized_data[i] = (data[i] - stats[f"{prefix}_mean"][0, i]) / stats[f"{prefix}_std"][0, i]
            # 最后一个维度使用minmax（如果存在）
            if len(data) > 7:
                denom = stats[f"{prefix}_max"][0, 7] - stats[f"{prefix}_min"][0, 7]
                denom = np.clip(denom, 1e-2, None)
                normalized_data[7] = (data[7] - stats[f"{prefix}_min"][0, 7]) / denom
            return normalized_data
        else:
            normalized_data = np.zeros_like(data, dtype=np.float32)
            # 前7个维度使用meanstd
            for i in range(min(7, data.shape[-1])):
                normalized_data[..., i] = (data[..., i] - stats[f"{prefix}_mean"][0, i]) / stats[f"{prefix}_std"][0, i]
            # 最后一个维度使用minmax（如果存在）
            if data.shape[-1] > 7:
                denom = stats[f"{prefix}_max"][0, 7] - stats[f"{prefix}_min"][0, 7]
                denom = np.clip(denom, 1e-2, None)
                normalized_data[..., 7] = (data[..., 7] - stats[f"{prefix}_min"][0, 7]) / denom
            return normalized_data
    else:
        # mean_std 模式
        return (data - stats[f"{prefix}_mean"]) / stats[f"{prefix}_std"]


def denormalize(data, stats, prefix):
    mode = stats.get("norm_mode", "mean_std")
    data = np.array(data, dtype=np.float32)
    if mode == "min_max":
        denom = stats[f"{prefix}_max"] - stats[f"{prefix}_min"]
        denom = np.clip(denom, 1e-2, None)
        return data * denom + stats[f"{prefix}_min"]
    elif mode == "mixed":
        # 对应的反归一化：前7个自由度使用meanstd，最后一个使用minmax
        if len(data.shape) == 1:
            denormalized_data = np.zeros_like(data, dtype=np.float32)
            # 前7个维度使用meanstd反归一化
            for i in range(min(7, len(data))):
                denormalized_data[i] = data[i] * stats[f"{prefix}_std"][0, i] + stats[f"{prefix}_mean"][0, i]
            # 最后一个维度使用minmax反归一化（如果存在）
            if len(data) > 7:
                denom = stats[f"{prefix}_max"][0, 7] - stats[f"{prefix}_min"][0, 7]
                denom = np.clip(denom, 1e-2, None)
                denormalized_data[7] = data[7] * denom + stats[f"{prefix}_min"][0, 7]
            return denormalized_data
        else:
            denormalized_data = np.zeros_like(data, dtype=np.float32)
            # 前7个维度使用meanstd反归一化
            for i in range(min(7, data.shape[-1])):
                denormalized_data[..., i] = data[..., i] * stats[f"{prefix}_std"][0, i] + stats[f"{prefix}_mean"][0, i]
            # 最后一个维度使用minmax反归一化（如果存在）
            if data.shape[-1] > 7:
                denom = stats[f"{prefix}_max"][0, 7] - stats[f"{prefix}_min"][0, 7]
                denom = np.clip(denom, 1e-2, None)
                denormalized_data[..., 7] = data[..., 7] * denom + stats[f"{prefix}_min"][0, 7]
            return denormalized_data
    else:
        # mean_std 模式
        return data * stats[f"{prefix}_std"] + stats[f"{prefix}_mean"]
# ------------------------------

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
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    await asyncio.sleep(0.001)

async def send_hand_message(ws_hand, claw_value, last_sent_time_ms, current_time_ms):
    # 根据 rundrive_test_001.py 的逻辑：
    # 如果模型输出的最后一维是负数，说明要张开夹爪，对应 clawStatus[0] == 0，发送 255
    # 如果模型输出的最后一维是正数，说明要闭合夹爪，对应 clawStatus[0] != 0，发送 0
    claw_status = [255] if claw_value < 0.0 else [0]
    if (current_time_ms - last_sent_time_ms) >= HAND_INTERVAL_MS:
        message = {
            "name": ["left-0"],
            "position": claw_status,
            "velocity": [100],
            "effort": [255]
        }
        await ws_hand.send(json.dumps(message))
        print(f"🤏 夹爪指令: {claw_status[0]} (模型输出: {claw_value:.4f})")
        await asyncio.sleep(0.001)
        return current_time_ms
    else:
        await asyncio.sleep(0.001)
        return last_sent_time_ms


async def receive_arm_status_continuous(ws_arm_status, current_state, stop_event):
    """持续接收手臂状态"""
    while not stop_event.is_set():
        try:
            message = await asyncio.wait_for(ws_arm_status.recv(), timeout=2.0)
            joint_data = json.loads(message)
            # print(f"🔍 接收到的手臂原始数据: {joint_data}")
            
            if isinstance(joint_data, list):
                try:
                    sorted_joints = sorted(joint_data, key=lambda x: int(str(x['name'])))
                except (ValueError, KeyError, TypeError):
                    sorted_joints = joint_data
                
                arm_positions_deg = [joint['pos'] for joint in sorted_joints]
                arm_positions_rad = [math.radians(pos) for pos in arm_positions_deg]
                current_state['arm'] = arm_positions_rad[:7]
                
                # 安全打印角度信息
                try:
                    formatted_angles = [f"{float(x):.2f}°" for x in arm_positions_deg[:7]]
                    print(f"✅ 手臂状态已更新 (角度): {formatted_angles}")
                except (ValueError, TypeError) as e:
                    print(f"⚠️ 手臂角度打印失败，数据异常: {arm_positions_deg[:7]}")
        except asyncio.TimeoutError:
            # 正常超时，继续循环
            continue
        except websockets.exceptions.ConnectionClosed:
            print("⚠️ 手臂状态连接已关闭")
            try:
                print("🔄 尝试重新连接手臂状态...")
                new_ws = await websockets.connect(WS_URL_ARM_STATUS)
                ws_arm_status = new_ws
                print("✅ 手臂状态重连成功")
                continue
            except Exception as reconnect_err:
                print(f"❌ 手臂状态重连失败: {reconnect_err}")
            break
        except Exception as e:
            print(f"⚠️ 手臂状态处理异常: {e}")
        await asyncio.sleep(0.001)

async def receive_hand_status_continuous(ws_hand_status, current_state, stop_event):
    """持续接收手部状态"""
    while not stop_event.is_set():
        try:
            message = await asyncio.wait_for(ws_hand_status.recv(), timeout=2.0)
            hand_data = json.loads(message)
            # print(f"🔍 接收到的夹爪原始数据: {hand_data}")
            
            if 'position' in hand_data and len(hand_data['position']) > 0:
                claw_pos = hand_data['position'][0]
                if claw_pos == 0:
                    claw_normalized = -math.pi
                elif claw_pos == 255:
                    claw_normalized = math.pi
                else:
                    claw_normalized = -math.pi + (claw_pos / 255.0) * (2 * math.pi)
                current_state['hand'] = [claw_normalized]
                # print(f"✅ 夹爪状态已更新: {claw_normalized:.4f}")
        except asyncio.TimeoutError:
            continue
        except websockets.exceptions.ConnectionClosed:
            print("⚠️ 手部状态连接已关闭")
            break
        except Exception as e:
            print(f"⚠️ 手部状态处理异常: {e}")
        await asyncio.sleep(0.001)


    
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
        print(f"Normalization Mode: {self.norm_stats.get('norm_mode', 'mean_std')}")

    @torch.no_grad()
    def predict(self, head_img, hand_img, qpos_raw):
        """
        输入: OpenCV图像和原始qpos(list or numpy)
        输出: 反归一化后的预测动作
        """
        # 1. 处理图像 (Batch=1, Num_Cam=2, C, H, W)
        head_tensor = get_image_tensor_from_cv2(head_img)
        hand_tensor = get_image_tensor_from_cv2(hand_img)
        # 堆叠摄像头维度: (2, 3, 480, 640)
        images = torch.stack([head_tensor, hand_tensor], dim=0)
        # 增加 Batch 维度: (1, 2, 3, 480, 640)
        images = images.unsqueeze(0).to(self.device)

        # 2. 处理 qpos - 使用统一的归一化函数
        qpos_np = np.array(qpos_raw)
        qpos_norm = normalize(qpos_np, self.norm_stats, "qpos")
        qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(self.device)

        if qpos_tensor.ndim == 3:
            qpos_tensor = qpos_tensor.squeeze(1)  # [1, 1, 7] ->

        # 3. 推理
        # 注意：ACT模型返回的是一个动作序列 (batch, sequence_length, action_dim)
        action_pred_norm = self.policy(qpos_tensor, images)
        action_pred_norm = action_pred_norm.cpu().numpy()[0]  # 取出第0个batch

        # 4. 反归一化动作 - 使用统一的反归一化函数
        action_unnorm = denormalize(action_pred_norm, self.norm_stats, "action")

        # 5. 限制范围 (防止弧度溢出)
        if self.norm_stats.get("norm_mode") == "min_max":
            action_unnorm = np.clip(action_unnorm, self.norm_stats["action_min"], self.norm_stats["action_max"])

        return action_unnorm

# --- 主异步函数 ---
async def async_main():
    parser = make_parser()
    args = parser.parse_args()
    engine = ACTInference(args.checkpoint)
    
    # --- 创建线程池执行器 ---
    # 这将允许 predict 在独立线程中运行，不阻塞网络IO
    executor = ThreadPoolExecutor(max_workers=1) 
    loop = asyncio.get_event_loop()
    loop.set_default_executor(executor)

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
            websockets.connect(WS_URL_ARM_STATUS) as ws_arm_status,
            websockets.connect(WS_URL_HAND_STATUS) as ws_hand_status
        ):
            print("✅ 所有 WebSocket 连接成功！")
            
            current_state = {'arm': [0.0] * 7, 'hand': [3.14159]}
            stop_event = asyncio.Event()
            
            # 启动持续接收任务
            receive_arm_task = asyncio.create_task(
                receive_arm_status_continuous(ws_arm_status, current_state, stop_event)
            )
            receive_hand_task = asyncio.create_task(
                receive_hand_status_continuous(ws_hand_status, current_state, stop_event)
            )
            
            frame_count = 0
            is_first_frame = True
            last_hand_sent_time_ms = -HAND_INTERVAL_MS
            
            try:
                while True:
                    # 1. 读取图像并测量耗时
                    image_start_time = time.time()
                    ret1, head_frame = head_cap.read()
                    ret2, hand_frame = hand_cap.read()
                    image_end_time = time.time()
                    image_duration_ms = (image_end_time - image_start_time) * 1000
                    if not ret1 or not ret2:
                        print("⚠️ 摄像头读取失败，退出循环。")
                        break

                    # --- 关键改动：获取当前状态的快照 ---
                    # 在推理开始前锁定当前状态，防止推理期间状态突变导致数据错位
                    current_qpos = current_state['arm'] + current_state['hand']
                    
                    # 2. 异步推理（非阻塞）并测量耗时
                    # 将 predict 包装在 run_in_executor 中
                    inference_start_time = time.time()
                    try:
                        # 注意：这里传入的是函数对象和参数，不要加括号()
                        action_seq = await loop.run_in_executor(
                            None, # 使用默认执行器（即上面设置的线程池）
                            engine.predict, # 要运行的函数
                            head_frame, hand_frame, current_qpos # 函数参数
                        )
                    except Exception as e:
                        print(f"❌ 推理任务出错: {e}")
                        continue
                    inference_end_time = time.time()
                    inference_duration_ms = (inference_end_time - inference_start_time) * 1000
                    
                    target_action = action_seq[0]
                    
                    # 打印耗时信息
                    print(f"📊 图像获取耗时: {image_duration_ms:.2f}ms, 推理耗时: {inference_duration_ms:.2f}ms")
                    
                    # 3. 发送控制指令
                    current_time_ms = time.time() * 1000
                    
                    # 发送手臂指令
                    await send_arm_message(ws_arm, target_action[:7])
                    
                    # 发送夹爪指令
                    new_hand_time = await send_hand_message(
                        ws_hand, 
                        target_action[7], 
                        last_hand_sent_time_ms, 
                        current_time_ms
                    )
                    last_hand_sent_time_ms = new_hand_time

                    # 4. 等待动作执行完成（只检查第一个关节）
                    target_joint_0 = target_action[0]
                    execution_start_time = time.time()
                    while True:
                        current_joint_0 = current_state['arm'][0]
                        error = abs(current_joint_0 - target_joint_0)
                        if error <= JOINT_TOLERANCE_RAD:
                            execution_end_time = time.time()
                            execution_duration_ms = (execution_end_time - execution_start_time) * 1000
                            print(f"⏱️  关节0执行完成! 目标: {rad_to_deg(target_joint_0):.4f}°, 当前: {rad_to_deg(current_joint_0):.4f}°, 误差: {error:.6f}rad, 耗时: {execution_duration_ms:.2f}ms")
                            break
                        await asyncio.sleep(0.001)  # 短暂等待后重试

                    # 5. 首帧安全等待（如果需要）
                    if is_first_frame:
                        print(f"\n首帧动作已发送！等待机器人归位...")
                        await asyncio.sleep(INITIAL_DELAY_SEC)
                        is_first_frame = False

                    frame_count += 1
                    
                    if args.show_video:
                        cv2.imshow('Head Camera', head_frame)
                        cv2.imshow('Hand Camera', hand_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

            except Exception as e:
                print(f"❌ 主循环错误: {e}")
            finally:
                stop_event.set()
                await asyncio.gather(receive_arm_task, receive_hand_task, return_exceptions=True)
                executor.shutdown(wait=True) # 关闭线程池

    except Exception as e:
        print(f"❌ 连接错误: {e}")
    finally:
        head_cap.release()
        hand_cap.release()
        if args.show_video:
            cv2.destroyAllWindows()

# --- 入口 ---
if __name__ == '__main__':
    asyncio.run(async_main())