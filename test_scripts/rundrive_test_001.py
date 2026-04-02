import asyncio
import json
import math
import sys
import websockets
from pathlib import Path

# --- 配置 ---
ROBOT_IP = "10.1.49.9"
WS_PORT = "19891"
FILENAME = "data_log_correct.txt"

# 首帧发送后的安全等待时间（单位：秒）
INITIAL_DELAY_SEC = 5.0

# [新增配置] 夹爪发送频率限制
HAND_MAX_FREQ_HZ = 5.0
HAND_INTERVAL_MS = 1000.0 / HAND_MAX_FREQ_HZ  # 5Hz 对应 200ms 的最小发送间隔

# WebSocket URLs
WS_URL_ARM = f"ws://{ROBOT_IP}:{WS_PORT}/control_joint"
WS_URL_HAND = f"ws://{ROBOT_IP}:{WS_PORT}/control_hand"


def rad_to_deg(rad):
    return math.degrees(rad)


def parse_log_file(filename):
    log_entries = []
    file_path = Path(filename)
    if not file_path.exists():
        print(f"❌ 错误: 找不到文件 {filename}")
        sys.exit(1)

    with file_path.open('r') as f:
        print(f"✅ 正在读取文件: {filename}...")
        for line in f:
            try:
                entry = json.loads(line)
                log_entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"⚠ 警告: 无法解析行: {line.strip()}, 错误: {e}")

    print(f"✅ 解析完成，共 {len(log_entries)} 条记录。")
    return log_entries


async def send_arm_message(ws, robot_state):
    joint_pos_rad = robot_state.get("jointPos", [])
    if len(joint_pos_rad) != 7:
        return

    cmds = []
    for i, pos_rad in enumerate(joint_pos_rad):
        pos_deg = rad_to_deg(pos_rad)
        motor_cmd = {
            "motor_index": i,
            "mode": 8,
            "pos": pos_deg,
            "spd": rad_to_deg(1.305),
            "cur": 0.0
        }
        cmds.append(motor_cmd)

    message = {
        "type": "arm",
        "cmds": cmds
    }

    try:
        await ws.send(json.dumps(message))
        # 简化日志，避免刷屏
        print(f"📤 手臂 [Motor 0]: {cmds[0]['pos']:.2f}°")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ 手臂 WebSocket 连接已关闭: {e}")


async def send_hand_message(ws, robot_state):
    claw_status = robot_state.get("clawStatus", [])
    if not claw_status:
        return

    claw_status = [255] if claw_status[0] == 0 else [0]

    names = [f"left-{i}" for i in range(len(claw_status))]
    names = ["left-0"]
    message = {
        "name": names,
        "position": claw_status,
        "velocity": [100],
        "effort": [255]
    }

    try:
        await ws.send(json.dumps(message))
        print(f"🤏 夹爪发送帧: {claw_status[0]:.2f} (触发 5Hz 限制)")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ 夹爪 WebSocket 连接已关闭: {e}")


async def simulate_robot():
    entries = parse_log_file(FILENAME)
    if not entries:
        print("❌ 无数据，退出...")
        return

    try:
        print(f"🔗 正在建立 WebSocket 连接到 {ROBOT_IP}...")
        async with websockets.connect(WS_URL_ARM) as ws_arm, \
                websockets.connect(WS_URL_HAND) as ws_hand:

            print("✅ 两个连接已建立。开始模拟发送...")

            previous_time_ms = None
            is_first_frame = True

            # [新增] 记录夹爪上一次发送时的日志时间戳
            # 初始值设为一个极小值，确保第一帧一定会被发送
            last_hand_sent_time_ms = -HAND_INTERVAL_MS

            for entry in entries:
                current_time_ms = entry.get("time")
                robot_state = entry.get("robot_state")

                if not robot_state:
                    continue

                # 1. 正常帧间延时（用于整体时间轴推进）
                if not is_first_frame and previous_time_ms is not None:
                    dt_ms = current_time_ms - previous_time_ms
                    if dt_ms > 0:
                        await asyncio.sleep(dt_ms / 1000.0)

                # 2. 准备发送任务
                tasks = []

                # 手臂无频率限制，每帧必发
                tasks.append(send_arm_message(ws_arm, robot_state))

                # [核心逻辑] 夹爪 5Hz 降采样逻辑
                # 判断：当前日志时间戳 - 上次发送夹爪的日志时间戳 >= 200ms
                if (current_time_ms - last_hand_sent_time_ms) >= HAND_INTERVAL_MS:
                    tasks.append(send_hand_message(ws_hand, robot_state))
                    last_hand_sent_time_ms = current_time_ms  # 更新最后一次发送的时间戳
                else:
                    # 距离上次发送不足 200ms，直接丢弃该帧的夹爪数据（不加入 tasks）
                    pass

                    # 3. 并行执行准备好的发送任务
                await asyncio.gather(*tasks)

                # 4. 首帧发送完毕后，强制等待物理归位
                if is_first_frame:
                    print(f"\n=======================================================")
                    print(f"⏳ 首帧发送完毕！机器人位姿初始化中，安全等待 {INITIAL_DELAY_SEC} 秒...")
                    print(f"=======================================================\n")
                    await asyncio.sleep(INITIAL_DELAY_SEC)
                    is_first_frame = False

                # 更新整体时间戳给下一帧计算用
                previous_time_ms = current_time_ms

            print("✅ 所有数据已模拟发送完毕。")

    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket 连接出错: {e}")
    except ConnectionRefusedError:
        print(f"❌ 错误: 连接被拒绝。请确保目标IP服务运行正常。")
    except Exception as e:
        print(f"❌ 发生未预期错误: {e}")


if __name__ == "__main__":
    if ROBOT_IP == "10.1.49.9":
        pass
    asyncio.run(simulate_robot())