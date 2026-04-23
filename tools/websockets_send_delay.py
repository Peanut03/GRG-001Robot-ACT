import asyncio
import time
import json
import websockets

# --- 机器人通信配置 ---
ROBOT_IP = "192.168.50.83"
WS_PORT = "19891"
WS_URL_HAND_CONTROL = f"ws://{ROBOT_IP}:{WS_PORT}/control_hand"
WS_URL_HAND_STATUS = f"ws://{ROBOT_IP}:{WS_PORT}/left_hand_status"

# --- 测试配置 ---
MAX_WAIT_TIME_SECONDS = 10  # 最大等待时间（秒）

async def test_single_hand_send_delay():
    """测试单次夹爪发送延迟"""
    print("🚀 WebSocket 夹爪单次发送延迟测试")
    print(f"🎯 目标 IP: {ROBOT_IP}:{WS_PORT}")
    print(f"⏱️  检测条件: 速度 > 0 | 最大等待时间: {MAX_WAIT_TIME_SECONDS}秒")
    print("=" * 60)
    
    test_id = 1
    # 测试发送张开命令（发送255）
    target_position = 255
    
    
    # 使用您提供的格式
    message = {
        "name": ["left-0"],
        "position": [target_position],
        "velocity": [100],
        "effort": [255]
    }
    
    try:
        # 建立控制和状态连接
        async with websockets.connect(WS_URL_HAND_CONTROL) as ws_control, \
                   websockets.connect(WS_URL_HAND_STATUS) as ws_status:
            
            print(f"✅ 控制连接: {WS_URL_HAND_CONTROL}")
            print(f"✅ 状态连接: {WS_URL_HAND_STATUS}")
            
            # 发送命令
            send_time = time.time()
            await ws_control.send(json.dumps(message))
            print(f"📤 已发送夹爪命令 | 目标位置: {target_position} | 发送时间: {send_time:.6f}")
            
            # 等待状态反馈
            start_wait_time = time.time()
            latency_measured = False
            
            while not latency_measured and (time.time() - start_wait_time) < MAX_WAIT_TIME_SECONDS:
                try:
                    # 设置接收超时
                    message_recv = await asyncio.wait_for(ws_status.recv(), timeout=1.0)
                    status_data = json.loads(message_recv)
                    
                    # 获取消息时间戳
                    message_timestamp_ms = status_data.get('timestamp', time.time() * 1000)
                    
                    # 检查夹爪速度
                    if 'velocity' in status_data and isinstance(status_data['velocity'], list) and len(status_data['velocity']) > 0:
                        velocity = status_data['velocity'][0]
                        actual_position = status_data.get('position', [0])[0] if 'position' in status_data else 0
                        
                        print(f"📡 接收到夹爪状态 | 位置: {actual_position} | 速度: {velocity}")
                        
                        # 如果速度大于0，说明夹爪正在移动，计算延迟
                        if velocity > 0:
                            send_time_ms = send_time * 1000
                            latency_ms = message_timestamp_ms - send_time_ms
                            
                            if latency_ms > 0:
                                print(f"✅ 检测到夹爪运动! | 速度: {velocity} | 延迟: {latency_ms:.2f} ms")
                                latency_measured = True
                                break
                
                except asyncio.TimeoutError:
                    # 继续等待
                    continue
                except Exception as e:
                    print(f"❌ 接收状态数据出错: {e}")
                    break
            
            if not latency_measured:
                print(f"❌ 超时! 在 {MAX_WAIT_TIME_SECONDS} 秒内未检测到夹爪运动")
                print("可能原因:")
                print("  - 夹爪响应延迟较长")
                print("  - 夹爪已处于目标位置（无需移动）")
                print("  - 夹爪控制未正确响应")
                
    except Exception as e:
        print(f"❌ 连接失败: {e}")

async def main():
    """主函数"""
    await test_single_hand_send_delay()

if __name__ == "__main__":
    asyncio.run(main())