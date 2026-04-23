import asyncio
import time
import json
import websockets

# --- 机器人通信配置 ---
ROBOT_IP = "192.168.50.83"
WS_PORT = "19891"
WS_URL_HAND_STATUS = f"ws://{ROBOT_IP}:{WS_PORT}/left_hand_status"

async def get_hand_status():
    """持续获取手部状态并打印耗时"""
    try:
        # 连接到WebSocket服务器
        async with websockets.connect(WS_URL_HAND_STATUS) as websocket:
            print(f"✅ 已连接到 {WS_URL_HAND_STATUS}")
            print("🔄 开始持续接收手部数据 (按 Ctrl+C 停止)...\n")
            
            message_count = 0
            start_time_total = time.time()
            
            while True:
                message_start_time = time.time()
                
                # 接收数据
                message = await websocket.recv()
                message_end_time = time.time()
                
                # 解析JSON数据
                hand_data = json.loads(message)
                message_count += 1
                
                # 计算单次接收耗时
                single_elapsed = (message_end_time - message_start_time) * 1000
                
                # 计算总运行时间
                total_elapsed = time.time() - start_time_total
                
                # 打印手部状态
                print(f"=== 接收次数: {message_count} | 总运行时间: {total_elapsed:.2f}s ===")
                print(f"手指名称: {hand_data['name']}")
                print(f"位置: {[f'{pos:.2f}' for pos in hand_data['position']]}")
                print(f"速度: {[f'{vel:.2f}' for vel in hand_data['velocity']]}")
                print(f"力矩: {[f'{eff:.2f}' for eff in hand_data['effort']]}")
                print(f"⏱️ 单次接收耗时: {single_elapsed:.2f} 毫秒\n")
                
    except KeyboardInterrupt:
        total_time = time.time() - start_time_total
        print(f"\n⏹️  用户中断接收")
        print(f"📊 总共接收了 {message_count} 条消息，总运行时间: {total_time:.2f} 秒")
        print(f"📈 平均频率: {message_count/total_time:.2f} Hz")
    except Exception as e:
        total_time = time.time() - start_time_total
        print(f"❌ 连接失败或接收数据出错: {e}")
        print(f"⏱️  总运行时间: {total_time:.2f} 秒")
        print(f"📊 总共成功接收了 {message_count} 条消息")

if __name__ == "__main__":
    asyncio.run(get_hand_status())