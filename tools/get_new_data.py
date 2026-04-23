import os
import json
import shutil
from pathlib import Path

def process_data_folders(interval=8):
    """
    处理data目录下的所有embodiment_data_文件夹，按指定间隔取数据，并重新编号
    
    Args:
        interval (int): 采样间隔，默认为8（即每8张图片取第1张）
    """
    
    # 源数据目录和目标数据目录
    data_dir = Path("data")
    new_data_dir = Path("data_new")
    
    # 创建新的data_new目录
    new_data_dir.mkdir(exist_ok=True)
    
    # 找到所有embodiment_data_开头的文件夹
    embodiment_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("embodiment_data_")]
    
    print(f"找到 {len(embodiment_dirs)} 个embodiment_data文件夹")
    print(f"采样间隔: 每{interval}张图片取第1张")
    
    for embodiment_dir in embodiment_dirs:
        print(f"处理文件夹: {embodiment_dir.name}")
        
        # 创建对应的新文件夹
        new_embodiment_dir = new_data_dir / embodiment_dir.name
        new_embodiment_dir.mkdir(exist_ok=True)
        
        # 创建front和hand子文件夹
        new_front_dir = new_embodiment_dir / "front"
        new_hand_dir = new_embodiment_dir / "hand"
        new_front_dir.mkdir(exist_ok=True)
        new_hand_dir.mkdir(exist_ok=True)
        
        # 处理图片文件
        front_dir = embodiment_dir / "front"
        hand_dir = embodiment_dir / "hand"
        
        # 获取所有图片文件并排序
        front_files = sorted([f for f in front_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        hand_files = sorted([f for f in hand_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        # 按指定间隔取图片（索引0, interval, 2*interval, 3*interval, ...）
        selected_indices = list(range(0, len(front_files), interval))
        
        # 复制选中的图片并重新编号
        new_file_index = 1
        
        for idx in selected_indices:
            if idx < len(front_files) and idx < len(hand_files):
                # 复制front图片并重新编号
                src_front = front_files[idx]
                dst_front = new_front_dir / f"{new_file_index:06d}{src_front.suffix}"
                shutil.copy2(src_front, dst_front)
                
                # 复制hand图片并重新编号
                src_hand = hand_files[idx]
                dst_hand = new_hand_dir / f"{new_file_index:06d}{src_hand.suffix}"
                shutil.copy2(src_hand, dst_hand)
                
                new_file_index += 1
        
        # 处理data_log_correct.txt文件
        log_file = embodiment_dir / "data_log_correct.txt"
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()
            
            # 按指定间隔取行，并更新图片引用
            selected_log_lines = []
            new_file_index = 1
            
            for i in range(0, len(log_lines), interval):
                if i < len(log_lines):
                    try:
                        # 解析JSON
                        log_entry = json.loads(log_lines[i].strip())
                        
                        # 更新图片引用为新的编号
                        old_front_img = log_entry.get("front_image", "")
                        old_hand_img = log_entry.get("hand_image", "")
                        
                        # 提取文件扩展名
                        front_ext = Path(old_front_img).suffix if old_front_img else ".jpg"
                        hand_ext = Path(old_hand_img).suffix if old_hand_img else ".jpg"
                        
                        # 更新为新的文件名
                        log_entry["front_image"] = f"front_{new_file_index:06d}{front_ext}"
                        log_entry["hand_image"] = f"hand_{new_file_index:06d}{hand_ext}"
                        
                        # 添加到新的日志条目列表
                        selected_log_lines.append(json.dumps(log_entry, ensure_ascii=False) + "\n")
                        new_file_index += 1
                        
                    except json.JSONDecodeError as e:
                        print(f"  警告: 无法解析第{i+1}行的JSON数据: {e}")
                        # 如果解析失败，直接复制原行
                        selected_log_lines.append(log_lines[i])
            
            # 写入新的log文件
            new_log_file = new_embodiment_dir / "data_log_correct.txt"
            with open(new_log_file, 'w', encoding='utf-8') as f:
                f.writelines(selected_log_lines)
        
        print(f"  完成处理，原始数据: {len(front_files)} 张图片, 新数据: {len(selected_indices)} 张图片")
    
    print(f"\n所有数据处理完成！新数据保存在: {new_data_dir.absolute()}")

if __name__ == "__main__":
    # 在这里修改interval参数来设置采样间隔
    # 例如：interval=5 表示每5张取1张，interval=10 表示每10张取1张
    INTERVAL = 8
    process_data_folders(interval=INTERVAL)