import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import re

def parse_output_file(file_path):
    """
    解析 output.txt 文件，提取关节位置数据
    """
    joint_positions = []
    indices = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            # 使用正则表达式匹配数据行
            match = re.match(r'(\d+):\s*\[(.+)\]', line.strip())
            if match:
                idx = int(match.group(1))
                values_str = match.group(2)
                try:
                    values = [float(x.strip()) for x in values_str.split(',')]
                    # 前7个值是关节位置
                    joint_pos = values[:7]
                    joint_positions.append(joint_pos)
                    indices.append(idx)
                except ValueError:
                    print(f"Warning: Skipping invalid line: {line.strip()}")
                    continue
    
    return np.array(indices), np.array(joint_positions)

def visualize_output_data(output_file, joint_idx=0):
    """
    可视化 output.txt 中的数据
    :param output_file: output.txt 文件路径
    :param joint_idx: 要可视化的关节索引 (0-6)
    """
    # 解析数据
    indices, joint_positions = parse_output_file(output_file)
    
    if len(indices) == 0:
        print("No valid data found in the file.")
        return
    
    print(f"Loaded {len(indices)} data points.")
    print(f"Joint positions shape: {joint_positions.shape}")
    
    # 创建时间轴（使用索引作为时间）
    times = indices.astype(float)
    
    # 绘制图表
    plt.figure(figsize=(14, 7))
    
    # 绘制原始数据点
    plt.scatter(times, joint_positions[:, joint_idx], 
               label=f'Joint Data (Joint {joint_idx})', 
               color='blue', alpha=0.5, s=15)
    
    # 如果数据点足够多，尝试拟合平滑曲线
    if len(times) > 3:
        try:
            # 使用三次样条插值创建平滑曲线
            smooth_func = interp1d(times, joint_positions[:, joint_idx], 
                                 kind='cubic', fill_value="extrapolate")
            smooth_times = np.linspace(min(times), max(times), 1000)
            smooth_joints = smooth_func(smooth_times)
            plt.plot(smooth_times, smooth_joints, 
                    label='Fitted Spline Curve', color='cyan', alpha=0.7)
        except Exception as e:
            print(f"Could not create spline curve: {e}")
            # 如果样条插值失败，使用线性插值
            if len(times) > 1:
                linear_func = interp1d(times, joint_positions[:, joint_idx], 
                                     kind='linear', fill_value="extrapolate")
                smooth_times = np.linspace(min(times), max(times), 1000)
                smooth_joints = linear_func(smooth_times)
                plt.plot(smooth_times, smooth_joints, 
                        label='Linear Interpolation', color='orange', alpha=0.7)
    
    plt.title(f'Temporal Distribution of Joint {joint_idx} Data')
    plt.xlabel('Index (Time)')
    plt.ylabel('Joint Position (rad)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"[Success] Visualized {len(times)} data points for joint {joint_idx}.")

# 运行可视化
if __name__ == "__main__":
    output_file = "./output.txt"
    visualize_output_data(output_file, joint_idx=0)