import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

def process_and_align_teleop_data(log_file, output_file, arm_choice='both'):
    """
    读取、平滑并对齐摇操数据。
    :param log_file: 原始数据日志路径
    :param output_file: 对齐后输出的数据路径
    :param arm_choice: 'left' (左臂), 'right' (右臂), 'both' (双臂)
    """
    ros_times = []
    ros_joints = []
    ros_claws = []

    img_times = []
    img_fronts = []
    img_hands = []

    # 1. 解析原始数据日志
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            t = data['time']

            # 读取 ROS 摇操数据
            if 'ros' in data:
                ros_times.append(t)

                # jointPos[0] 为左臂, jointPos[1] 为右臂
                left_joint = data['ros']['jointPos'][0]
                right_joint = data['ros']['jointPos'][1]

                # clawStatus[0] 为左臂, clawStatus[1] 为右臂 (1为闭合，0为打开)
                left_claw = data['ros']['clawStatus'][0]
                right_claw = data['ros']['clawStatus'][1]

                if arm_choice == 'left':
                    ros_joints.append(left_joint)
                    ros_claws.append([left_claw])
                elif arm_choice == 'right':
                    ros_joints.append(right_joint)
                    ros_claws.append([right_claw])
                else:  # both
                    ros_joints.append(left_joint + right_joint)
                    ros_claws.append([left_claw, right_claw])

            # 读取图像数据记录
            elif 'front' in data or 'image' in data:
                img_times.append(t)
                img_fronts.append(data.get('front', ''))
                img_hands.append(data.get('hand', ''))

    # 转换为 Numpy 数组
    ros_times = np.array(ros_times)
    ros_joints = np.array(ros_joints)
    ros_claws = np.array(ros_claws)
    img_times = np.array(img_times)

    unique_times, unique_indices = np.unique(ros_times, return_index=True)

    ros_times = unique_times
    ros_joints = ros_joints[unique_indices]
    ros_claws = ros_claws[unique_indices]
    # ==========================================================

    # 2. 插值算法平滑数据并向图像时间节点对齐
    # 对于连续的机械臂角度，使用三次样条插值 (cubic) 来保证速度和加速度的平滑
    joint_interp_func = interp1d(ros_times, ros_joints, axis=0, kind='cubic', fill_value="extrapolate")

    # 对于夹爪开闭 (0或1) 这种离散状态，使用最近邻插值 (nearest)
    claw_interp_func = interp1d(ros_times, ros_claws, axis=0, kind='nearest', fill_value="extrapolate")

    # 获取图像时间节点上对应的机械臂平滑数据
    aligned_joints = joint_interp_func(img_times)
    aligned_claws = claw_interp_func(img_times)

    # 3. 构建可视化时序分布图
    plt.figure(figsize=(14, 7))

    # 提取第一个关节的数据作为可视化代表
    joint_idx = 0
    plt.scatter(ros_times, ros_joints[:, joint_idx], label=f'Raw ROS Data (Joint {joint_idx})', color='blue', alpha=0.5,
                s=15)

    # 绘制三次样条拟合出的平滑曲线
    smooth_times = np.linspace(min(ros_times), max(ros_times), 1000)
    smooth_joints = joint_interp_func(smooth_times)
    plt.plot(smooth_times, smooth_joints[:, joint_idx], label='Fitted Spline Curve', color='cyan', alpha=0.7)

    # 绘制图像帧对应时间节点的采样点
    plt.scatter(img_times, aligned_joints[:, joint_idx], label='Image Timestamps (Target)', color='red', marker='x',
                s=50)

    # 用垂直虚线表示图像的接收时间，方便观察其相对于 ROS 通信点是否偏离严重
    for i, t in enumerate(img_times):
        # 仅为第一个画图例避免重复
        label = 'Camera Capture Time' if i == 0 else ""
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.2, label=label)

    plt.title(f'Temporal Alignment & Smoothing for Teleop Data ({arm_choice.upper()} Arm)')
    plt.xlabel('Host Relative Time (ms)')
    plt.ylabel('Joint Position (rad)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()  # 渲染可视化图表

    # 4. 输出矫正和对齐后的数据文件
    with open(output_file, 'w') as f:
        for i, t in enumerate(img_times):
            correct_data = {
                "time": int(t),
                "front_image": img_fronts[i],
                "hand_image": img_hands[i],
                "robot_state": {
                    "jointPos": aligned_joints[i].tolist(),
                    "clawStatus": [int(c) for c in aligned_claws[i]]
                }
            }
            f.write(json.dumps(correct_data) + '\n')

    print(f"[Success] Processed {len(img_times)} frames. Aligned data saved to {output_file}.")


# 运行代码：可以修改 arm_choice 为 'left', 'right', 或 'both'
root_dir = "./data_test"

for subdir, dirs, files in os.walk(root_dir):
    if "data_log.txt" in files:
        log_path = os.path.join(subdir, "data_log.txt")
        output_path = os.path.join(subdir, "data_log_correct.txt")
        print(f"\n[Processing] {log_path}")
        try:
            process_and_align_teleop_data(
                log_file=log_path,
                output_file=output_path,
                arm_choice='left'  # 可根据需要调整
            )
        except Exception as e:
            print(f"[Error] Failed to process {log_path}: {e}")
