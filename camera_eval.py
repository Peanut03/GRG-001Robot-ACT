import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse

# 保持原有引用路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from act_pytorch.policies.act_policy import ACTPolicy
from act_pytorch.utils.train_utils import set_seed


def make_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate ACT model with real-time camera input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--head_cam", type=int, default=0, help="Head camera device index (default: 0).")
    parser.add_argument("--hand_cam", type=int, default=1, help="Hand camera device index (default: 1).")
    parser.add_argument("--show_video", action="store_true", help="Show video feed from cameras.")
    return parser


def get_image_tensor_from_cv2(cv2_image):
    """从OpenCV图像转换单张图像为Tensor"""
    # 将BGR转换为RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # 转换为PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    # ACT通常期望图像被缩放到特定的尺寸，这里根据你要求的 640x480 处理
    preprocess = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 常用ImageNet归一化
    ])
    return preprocess(pil_image)


class ACTInference:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载权重
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.train_args = ckpt["args"]
        self.norm_stats = ckpt["norm_stats"]

        # 实例化 Policy
        set_seed(self.train_args.seed)
        self.policy = ACTPolicy(self.train_args).to(self.device)
        self.policy.model.load_state_dict(ckpt["model"])
        self.policy.eval()
        print(f"Model loaded from {checkpoint_path}")

    def normalize_qpos(self, qpos_raw):
        """根据训练时的归一化模式对qpos进行归一化"""
        qpos_np = np.array(qpos_raw)
        
        # 获取归一化模式，默认为"mean_std"
        norm_mode = self.norm_stats.get("norm_mode", "mean_std")
        
        if norm_mode == "min_max":
            qpos_min = self.norm_stats["qpos_min"]
            qpos_max = self.norm_stats["qpos_max"]
            # Min-Max归一化: (x - min) / (max - min)
            qpos_norm = (qpos_np - qpos_min) / (qpos_max - qpos_min + 1e-8)
        else:  # 默认使用mean_std
            qpos_mean = self.norm_stats["qpos_mean"]
            qpos_std = self.norm_stats["qpos_std"]
            # Mean-Std归一化: (x - mean) / std
            qpos_norm = (qpos_np - qpos_mean) / qpos_std
            
        return qpos_norm

    def unnormalize_action(self, action_pred):
        """根据训练时的归一化模式对动作进行反归一化"""
        # 获取归一化模式，默认为"mean_std"
        norm_mode = self.norm_stats.get("norm_mode", "mean_std")
        
        if norm_mode == "min_max":
            action_min = self.norm_stats["action_min"]
            action_max = self.norm_stats["action_max"]
            # Min-Max反归一化: x * (max - min) + min
            action_unnorm = action_pred * (action_max - action_min) + action_min
        else:  # 默认使用mean_std
            action_mean = self.norm_stats["action_mean"]
            action_std = self.norm_stats["action_std"]
            # Mean-Std反归一化: x * std + mean
            action_unnorm = action_pred * action_std + action_mean
            
        return action_unnorm

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

        # 2. 处理 qpos - 使用新的归一化方法
        qpos_norm = self.normalize_qpos(qpos_raw)
        qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(self.device)

        if qpos_tensor.ndim == 3:
            qpos_tensor = qpos_tensor.squeeze(1)  # [1, 1, 7] ->

        # 3. 推理
        # 注意：ACT模型返回的是一个动作序列 (batch, sequence_length, action_dim)
        action_pred = self.policy(qpos_tensor, images)
        action_pred = action_pred.cpu().numpy()[0]  # 取出第0个batch

        # 4. 反归一化动作 - 使用新的反归一化方法
        action_unnorm = self.unnormalize_action(action_pred)

        return action_unnorm


def main():
    parser = make_parser()
    args = parser.parse_args()

    # 初始化推理引擎
    engine = ACTInference(args.checkpoint)

    # 初始化摄像头
    print(f"Opening head camera (device {args.head_cam})...")
    head_cap = cv2.VideoCapture(args.head_cam)
    print(f"Opening hand camera (device {args.hand_cam})...")
    hand_cap = cv2.VideoCapture(args.hand_cam)
    
    if not head_cap.isOpened():
        print(f"Error: Could not open head camera (device {args.head_cam})!")
        return
    if not hand_cap.isOpened():
        print(f"Error: Could not open hand camera (device {args.hand_cam})!")
        head_cap.release()
        return

    print("Cameras opened successfully!")
    print("Press 'q' to quit.")

    # 假设这里有一个初始的qpos（根据你的机器人实际状态获取）
    # 这里先用随机值占位，实际应用中应从传感器读取
    current_qpos = [0.0] * 8

    frame_count = 0
    while True:
        # 从两个摄像头读取图像
        ret1, head_frame = head_cap.read()
        ret2, hand_frame = hand_cap.read()
        
        if not ret1 or not ret2:
            print("Error: Could not read frames from one or both cameras!")
            break
        
        # 执行推理
        action_seq = engine.predict(head_frame, hand_frame, current_qpos)
        
        # 通常我们取预测序列的第一帧作为当前执行动作
        target_action = action_seq[0]
        
        print(f"Frame {frame_count}: Predicted Action (first step): {target_action}")
        
        # 在真实的闭环控制中，你会把 target_action 发给机器人，
        # 然后更新 current_qpos 为机器人当前真实的位置
        current_qpos = target_action
        
        frame_count += 1
        
        # 显示图像（如果启用）
        if args.show_video:
            cv2.imshow('Head Camera', head_frame)
            cv2.imshow('Hand Camera', hand_frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    head_cap.release()
    hand_cap.release()
    if args.show_video:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()