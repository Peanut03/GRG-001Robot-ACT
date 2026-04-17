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


# --- 新增：统一归一化工具函数 ---
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

def make_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate ACT model with image sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--head_dir", type=str, required=True, help="Path to head camera images.")
    parser.add_argument("--hand_dir", type=str, required=True, help="Path to hand camera images.")
    return parser


def get_image_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image)


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
        print(f"Normalization Mode: {self.norm_stats.get('norm_mode', 'mean_std')}")

    @torch.no_grad()
    def predict(self, head_img_path, hand_img_path, qpos_raw):
        # 1. 处理图像
        head_tensor = get_image_tensor(head_img_path)
        hand_tensor = get_image_tensor(hand_img_path)
        images = torch.stack([head_tensor, hand_tensor], dim=0).unsqueeze(0).to(self.device)

        # 2. 处理 qpos
        qpos_np = np.array(qpos_raw)
        qpos_norm = normalize(qpos_np, self.norm_stats, "qpos")
        qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(self.device)
        if qpos_tensor.ndim == 3:
            qpos_tensor = qpos_tensor.squeeze(1)


        # 3. 推理
        # 注意：ACT 返回的是 action chunk (sequence)
        action_pred_norm = self.policy(qpos_tensor, images)
        action_pred_norm = action_pred_norm.cpu().numpy()[0]  # 取第0个batch

        # 4. 反归一化
        action_unnorm = denormalize(action_pred_norm, self.norm_stats, "action")

        # 5. 限制范围 (防止弧度溢出)
        if self.norm_stats.get("norm_mode") == "min_max":
            action_unnorm = np.clip(action_unnorm, self.norm_stats["action_min"], self.norm_stats["action_max"])

        return action_unnorm


def main():
    parser = make_parser()
    args = parser.parse_args()
    engine = ACTInference(args.checkpoint)

    head_files = sorted([f for f in os.listdir(args.head_dir) if f.endswith(('.png', '.jpg'))])
    hand_files = sorted([f for f in os.listdir(args.hand_dir) if f.endswith(('.png', '.jpg'))])
    num_frames = min(len(head_files), len(hand_files))

    # 初始 qpos 设置为 8 维
    current_qpos = [0.0] * 8

    for i in range(num_frames):
        hand_path = os.path.join(args.head_dir, head_files[i])
        head_path = os.path.join(args.hand_dir, hand_files[i])

        action_seq = engine.predict(head_path, hand_path, current_qpos)
        target_action = action_seq[0]

        print(f"Frame {i}: Predicted Joint States: [{', '.join([f'{x:.4f}' for x in target_action])}]")

        # 更新状态以便自回归测试
        current_qpos = target_action


if __name__ == '__main__':
    main()