import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
import argparse
import torch
from act_pytorch.policies.act_policy import ACTPolicy
from act_pytorch.utils.train_utils import set_seed
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate the model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint path."
    )
    return parser


@torch.no_grad()
def test(checkpoint: str, image: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
    torch.cuda.empty_cache()
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load checkpoint
    ckpt = torch.load(checkpoint, map_location=device)
    train_args = ckpt["args"]
    # set seed
    set_seed(train_args.seed)
    # instantiate policy
    policy = ACTPolicy(train_args).to(device)
    policy.model.load_state_dict(ckpt["model"])
    # get norm status
    norm_stats = ckpt["norm_stats"]
    # normalize qpos - 支持多种归一化模式
    mode = norm_stats.get("norm_mode", "mean_std")
    qpos_np = qpos.cpu().numpy()
    if len(qpos_np.shape) == 2:
        if mode == "min_max":
            denom = norm_stats["qpos_max"] - norm_stats["qpos_min"]
            denom = np.clip(denom, 1e-2, None)
            normalized_qpos = (qpos_np - norm_stats["qpos_min"]) / denom
        elif mode == "mixed":
            normalized_qpos = np.zeros_like(qpos_np, dtype=np.float32)
            # 前7个维度使用meanstd
            for i in range(min(7, qpos_np.shape[-1])):
                normalized_qpos[:, i] = (qpos_np[:, i] - norm_stats["qpos_mean"][0, i]) / norm_stats["qpos_std"][0, i]
            # 最后一个维度使用minmax（如果存在）
            if qpos_np.shape[-1] > 7:
                denom = norm_stats["qpos_max"][0, 7] - norm_stats["qpos_min"][0, 7]
                denom = np.clip(denom, 1e-2, None)
                normalized_qpos[:, 7] = (qpos_np[:, 7] - norm_stats["qpos_min"][0, 7]) / denom
        else:  # mean_std
            normalized_qpos = (qpos_np - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
        qpos = torch.from_numpy(normalized_qpos).to(device)
    else:
        # 处理其他形状的情况
        qpos = qpos.to(device)
    
    # inference
    policy.eval()
    image, qpos = image.to(device), qpos
    action_pred = policy(qpos, image)
    action_pred = action_pred.cpu()
    # unnormalize actions - 支持多种反归一化模式
    action_pred_np = action_pred.numpy()
    if len(action_pred_np.shape) == 3:
        if mode == "min_max":
            denom = norm_stats["action_max"] - norm_stats["action_min"]
            denom = np.clip(denom, 1e-2, None)
            denormalized_action = action_pred_np * denom + norm_stats["action_min"]
        elif mode == "mixed":
            denormalized_action = np.zeros_like(action_pred_np, dtype=np.float32)
            # 前7个维度使用meanstd反归一化
            for i in range(min(7, action_pred_np.shape[-1])):
                denormalized_action[:, :, i] = action_pred_np[:, :, i] * norm_stats["action_std"][0, i] + norm_stats["action_mean"][0, i]
            # 最后一个维度使用minmax反归一化（如果存在）
            if action_pred_np.shape[-1] > 7:
                denom = norm_stats["action_max"][0, 7] - norm_stats["action_min"][0, 7]
                denom = np.clip(denom, 1e-2, None)
                denormalized_action[:, :, 7] = action_pred_np[:, :, 7] * denom + norm_stats["action_min"][0, 7]
        else:  # mean_std
            denormalized_action = action_pred_np * norm_stats["action_std"] + norm_stats["action_mean"]
        action_pred = torch.from_numpy(denormalized_action)
    else:
        # 处理其他形状的情况
        pass
    
    return action_pred


def main(argv=sys.argv[1:]):
    parser = make_parser()
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint
    image = torch.rand(1, 1, 3, 480, 640)
    qpos = torch.rand(1, 8)  # 修改为8维
    action_pred = test(checkpoint, image, qpos)
    
    
if __name__ == '__main__':
    main()