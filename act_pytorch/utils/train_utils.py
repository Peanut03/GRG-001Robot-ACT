import os
import h5py
import torch
import numpy as np
from glob import glob

import IPython
e = IPython.embed

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class Logger:
    """Information logger"""
    def __init__(self, path: str):
        self.root = open(path, 'a')

    def dump(self, info: str):
        print(info)
        self.root.write(info + '\n')
        self.root.flush()

    def close(self):
        self.root.close()


def get_norm_stats(args):
    all_qpos_data = []
    all_action_data = []
    file_paths = glob(os.path.join(args.dataset_dir, '*.h5'))
    for path in file_paths:
        with h5py.File(path, 'r') as f:
            qpos = f['/observations/qpos'][:]
            action = f['/action'][:]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)

    # 1. 计算 Mean 和 Std
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, torch.inf)

    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, torch.inf)

    # 2. 计算 Min 和 Max (新增内容)
    action_min = all_action_data.min(dim=0, keepdim=True).values
    action_max = all_action_data.max(dim=0, keepdim=True).values

    qpos_min = all_qpos_data.min(dim=0, keepdim=True).values
    qpos_max = all_qpos_data.max(dim=0, keepdim=True).values

    # 将所有信息和模式一同打包
    stats = {
        "action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
        "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
        "action_min": action_min.numpy(), "action_max": action_max.numpy(),
        "qpos_min": qpos_min.numpy(), "qpos_max": qpos_max.numpy(),
        "norm_mode": args.norm_mode  # 将训练配置注入 stats
    }

    return stats