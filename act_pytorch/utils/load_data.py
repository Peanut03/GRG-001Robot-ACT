import os
import h5py
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
from act_pytorch.utils.train_utils import get_norm_stats



class ACTDataset(Dataset):
    def __init__(self, args, norm_stats):
        super().__init__()
        self.num_queries = args.action_horizon
        self.dataset_dir = args.dataset_dir
        self.camera_names = args.cameras
        self.norm_stats = norm_stats
        self.full_episode = args.full_episode
        file_paths = os.path.join(self.dataset_dir, '*.h5')
        self.file_paths = glob(file_paths)

    def __len__(self):
        return len(self.file_paths)

    def normalize(self, data, stats, prefix):
        mode = stats.get("norm_mode", "mean_std")  # 默认回退到 mean_std
        if mode == "min_max":
            # 防止除以 0，使用 1e-5 或已在 get_norm_stats 中做过的 clipping
            denom = stats[f"{prefix}_max"] - stats[f"{prefix}_min"]
            denom = np.clip(denom, 1e-2, None)
            return (data - stats[f"{prefix}_min"]) / denom
        elif mode == "mixed":
            # 混合归一化：前7个自由度使用meanstd，最后一个自由度使用minmax
            if len(data.shape) == 1:
                # 处理一维数据（如qpos）
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
                # 处理二维数据（如action序列）
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
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with h5py.File(path, 'r') as f:
            action = f['/action'][:]  # (time_steps, action_dim)
            time_steps = action.shape[0]
            if self.full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(time_steps)
            qpos = f['/observations/qpos'][:][start_ts]  # (pos_dim,)
            images = []
            for cam_name in self.camera_names:
                images.append(f[f'/observations/images/{cam_name}'][:][start_ts])        
            # concatenate images
            image = np.stack(images, axis=0)  # (num_camera, h, w, c)
            # normalize actions and joint positions
            action = self.normalize(action, self.norm_stats, "action").squeeze()
            qpos = self.normalize(qpos, self.norm_stats, "qpos").squeeze()
            # action sequence zero padding
            zero_pad = np.zeros((self.num_queries - 1, action.shape[1]), dtype=np.float32)
            action = np.concatenate([action, zero_pad], axis=0)
            is_pad = np.zeros(action.shape[0])
            is_pad[time_steps: ] = 1  # define where sequences of zero padding are
            # get action chunkings
            action_seq = action[start_ts: (start_ts + self.num_queries)]
            is_pad = is_pad[start_ts: (start_ts + self.num_queries)]
            # transform nd.array to torch.tensor
            image = torch.from_numpy(image).permute(0, 3, 1, 2)  # (num_camera, c, h, w)
            qpos = torch.from_numpy(qpos).float()  # (pos_dim,)
            action_seq = torch.from_numpy(action_seq).float()  # (num_queries, action_dim)
            is_pad = torch.from_numpy(is_pad).bool()  # (num_queries,)
            # normalize images pixel intensity to [0, 1] (if necessary)
            image = image / 255.0
            return image, qpos, action_seq, is_pad
                        

def load_data(args):
    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(args)
    # Construct dataset and dataloader
    dataset = ACTDataset(args, norm_stats)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=1
    )
    return dataloader, norm_stats