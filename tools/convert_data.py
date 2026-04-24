import os
import json
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_embodiment_data_to_hdf5(input_dir, output_dir):
    """
    Convert embodiment data from JSON format to HDF5 format for ACT training.
    
    Args:
        input_dir: Directory containing embodiment_data_* folders
        output_dir: Directory to save converted .h5 files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all embodiment data directories
    data_dirs = [d for d in os.listdir(input_dir) 
                 if d.startswith('embodiment_data_') and os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"Found {len(data_dirs)} data directories")
    
    for data_dir in tqdm(data_dirs, desc="Converting data"):
        full_path = os.path.join(input_dir, data_dir)
        json_file = os.path.join(full_path, 'data_log_correct.txt')
        front_dir = os.path.join(full_path, 'front')
        hand_dir = os.path.join(full_path, 'hand')
        
        if not os.path.exists(json_file):
            print(f"Warning: {json_file} not found, skipping...")
            continue
            
        # Read JSON data
        with open(json_file, 'r') as f:
            lines = f.readlines()
        
        # Parse JSON data
        parsed_data = []
        for line in lines:
            try:
                data = json.loads(line.strip())
                parsed_data.append(data)
            except json.JSONDecodeError:
                continue
        
        if not parsed_data:
            print(f"Warning: No valid data in {json_file}, skipping...")
            continue
        
        # Sort by time to ensure correct order
        parsed_data.sort(key=lambda x: x['time'])
        
        # Extract actions (joint positions + claw status)
        actions = []
        qpos_list = []  # qpos should be the same as actions for each timestep
        front_images = []
        hand_images = []
        
        for data in parsed_data:
            # Extract 7 joint positions + 1 claw status = 8 dimensions
            joint_pos = data['robot_state']['jointPos']
            claw_status = data['robot_state']['clawStatus'][0]  # clawStatus is a list with one element
            claw_status = 3.14159265 if claw_status == 1 else -3.14159265


            action = joint_pos + [claw_status]
            actions.append(action)
            qpos_list.append(action)  # qpos is the same as action at each timestep
            
            # Load images
            front_img_path = os.path.join(front_dir, data['front_image'].replace('front_', '').replace('.jpg', '.jpg'))
            hand_img_path = os.path.join(hand_dir, data['hand_image'].replace('hand_', '').replace('.jpg', '.jpg'))
            
            # Some entries might have different naming, try alternative
            if not os.path.exists(front_img_path):
                front_img_path = os.path.join(front_dir, data['front_image'])
            if not os.path.exists(hand_img_path):
                hand_img_path = os.path.join(hand_dir, data['hand_image'])
            
            if os.path.exists(front_img_path):
                front_img = np.array(Image.open(front_img_path))
                front_images.append(front_img)
            else:
                print(f"Warning: Front image {front_img_path} not found")
                continue
                
            if os.path.exists(hand_img_path):
                hand_img = np.array(Image.open(hand_img_path))
                hand_images.append(hand_img)
            else:
                print(f"Warning: Hand image {hand_img_path} not found")
                continue
        
        if len(actions) == 0:
            continue
            
        # Convert to numpy arrays
        actions = np.array(actions, dtype=np.float32)  # (time_steps, 8)
        qpos_array = np.array(qpos_list, dtype=np.float32)  # (time_steps, 8)
        front_images = np.array(front_images, dtype=np.uint8)  # (time_steps, H, W, C)
        hand_images = np.array(hand_images, dtype=np.uint8)  # (time_steps, H, W, C)
        
        # 删除action前5帧
        if len(actions) > 5:
            actions = actions[5:]
        elif len(actions) > 0:
            # 如果序列长度小于等于5，保留至少1帧以避免空数组
            actions = actions[-1:]  # 保留最后一帧
        
        # 删除qpos最后5帧
        if len(qpos_array) > 5:
            qpos_array = qpos_array[:-5]
        elif len(qpos_array) > 0:
            # 如果序列长度小于等于5，保留至少1帧以避免空数组
            qpos_array = qpos_array[:1]  # 保留第一帧
            
        # 删除图像最后5帧
        if len(front_images) > 5 and len(hand_images) > 5:
            front_images = front_images[:-5]
            hand_images = hand_images[:-5]
        elif len(front_images) > 0 and len(hand_images) > 0:
            # 如果序列长度小于等于5，保留至少1帧以避免空数组
            front_images = front_images[:1]
            hand_images = hand_images[:1]
        
        # Save to HDF5
        output_file = os.path.join(output_dir, f"{data_dir}.h5")
        with h5py.File(output_file, 'w') as f:
            # Save actions
            f.create_dataset('action', data=actions, dtype=np.float32)
            
            # Save observations
            obs_group = f.create_group('observations')
            obs_group.create_dataset('qpos', data=qpos_array, dtype=np.float32)
            
            # Save images
            img_group = obs_group.create_group('images')
            img_group.create_dataset('front_camera', data=front_images, dtype=np.uint8)
            img_group.create_dataset('hand_camera', data=hand_images, dtype=np.uint8)
        
        print(f"Converted {data_dir} to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert embodiment data to HDF5 format")
    parser.add_argument("--input_dir", type=str, default="data_new", 
                        help="Input directory containing embodiment data")
    parser.add_argument("--output_dir", type=str, default="converted_data", 
                        help="Output directory for HDF5 files")
    args = parser.parse_args()
    
    convert_embodiment_data_to_hdf5(args.input_dir, args.output_dir)
    