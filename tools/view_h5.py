import h5py
import numpy as np
import sys

def print_h5_structure(name, obj):
    """递归打印 HDF5 文件结构"""
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}")
        print(f"  Shape: {obj.shape}")
        print(f"  Dtype: {obj.dtype}")
        # 可选：打印前几个元素（避免大数组）
        if obj.size < 100:  # 小数组直接打印
            print(f"  Data: {np.array(obj)}")
        else:
            print(f"  Sample (first 5 elements): {np.array(obj).flat[:5]}...")
        print()
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")
        print()

def view_h5_file(file_path):
    """查看 HDF5 文件内容"""
    print(f"🔍 Viewing HDF5 file: {file_path}\n")
    with h5py.File(file_path, 'r') as f:
        f.visititems(print_h5_structure)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python view_h5.py <path_to_file.h5>")
        sys.exit(1)
    
    h5_file = sys.argv[1]
    if not h5_file.endswith('.h5') and not h5_file.endswith('.hdf5'):
        print("Warning: File does not have .h5 or .hdf5 extension.")
    
    try:
        view_h5_file(h5_file)
    except Exception as e:
        print(f"❌ Error reading file: {e}")