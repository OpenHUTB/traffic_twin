import os
import glob
import numpy as np
from collections import defaultdict

# 文件夹路径
label_folder = './data/custom/labels'

# 用于按类别存储尺寸
dimensions = defaultdict(lambda: {'x': [], 'y': [], 'z': []})

# 遍历所有 txt 文件
filepaths = glob.glob(os.path.join(label_folder, '*.txt'))
print(f"共找到 {len(filepaths)} 个文件，开始统计...")

for filepath in filepaths:
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            # 确保行数据完整
            if len(parts) >= 8:
                try:
                    # 获取第 4, 5, 6 位数据
                    dx = float(parts[3])
                    dy = float(parts[4])
                    dz = float(parts[5])
                    obj_class = parts[-1] # 最后一列是类别名称

                    dimensions[obj_class]['x'].append(dx)
                    dimensions[obj_class]['y'].append(dy)
                    dimensions[obj_class]['z'].append(dz)
                except ValueError:
                    # 跳过无法转换为数字的异常行
                    continue

# 2. 打印最终结果
print("-" * 30)
for obj_class, dims in dimensions.items():
    if len(dims['x']) > 0:
        avg_x = np.mean(dims['x'])
        avg_y = np.mean(dims['y'])
        avg_z = np.mean(dims['z'])
        print(f"类别: {obj_class} | 总样本数: {len(dims['x'])}")
        print(f"请将 OpenPCDet 中 {obj_class} 的 anchor_sizes 修改为: [[{avg_x:.2f}, {avg_y:.2f}, {avg_z:.2f}]]\n")