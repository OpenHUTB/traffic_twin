#!/bin/bash
echo "========== 开始执行自动化目标检测 =========="

# 环境变量设置
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openpcdet

# 先进入工作目录
WORK_DIR="/home/yons/traffic_twin/waypoint_control/multi_obj_track/OpenPCDet"
cd $WORK_DIR

# 运行 Python 脚本
python -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml
python tools/test.py --cfg_file output/cfgs/custom_models/pv_rcnn/default/pv_rcnn.yaml  --ckpt latest_model.pth

echo "========== 完成自动化目标检测 =========="