#!/bin/bash

# CARLA 的启动文件路径
CARLA_SH_PATH="/home/yons/Carla_0.9.16/CarlaUE4.sh" 

# Python 脚本所在的文件夹路径
PYTHON_DIR="/mnt/mydrive/traffic_twin/waypoint_control/multi_obj_track"

#  一键清理机制
PIDS=()
cleanup() {
    echo -e "\n 正在一键清理所有后台的 CARLA 实例和 Python 脚本..."
    for pid in "${PIDS[@]}"; do
        kill -9 $pid 2>/dev/null 
    done
    echo " 所有进程已清理干净，端口已释放！"
    exit 0
}
trap cleanup SIGINT SIGTERM

# carla所用到的端口：
PORTS=(2000 2020 2040 2060 2080)


for i in {1..5}; do
    # 获取数组中的对应端口 (数组索引从0开始)
    INDEX=$((i-1))
    CURRENT_PORT=${PORTS[$INDEX]}

    echo "--------------------------------------------------"
    echo "▶ 正在启动第 $i 组 (使用端口: $CURRENT_PORT)..."

    # 启动 CARLA 并指定与 Python 脚本匹配的端口
    bash "$CARLA_SH_PATH" -RenderOffScreen -rpc_port=$CURRENT_PORT &
    PIDS+=($!)

    # 给 CARLA 引擎留出 6 秒钟的初始化时间
    sleep 6 

    # 启动 Python 脚本
    python3 "$PYTHON_DIR/collect_intersection_camera_lidar-${i}.py" &
    PIDS+=($!)
    
    # 脚本启动缓冲
    sleep 2
done

echo "--------------------------------------------------"
echo " 仿真已全部启动！"
echo " 想结束测试时，请直接在此窗口按 [Ctrl + C]。"

wait