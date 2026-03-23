#!/bin/bash

# 配置路径
MATLAB_SCRIPT_DIR="/mnt/mydrive/traffic_twin/waypoint_control/multi_obj_track"
MATLAB_SCRIPT_NAME="demo"
LOG_FILE="matlab_output.log"

# 定义触发条件
TRIGGER_FILE="$MATLAB_SCRIPT_DIR/start_signal.txt"

# 切换目录
cd "$MATLAB_SCRIPT_DIR" || { echo "切换目录失败！"; exit 1; }

echo "========================================"
echo " 监听模式已启动..."
echo "正在等待触发条件满足 (按 Ctrl+C 退出)..."
echo "========================================"

# 开启条件判定循环
while true; do
    
    # 判断 TRIGGER_FILE 是否存在 
    if [ -f "$TRIGGER_FILE" ]; then
        
        echo ""
        echo " [$(date)] 触发条件已满足！开始执行 MATLAB 脚本..."
        
        # 执行 MATLAB 代码
        matlab -nodisplay -nosplash -logfile "$LOG_FILE" -batch "$MATLAB_SCRIPT_NAME"
        
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo " 执行成功！"
        else
            echo " 执行失败！(退出码: $EXIT_CODE)"
        fi

        # 重置触发条件
        rm -f "$TRIGGER_FILE"
        
        echo " 继续监听中..."
        
    else
        # 如果条件不满足，让脚本睡 1 秒钟再查。
        sleep 1
    fi

done