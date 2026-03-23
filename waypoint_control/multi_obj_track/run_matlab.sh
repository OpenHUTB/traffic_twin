#!/bin/bash

# MATLAB 脚本名字 

MATLAB_SCRIPT="demo"

# 2. MATLAB 可执行文件的路径
# 如果你在终端直接输入 matlab 能打开，这里保持 "matlab" 即可。
# 如果不行，请改成绝对路径，例如 "/usr/local/MATLAB/R2023a/bin/matlab"
MATLAB_CMD="/home/yons/matlab/2024B/bin/matlab"

# 3. 日志文件名 (运行结果会保存在这个文件里，方便你以后查看)
LOG_FILE="matlab_output.log"

# ==========================================
# 🚀 执行区 (以下代码不需要修改)
# ==========================================

# 获取当前 .sh 脚本所在的绝对路径，并切换过去
# (这一步非常关键！防止你在其他文件夹运行这个脚本时，MATLAB 找不到相对路径下的数据文件)
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR" || exit 1

echo "========================================"
echo "▶ 开始时间: $(date)"
echo "▶ 工作目录: $WORK_DIR"
echo "▶ 运行脚本: ${MATLAB_SCRIPT}.m"
echo "========================================"
echo "正在后台调用 MATLAB，请稍候..."

# 组合 MATLAB 运行命令
# -nodisplay: 不加载图形界面 (纯命令行运行)
# -nosplash: 不显示启动时的 MATLAB Logo
# -logfile: 将所有终端输出保存到日志文件中
# -batch: 执行代码并在完成后自动退出 MATLAB
$MATLAB_CMD -nodisplay -nosplash -logfile "$LOG_FILE" -batch "$MATLAB_SCRIPT"

# 检查上一条命令 (MATLAB) 的退出状态码
EXIT_CODE=$?

echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 运行成功！"
    echo "结束时间: $(date)"
    echo "详细输出已保存至: $WORK_DIR/$LOG_FILE"
else
    echo "❌ 运行失败！(退出码: $EXIT_CODE)"
    echo "请检查日志文件排查错误: $WORK_DIR/$LOG_FILE"
    exit 1
fi
echo "========================================"