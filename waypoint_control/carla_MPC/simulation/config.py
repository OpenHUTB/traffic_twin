import numpy as np
import math

# 状态空间模型参数
NX = 4  # 状态维度: x, y, v, yaw
NU = 2  # 控制输入维度: 加速度, 转向角
T = 5   # 预测时域长度

# 车辆限制
STOP_SPEED = 0.5 / 3.6  # 停止速度
MAX_ITER = 3  # 最大迭代次数
DU_TH = 0.1  # 迭代完成阈值
TARGET_SPEED = 10.0 / 3.6  # 目标速度 [m/s]
N_IND_SEARCH = 10  # 搜索索引数量
DT = 0.2  # 时间步长 [s]
MAX_STEER = np.deg2rad(45.0)  # 最大转向角 [rad]
MAX_SPEED = 55.0 / 3.6  # 最大速度 [m/s]
MIN_SPEED = 0  # 最小速度 [m/s]
MAX_ACCEL = 2.0  # 最大加速度 [m/s²]

# 车辆参数（在运行时由车辆尺寸确定）
WB = 2.0  # 初始化轮距值，会在setup_environment中根据实际车辆调整 