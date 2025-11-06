import matplotlib.pyplot as plt
import numpy as np

def plot_results(cx, cy, path_x, path_y, throttle_history, steer_history, brake_history):
    """
    绘制仿真结果，包括路径对比、油门/转向/制动控制历史图
    
    参数:
        cx, cy: 参考路径坐标
        path_x, path_y: 实际车辆行驶路径坐标
        throttle_history: 油门控制历史
        steer_history: 转向控制历史
        brake_history: 制动控制历史
    """
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    
    # 路径对比图
    axs[0].plot(cx, cy, 'r-', label='参考路径')
    axs[0].plot(path_x, path_y, 'b-', label='实际路径')
    axs[0].set_title('路径对比')
    axs[0].set_xlabel('X坐标 (m)')
    axs[0].set_ylabel('Y坐标 (m)')
    axs[0].grid(True)
    axs[0].legend()
    
    # 油门历史图
    axs[1].plot(throttle_history, label='油门', color='g')
    axs[1].set_title('油门控制历史')
    axs[1].set_xlabel('时间步')
    axs[1].set_ylabel('油门值 (0-1)')
    axs[1].grid(True)
    
    # 转向历史图
    axs[2].plot(steer_history, label='转向', color='b')
    axs[2].set_title('转向控制历史')
    axs[2].set_xlabel('时间步')
    axs[2].set_ylabel('转向值 (-1 至 1)')
    axs[2].grid(True)
    
    # 制动历史图
    axs[3].plot(brake_history, label='制动', color='r')
    axs[3].set_title('制动控制历史')
    axs[3].set_xlabel('时间步')
    axs[3].set_ylabel('制动值 (0-1)')
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.savefig('mpc_control_results.png')
    print("结果已保存到 mpc_control_results.png")
    plt.show()

def plot_tracking_error(cx, cy, path_x, path_y):
    """
    计算并绘制跟踪误差
    
    参数:
        cx, cy: 参考路径坐标
        path_x, path_y: 实际车辆行驶路径坐标
    """
    # 确保两个路径具有相同的长度进行比较
    min_len = min(len(cx), len(path_x))
    errors = []
    
    for i in range(min_len):
        # 计算欧几里得距离作为跟踪误差
        error = np.sqrt((cx[i] - path_x[i])**2 + (cy[i] - path_y[i])**2)
        errors.append(error)
    
    plt.figure(figsize=(10, 6))
    plt.plot(errors, 'r-')
    plt.title('路径跟踪误差')
    plt.xlabel('时间步')
    plt.ylabel('跟踪误差 (m)')
    plt.grid(True)
    plt.savefig('tracking_error.png')
    print("跟踪误差已保存到 tracking_error.png")
    plt.show()
    
    # 计算统计信息
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    std_error = np.std(errors)
    
    print(f"平均跟踪误差: {avg_error:.3f} m")
    print(f"最大跟踪误差: {max_error:.3f} m")
    print(f"跟踪误差标准差: {std_error:.3f} m") 