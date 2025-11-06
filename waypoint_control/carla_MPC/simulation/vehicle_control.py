import time
import math
import numpy as np
import carla
import sys
from simulation import config
import simulation.mpc_utils as mpc_utils

# 状态类，用于表示车辆状态
class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

# 计算最近参考点索引
def calc_nearest_index(state, cx, cy, cyaw, pind, N_IND_SEARCH=10):
    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)
    ind = d.index(mind) + pind
    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = mpc_utils.pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

# 自适应MPC矩阵计算
def adaptive_mpc_matrices(state, cx, cy, cyaw, nearest_ind, target_ind, tracking_error=0, enable_dynamic=True, debug_output=False):
    # 基本权重矩阵
    Q = np.diag([3.0, 3.0, 0.5, 1.0])  # 状态误差权重 [x, y, v, yaw]
    R = np.diag([0.01, 0.1])  # 控制输入权重 [加速度, 转向]
    Rd = np.diag([0.01, 1.0])  # 控制变化率权重 [加速度变化, 转向变化]
    Qf = Q * 5.0  # 终端状态权重
    
    if enable_dynamic:
        # 根据跟踪误差动态调整权重
        if tracking_error > 1.0:
            # 当跟踪误差大时，增加位置权重
            Q[0, 0] = 5.0 + tracking_error 
            Q[1, 1] = 5.0 + tracking_error
            Qf[0, 0] = 10.0 + tracking_error * 2
            Qf[1, 1] = 10.0 + tracking_error * 2
        
        # 在接近目标时增加速度控制权重
        if target_ind > len(cx) - 30:
            Q[2, 2] = 1.0  # 增加速度权重
            Qf[2, 2] = 5.0
    
    if debug_output:
        print(f"Adaptive MPC - Tracking error: {tracking_error:.3f}")
        print(f"Position weights (Q): x={Q[0,0]:.2f}, y={Q[1,1]:.2f}")
    
    return Q, Qf, R, Rd

def run_mpc_loop(world, map, ego, cx, cy, cyaw, sp):
    ego_transform = ego.get_transform()
    WB = ego.bounding_box.extent.y * 2 - 0.2

    state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)
    print("Starting vehicle with initial thrust...")
    odelta, oa = None, None

    path_x, path_y = [], []
    throttle_history, steer_history, brake_history = [], [], []
    spectator = world.get_spectator()
    # 对于初始情况考虑适当吧速度调高一点
    for _ in range(10):
        ego.apply_control(carla.VehicleControl(throttle=0.7, steer=0.0))
        time.sleep(0.1)
        world.wait_for_tick()
    # Check if vehicle has started moving
    v = ego.get_velocity()
    speed = v.length()
    print(f"Vehicle started with initial speed: {speed:.2f} m/s")
    
    # Apply stronger acceleration if needed
    if speed < 0.1:
        print("WARNING: Vehicle didn't start moving. Applying stronger thrust...")
        for i in range(5):
            ego.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0))
            time.sleep(0.2)
            world.wait_for_tick()
        v = ego.get_velocity()
        speed = v.length()
        print(f"Second attempt speed: {speed:.2f} m/s")
    
    control_iteration = 0
    
    while True:
        control_iteration += 1
        transform = ego.get_transform()
        velocity = ego.get_velocity()
        speed = velocity.length()
        state.x, state.y = transform.location.x, transform.location.y
        state.yaw = math.radians(transform.rotation.yaw)
        state.v = speed

        spectator.set_transform(carla.Transform(
            transform.location + carla.Location(z=60),
            carla.Rotation(pitch=-90)
        ))

        nearest_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)
        tracking_error = np.linalg.norm([state.x - cx[nearest_ind], state.y - cy[nearest_ind]])

        Q, Qf, R, Rd = adaptive_mpc_matrices(
            state, cx, cy, cyaw, 
            nearest_ind, target_ind, 
            tracking_error=tracking_error,
            enable_dynamic=False,  # 设置为False可以禁用动态调整
            debug_output=(control_iteration % 20 == 0)  # 每20次迭代输出一次调试信息
        )

        if target_ind < nearest_ind:
            target_ind = nearest_ind

        x0 = [state.x, state.y, state.v, state.yaw]
        xref, target_ind, dref = mpc_utils.calc_ref_trajectory(state, cx, cy, cyaw, sp, 2.0, target_ind)

        oa, odelta, ox, oy, oyaw, ov = mpc_utils.iterative_linear_mpc_control(xref, x0, dref, oa, odelta)
        
        # Extract control commands or use fallback values if optimization fails
        if oa is not None and odelta is not None:
            a, delta = oa[0], odelta[0]
        else:
            print("WARNING: MPC solver failed to find solution")
            a, delta = 0.5, 0.0  # fallback control values

        # Convert MPC outputs to CARLA control inputs
        if a >= 0.0:
            throttle = min(max(a / config.MAX_ACCEL, 0.2), 1.0)  # maintain minimum throttle
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(max(-a / config.MAX_ACCEL, 0.0), 1.0)

        steer = min(max(delta / config.MAX_STEER, -1.0), 1.0)

        path_x.append(state.x)
        path_y.append(state.y)
        throttle_history.append(throttle)
        steer_history.append(steer)
        brake_history.append(brake)

        ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

        # Visualize predicted trajectory
        if ox is not None and oy is not None:
            for i in range(len(ox)):
                world.debug.draw_point(
                    carla.Location(x=ox[i], y=oy[i], z=ego.get_location().z+0.5),
                    size=0.1,
                    color=carla.Color(r=0, g=0, b=255),
                    life_time=0.2
                )

        if target_ind >= len(cx) - 1:
            print("Destination reached!")
            break

        time.sleep(0.1)
        world.wait_for_tick()

    return path_x, path_y, throttle_history, steer_history, brake_history 