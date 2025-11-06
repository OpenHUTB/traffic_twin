import numpy as np
import matplotlib.pyplot as plt
from src.mcp_controller import Vehicle
from env import Env, draw_waypoints
from collections import Counter
import sys, pathlib
import carla
sys.path.insert(0, str(pathlib.Path(__file__).with_name('src')))
from src.x_v2x_agent import Xagent, RoadOption
from src.global_route_planner import GlobalRoutePlanner

import time
import pygame

class DummyWaypoint:
    def __init__(self, x, y, z=0.5):
        self.transform = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation())


# Simulation parameters
simu_step = 0.05  # Time step per simulation step (seconds)
target_v = 40  # Target speed (km/h)
sample_res = 2.0  # Sampling resolution for path planning
display_mode = "spec"  # Options: "spec" or "pygame"

env = Env(display_method=display_mode, dt=simu_step)
env.clean()
spawn_points = env.map.get_spawn_points()

start_idx, end_idx = 87, 70  # Indices for start and end points

filename = "Waypoints.txt"  # 替换成你的文件名
data = np.loadtxt(filename)
first_col = data[:, 0]
counts = Counter(first_col)
most_common_val = counts.most_common(1)[0][0]  # 出现次数最多的数字

# 取该数字对应的行
filtered = data[first_col == most_common_val]

# 提取第二列和第三列
custom_trajectory = filtered[:, 1:3]

# grp = GlobalRoutePlanner(env.map, sample_res)

# route = grp.trace_route(spawn_points[start_idx].location, spawn_points[end_idx].location)
# draw_waypoints(env.world, [wp for wp, _ in route], z=0.5, color=(0, 255, 0))
# 绘制自定义轨迹
draw_waypoints(env.world, [DummyWaypoint(float(pt[0]), float(pt[1]), z=0.5)
                           for pt in custom_trajectory],
               z=0.5, color=(0, 255, 0))
start_loc = custom_trajectory[0]
start_transform = carla.Transform(carla.Location(x=start_loc[0], y=start_loc[1], z=1), carla.Rotation(yaw=-90))
env.reset(spawn_point=start_transform)
# env.reset(spawn_point=spawn_points[start_idx])

# routes = []
# for wp, _ in route:
#     wp_t = wp.transform
#     routes.append([wp_t.location.x, wp_t.location.y])

dynamic_model = Vehicle(actor=env.ego_vehicle, horizon=10, target_v=target_v, delta_t=simu_step, max_iter=30)
agent = Xagent(env, dynamic_model, dt=simu_step)
# agent.set_start_end_transforms(start_idx, end_idx)
agent.set_start_end_transforms(custom_trajectory=custom_trajectory)

# agent.plan_route(agent._start_transform, agent._end_transform)
agent._waypoints_queue.clear()
for pt in custom_trajectory:
    wp = DummyWaypoint(pt[0], pt[1], z=0.5)
    agent._waypoints_queue.append((wp, RoadOption.LANEFOLLOW))


sim_time = 0
max_sim_steps = 2000

trajectory = []
velocities = []
accelerations = []
steerings = []
times = []
if env.display_method == "pygame":
    env.init_display()
try:
    for step in range(max_sim_steps):
        try:
            a_opt, delta_opt, next_state = agent.run_step()

            x, y, yaw, vx, vy, omega = next_state[0]
            trajectory.append([x, y])
            velocities.append(vx)
            accelerations.append(a_opt)
            steerings.append(delta_opt)
            times.append(step * simu_step)
            env.step([a_opt, delta_opt])

            if env.display_method == "pygame":
                # update HUD
                env.hud.tick(env, env.clock)

                if step == 0:
                    env.display.fill((0, 0, 0))
                env.hud.render(env.display)
                pygame.display.flip()
                env.check_quit()

            if np.linalg.norm([next_state[0][0] - agent._end_transform.location.x,
                               next_state[0][1] - agent._end_transform.location.y]) < 1.0:
                print("Destination reached!")
                if env.display_method == "pygame":
                    pygame.quit()
                sys.exit()
                break

            if env.display_method == "pygame":
                time.sleep(simu_step)

        except Exception as e:
            print(f"Warning: {e}")
            break

except KeyboardInterrupt:
    print("Simulation interrupted by user.")

# --- 计算平均横向误差与纵向误差 ---
def compute_errors(traj_real, traj_ref):
    lateral_errors = []
    longitudinal_errors = []
    last_tangent = np.array([1.0, 0.0])  # 默认方向，防止最开始没有定义

    for pt in traj_real:
        # 找到最近的参考点索引
        dists = np.linalg.norm(traj_ref - pt, axis=1)
        idx = np.argmin(dists)

        # 获取参考点的局部切向方向
        if idx < len(traj_ref) - 1:
            tangent = traj_ref[idx + 1] - traj_ref[idx]
        else:
            tangent = traj_ref[idx] - traj_ref[idx - 1]

        norm_tangent = np.linalg.norm(tangent)
        if norm_tangent < 1e-6:  # 若方向向量太短（几乎重合）
            tangent = last_tangent
        else:
            tangent = tangent / norm_tangent
            last_tangent = tangent  # 保存当前方向以备下次使用

        # 法向方向（垂直于切向）
        normal = np.array([-tangent[1], tangent[0]])

        # 实际误差向量
        error_vec = pt - traj_ref[idx]

        # 投影得到纵向与横向误差
        lon_err = np.dot(error_vec, tangent)
        lat_err = np.dot(error_vec, normal)

        longitudinal_errors.append(lon_err)
        lateral_errors.append(lat_err)

    mean_lat_error = np.mean(np.abs(lateral_errors))
    mean_lon_error = np.mean(np.abs(longitudinal_errors))
    return mean_lat_error, mean_lon_error


mean_lat_error, mean_lon_error = compute_errors(trajectory, custom_trajectory)
print(f"Average Lateral Error: {mean_lat_error:.3f} m")
print(f"Average Longitudinal Error: {mean_lon_error:.3f} m")

trajectory = np.array(trajectory)
velocities = np.array(velocities)
accelerations = np.array(accelerations)
steerings = np.array(steerings)
times = np.array(times)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1], label="Vehicle Path", color='darkorange', linewidth=2)
axs[0, 0].scatter(agent._start_transform.location.x, agent._start_transform.location.y, color='green', label="Start",
                  zorder=5)
axs[0, 0].scatter(agent._end_transform.location.x, agent._end_transform.location.y, color='red', label="End", zorder=5)

route_points = np.array(custom_trajectory)
axs[0, 0].plot(route_points[:, 0], route_points[:, 1], '--', color='blue', label="Planned Route", alpha=0.6)
axs[0, 0].set_title("Vehicle Path and Planned Route-MPC", fontsize=14)
axs[0, 0].set_xlabel("X Position", fontsize=12)
axs[0, 0].set_ylabel("Y Position", fontsize=12)
axs[0, 0].legend(loc='upper left', fontsize=10)
axs[0, 0].grid(True)

axs[0, 1].plot(times, velocities, label="Velocity (m/s)", color='royalblue', linewidth=2)
axs[0, 1].set_title("Velocity over Time", fontsize=14)
axs[0, 1].set_xlabel("Time (s)", fontsize=12)
axs[0, 1].set_ylabel("Velocity (m/s)", fontsize=12)
axs[0, 1].legend(loc='upper right', fontsize=10)
axs[0, 1].grid(True)

axs[1, 0].plot(times, accelerations, label="Acceleration (m/s²)", color='orange', linewidth=2)
axs[1, 0].set_title("Acceleration over Time", fontsize=14)
axs[1, 0].set_xlabel("Time (s)", fontsize=12)
axs[1, 0].set_ylabel("Acceleration (m/s²)", fontsize=12)
axs[1, 0].legend(loc='upper right', fontsize=10)
axs[1, 0].grid(True)

axs[1, 1].plot(times, steerings, label="Steering Angle (rad)", color='green', linewidth=2)
axs[1, 1].set_title("Steering Angle over Time", fontsize=14)
axs[1, 1].set_xlabel("Time (s)", fontsize=12)
axs[1, 1].set_ylabel("Steering Angle (rad)", fontsize=12)
axs[1, 1].legend(loc='upper right', fontsize=10)
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()