import numpy as np
import matplotlib.pyplot as plt
from src.mcp_controller import Vehicle
from env import Env, draw_waypoints

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).with_name('src')))
from src.x_v2x_agent import Xagent
from src.global_route_planner import GlobalRoutePlanner

import time
import pygame

# Simulation parameters
simu_step = 0.05  # Time step per simulation step (seconds)
target_v = 40  # Target speed (km/h)
sample_res = 2.0  # Sampling resolution for path planning
display_mode = "spec"  # Options: "spec" or "pygame"

env = Env(display_method=display_mode, dt=simu_step)
env.clean()
spawn_points = env.map.get_spawn_points()

start_idx, end_idx = 87, 70  # Indices for start and end points

grp = GlobalRoutePlanner(env.map, sample_res)

route = grp.trace_route(spawn_points[start_idx].location, spawn_points[end_idx].location)
draw_waypoints(env.world, [wp for wp, _ in route], z=0.5, color=(0, 255, 0))
env.reset(spawn_point=spawn_points[start_idx])

routes = []
for wp,_ in route:
    wp_t = wp.transform
    routes.append([wp_t.location.x, wp_t.location.y])


dynamic_model = Vehicle(actor=env.ego_vehicle, horizon=10, target_v=target_v, delta_t=simu_step, max_iter=30)
agent = Xagent(env, dynamic_model, dt=simu_step)
agent.set_start_end_transforms(start_idx, end_idx)


agent.plan_route(agent._start_transform, agent._end_transform)

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

trajectory = np.array(trajectory)
velocities = np.array(velocities)
accelerations = np.array(accelerations)
steerings = np.array(steerings)
times = np.array(times)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1], label="Vehicle Path", color='darkorange', linewidth=2)
axs[0, 0].scatter(agent._start_transform.location.x, agent._start_transform.location.y, color='green', label="Start", zorder=5)
axs[0, 0].scatter(agent._end_transform.location.x, agent._end_transform.location.y, color='red', label="End", zorder=5)

route_points = np.array([[wp.transform.location.x, wp.transform.location.y] for wp, _ in route])
axs[0, 0].plot(route_points[:, 0], route_points[:, 1], '--', color='blue', label="Planned Route", alpha=0.6)
axs[0, 0].set_title("Vehicle Path and Planned Route", fontsize=14)
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