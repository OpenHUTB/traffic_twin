#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

# System level imports
import sys
import os
import argparse
import logging
import time
import math
import numpy as np
import csv
import matplotlib
# matplotlib.use('agg')
import configparser
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
# Local level imports
import Controller

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
# import Live_Plotter as lv   # Custom live plotting library
import carla

"""
Configurable Parameters
"""
ITER_FOR_SIM_TIMESTEP = 10  # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 5.00  # simulator seconds (time before controller start)
TOTAL_RUN_TIME = 200.00  # simulator seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER = 300  # number of frames to buffer after total runtime
NUM_PEDESTRIANS = 0  # total number of pedestrians to spawn
NUM_VEHICLES = 4  # total number of vehicles to spawn
SEED_PEDESTRIANS = 0  # seed for pedestrian spawn randomizer
SEED_VEHICLES = 0  # seed for vehicle spawn randomizer

WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}
SIMWEATHER = WEATHERID["CLEARNOON"]  # set simulation weather

PLAYER_START_INDEX = 1  # spawn index for player (keep to 1)
FIGSIZE_X_INCHES = 6  # x figure size of feedback in inches
FIGSIZE_Y_INCHES = 8  # y figure size of feedback in inches
PLOT_LEFT = 0.1  # in fractions of figure width and height
PLOT_BOT = 0.1
PLOT_WIDTH = 0.8
PLOT_HEIGHT = 0.8

WAYPOINTS_FILENAME = 'Waypoints.txt'  # waypoint file to load
DIST_THRESHOLD_TO_LAST_WAYPOINT = 1.0  # some distance from last position before simulation ends (6 for Bang-Bang, 1 for others)

# Path interpolation parameters
INTERP_MAX_POINTS_PLOT = 10  # number of points used for displaying
# lookahead path
INTERP_LOOKAHEAD_DISTANCE = 20  # lookahead in meters
INTERP_DISTANCE_RES = 0.005  # distance between interpolated points

# Controller output directory
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/Results/'
IMAGE_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/Images/'


# 绘制车辆轨迹图
def plot_trajectory(x_list, y_list, waypoints_x, waypoints_y, vehicle_id):
    create_controller_output_dir(IMAGE_OUTPUT_FOLDER)

    plt.figure()
    plt.plot(waypoints_x, waypoints_y, label="Expected Path", color='g', linestyle='--')
    plt.plot(x_list, y_list, label="Actual Trajectory", color='b')
    plt.title(f'Vehicle {vehicle_id} Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    # plt.show()
    file_path = os.path.join(IMAGE_OUTPUT_FOLDER, f'trajectory_vehicle_{vehicle_id}.png')
    plt.savefig(file_path)
    plt.close()


# 绘制误差图
def plot_errors(cte_list, he_list, t_list, vehicle_id):
    create_controller_output_dir(IMAGE_OUTPUT_FOLDER)

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(t_list, cte_list, label="Crosstrack Error (CTE)", color='r')
    plt.title(f'Vehicle {vehicle_id} Tracking Errors')
    plt.xlabel('Time (s)')
    plt.ylabel('CTE (m)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t_list, he_list, label="Heading Error (HE)", color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading Error (rad)')
    plt.grid(True)

    plt.tight_layout()
    # plt.show()
    file_path = os.path.join(IMAGE_OUTPUT_FOLDER, f'errors_vehicle_{vehicle_id}.png')
    plt.savefig(file_path)
    plt.close()


# 绘制速度图
def plot_speed(v_list, t_list, vehicle_id):
    create_controller_output_dir(IMAGE_OUTPUT_FOLDER)

    plt.figure()
    plt.plot(t_list, v_list, label="Speed (m/s)", color='g')
    plt.title(f'Vehicle {vehicle_id} Speed over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.grid(True)
    # plt.show()
    file_path = os.path.join(IMAGE_OUTPUT_FOLDER, f'speed_vehicle_{vehicle_id}.png')
    plt.savefig(file_path)
    plt.close()


# 绘制延迟图
def plot_latency(latency_list, t_list, vehicle_id):
    create_controller_output_dir(IMAGE_OUTPUT_FOLDER)

    plt.figure()
    plt.plot(t_list, latency_list, label="Latency (ms)", color='purple')
    plt.title(f'Vehicle {vehicle_id} Latency over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Latency (ms)')
    plt.grid(True)
    # plt.show()
    file_path = os.path.join(IMAGE_OUTPUT_FOLDER, f'latency_vehicle_{vehicle_id}.png')
    plt.savefig(file_path)
    plt.close()


def make_carla_settings():
    """Make a CarlaSettings Object with the Required Settings"""
    settings = carla.WorldSettings(
        synchronous_mode=True,
        no_rendering_mode=False,
        fixed_delta_seconds=0.05

    )

    # Set weather
    weather = carla.WeatherParameters.ClearNoon
    if SIMWEATHER == WEATHERID["CLEARNOON"]:
        weather = carla.WeatherParameters.ClearNoon
    elif SIMWEATHER == WEATHERID["CLOUDYNOON"]:
        weather = carla.WeatherParameters.CloudyNoon
    # Add other weather conditions as needed

    return settings, weather


class Timer(object):
    """ Timer Class """

    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        return self.elapsed_seconds_since_lap() >= self._period_for_lap

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


# 获取车辆当前的位置和偏航
def get_current_pose(vehicle):
    """
    Obtains current x,y,yaw pose from the vehicle

    Args:
        vehicle: The CARLA vehicle actor

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    transform = vehicle.get_transform()
    x = transform.location.x
    y = transform.location.y
    yaw = math.radians(transform.rotation.yaw)
    return (x, y, yaw)


# 控制车辆的刹车、油门、方向盘
def send_control_command(vehicle, throttle, steer, brake, hand_brake=False, reverse=False):
    """
    Send control command to CARLA vehicle

    Args:
        vehicle: The CARLA vehicle actor
        throttle: Throttle command for the sim car [0, 1]
        steer: Steer command for the sim car [-1, 1]
        brake: Brake command for the sim car [0, 1]
        hand_brake: Whether the hand brake is engaged
        reverse: Whether the sim car is in the reverse gear
    """
    control = carla.VehicleControl()
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = hand_brake
    control.reverse = reverse
    vehicle.apply_control(control)


def cleanup_resources(world):
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)


# 创建输出目录
def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


# 保存轨迹图
def store_trajectory_plot(graph, fname):
    """ Store the Resulting Plot """
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)


# 将某辆车的轨迹信息（包括时间、位置和速度）保存为一个CSV文件
def write_trajectory_file(x_list, y_list, v_list, t_list, vehicle_id):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, f'Trajectory_{vehicle_id}.csv')  # t (s), x (m), y (m), v (m/s)
    with open(file_name, 'w') as trajectory_file:
        for i in range(len(x_list)):
            trajectory_file.write('%0.3f, %0.3f, %0.3f, %0.3f\n' % (t_list[i], x_list[i], y_list[i], v_list[i]))


# 将某辆车的跟踪误差（包括横向误差和航向误差）保存为一个CSV文件
def write_error_log(cte_list, he_list, vehicle_id):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, f'Tracking Error Log_{vehicle_id}.csv')  # cte (m), he (rad)
    with open(file_name, 'w') as error_log:
        for i in range(len(cte_list)):
            error_log.write('%0.10f,%0.10f\n' % (cte_list[i], he_list[i]))


# 将车辆的延迟数据（以毫秒为单位）
def write_latency_log(latency_list, vehicle_id):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, f'Latency Log_{vehicle_id}.csv')  # latency (ms)
    with open(file_name, 'w') as latency_log:
        for i in range(len(latency_list)):
            latency_log.write('%0.10f\n' % (latency_list[i]))


def exec_waypoint_nav_demo(args):
    """ Executes Waypoint Navigation """
    print('---------------------------------------------------------------------------')
    if args.longitudinal_controller == 'PID':
        print("\nLongitudinal Control: PID Controller")
    elif args.longitudinal_controller == 'ALC':
        print("\nLongitudinal Control: Adaptive Throttle Controller")
    else:
        print("\nUndefined Longitudinal Control Method Selected")

    if args.lateral_controller == 'BangBang':
        print("Lateral Control: Bang-Bang Controller\n")
    elif args.lateral_controller == 'PID':
        print("Lateral Control: PID Controller\n")
    elif args.lateral_controller == 'PurePursuit':
        print("Lateral Control: Pure Pursuit Controller\n")
    elif args.lateral_controller == 'Stanley':
        print("Lateral Control: Stanley Controller\n")
    elif args.lateral_controller == 'POP':
        print("Lateral Control: Proximally Optimal Pursuit Controller\n")
    else:
        print("Undefined Lateral Control Method Selected\n")

    # 连接CARLA服务器并设置仿真环境,生成车辆
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    settings, weather = make_carla_settings()
    world.apply_settings(settings)
    world.set_weather(weather)
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.audi.tt')[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicles = []
    # for ind, spawn_point in enumerate(spawn_points):
    #     world.debug.draw_string(spawn_point.location, str(ind), life_time=1000, color=carla.Color(255, 0, 0))

    # for i in range(NUM_VEHICLES):
    #     vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[i])
    #     vehicles.append(vehicle)
    vehicle_1 = world.try_spawn_actor(vehicle_bp, spawn_points[0])
    vehicle_2 = world.try_spawn_actor(vehicle_bp, spawn_points[1])
    vehicle_3 = world.try_spawn_actor(vehicle_bp, spawn_points[104])
    vehicle_4 = world.try_spawn_actor(vehicle_bp, spawn_points[108])
    vehicles.append(vehicle_1)
    vehicles.append(vehicle_2)
    vehicles.append(vehicle_3)
    vehicles.append(vehicle_4)
    # spectator = world.get_spectator()
    # spectator.set_transform(carla.Transform(carla.Location(x=94.826576, y=34.715199, z=275.478973), carla.Rotation(pitch=-88.999062, yaw=9.844740, roll=0.000024)))
    # # 从CSV文件中读取路径点数据，并将其转换为NumPy数组，便于进一步处理
    waypoints_file = WAYPOINTS_FILENAME
    # with open(waypoints_file) as waypoints_file_handle:
    #     waypoints = list(csv.reader(waypoints_file_handle, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
    # waypoints_np = np.array(waypoints)
    waypoints_by_vehicle = {}
    with open(waypoints_file, 'r') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            vehicle_id = int(row[0])
            x = row[1]
            y = row[2]
            t = row[3]
            if vehicle_id not in waypoints_by_vehicle:
                waypoints_by_vehicle[vehicle_id] = []
            waypoints_by_vehicle[vehicle_id].append((x, y, t))

    # 计算路径点之间的距离
    wp_distances_by_vehicle = {}
    for vehicle_id, waypoints in waypoints_by_vehicle.items():
        waypoints_np = np.array(waypoints)
        wp_distance = [np.sqrt(
            (waypoints_np[i][0] - waypoints_np[i - 1][0]) ** 2 +
            (waypoints_np[i][1] - waypoints_np[i - 1][1]) ** 2
        ) for i in range(1, waypoints_np.shape[0])]
        wp_distance.append(0)  # 添加最后一个点的距离
        wp_distances_by_vehicle[vehicle_id] = wp_distance

    # 轨迹平滑
    smoothed_waypoints_by_vehicle = {}   # 存储每辆车平滑后的路径序列
    min_distance = 2 * INTERP_DISTANCE_RES  # 插值的最小距离阈值
    interp_hash_by_vehicle = {}         # 存储原始路径点在插值后路径点列表中的索引
    for vehicle_id, waypoints in waypoints_by_vehicle.items():
        # 单个车辆的轨迹
        waypoints_np = np.array(waypoints)
        # 单个车辆航点之间的距离
        wp_distance = wp_distances_by_vehicle[vehicle_id]
        # 插值
        wp_interp = []           # 插值后的路径点列表
        wp_interp_hash = []      # 原始路径点在插值后的路径点列表中的索引
        interp_counter = 0       # 插值计数器
        for i in range(waypoints_np.shape[0] - 1):
            ##################
            # d1  data   d2  #
            # 0    20    21  #
            ##################
            wp_interp.append(list(waypoints_np[i]))      # 首先将当前路径点添加到 wp_interp 和 wp_interp_hash
            wp_interp_hash.append(interp_counter)
            interp_counter += 1
            if wp_distance[i] < min_distance:
                continue  # 如果距离太小，跳过插值
            # 否则，计算插值点数
            num_pts_to_interp = int(np.floor(wp_distance[i] / float(INTERP_DISTANCE_RES)) - 1)

            # 计算两路径点之间的向量 wp_vector 和单位向量 wp_uvector
            wp_vector = waypoints_np[i + 1] - waypoints_np[i]
            wp_uvector = wp_vector / np.linalg.norm(wp_vector)

            # 在两个路径点之间插入若干插值点 ，并将它们添加到 wp_interp
            for j in range(num_pts_to_interp):
                next_wp_vector = INTERP_DISTANCE_RES * float(j + 1) * wp_uvector
                wp_interp.append(list(waypoints_np[i] + next_wp_vector))
                interp_counter += 1

        # 添加最后一个路径点
        wp_interp.append(list(waypoints_np[-1]))
        wp_interp_hash.append(interp_counter)
        smoothed_waypoints_by_vehicle[vehicle_id] = wp_interp
        interp_hash_by_vehicle[vehicle_id] = wp_interp_hash

    # 添加控制器
    controllers = []
    for vehicle_id, vehicle in enumerate(vehicles):
        waypoints = waypoints_by_vehicle[vehicle_id]
        if not waypoints:
            print(f"车辆{vehicle_id}没有航点。")
            continue
        controller = Controller.Controller(waypoints, args.lateral_controller, args.longitudinal_controller)
        controllers.append(controller)

    # 计算仿真时间步长
    # 迭代次数
    num_iterations = ITER_FOR_SIM_TIMESTEP
    if (ITER_FOR_SIM_TIMESTEP < 1):
        num_iterations = 1
    sim_start_stamp = world.get_snapshot().timestamp.elapsed_seconds
    for vehicle in vehicles:
        send_control_command(vehicle, throttle=0.0, steer=0, brake=1.0)
    sim_duration = 0
    for i in range(num_iterations):
        world.tick()
        current_stamp = world.get_snapshot().timestamp.elapsed_seconds
        sim_duration += (current_stamp - sim_start_stamp)
        for vehicle in vehicles:
            send_control_command(vehicle, throttle=0.0, steer=0, brake=1.0)
        sim_start_stamp = current_stamp
    # 计算平均仿真时间
    SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
    print("SERVER SIMULATION STEP APPROXIMATION: " + str(SIMULATION_TIME_STEP))

    #  计算总的仿真帧数
    TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) / SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER
    #  初始化车辆的初始状态
    x_histories, y_histories, yaw_histories, time_histories, speed_histories = [[] for _ in vehicles], [[] for _ in vehicles], [[]for _ in vehicles], [[] for _ in vehicles], [[] for _ in vehicles]
    cte_histories, he_histories, latency_histories = [[] for _ in vehicles], [[] for _ in vehicles], [[] for _ in vehicles]
    reached_the_end = [False for _ in vehicles]
    closest_indices = [0 for _ in vehicles]

    while True:
        world.tick()
        # 避免最后车辆越接近终点速度越快
        time.sleep(0.05)
        measurement = world.get_snapshot()
        for i, vehicle in enumerate(vehicles):

            # print(f"车辆 {i} 最接近的索引是 {closest_indices[i]}")
            dist_to_last_waypoint = 0.0
            if vehicle is not None:
                current_x, current_y, current_yaw = get_current_pose(vehicle)
                current_speed = vehicle.get_velocity().length()
                current_timestamp = measurement.timestamp.elapsed_seconds

                length = -1.5 if args.lateral_controller == 'PurePursuit' else 1.5 if args.lateral_controller in {
                    'BangBang', 'PID', 'Stanley', 'POP'} else 0.0
                # 坐标转换
                current_x, current_y = controllers[i].get_shifted_coordinate(current_x, current_y, current_yaw, length)
                # 设置时间等待资源就绪
                if current_timestamp <= WAIT_TIME_BEFORE_START:
                    send_control_command(vehicle, throttle=0.0, steer=0, brake=1.0)
                    continue
                else:
                    current_timestamp -= WAIT_TIME_BEFORE_START
                # 记录历史状态
                x_histories[i].append(current_x)
                y_histories[i].append(current_y)
                yaw_histories[i].append(current_yaw)
                speed_histories[i].append(current_speed)
                time_histories[i].append(current_timestamp)
                # 初始化最近距离
                closest_distance = np.linalg.norm(np.array([waypoints_by_vehicle[i][closest_indices[i]][0] - current_x, waypoints_by_vehicle[i][closest_indices[i]][1] - current_y]))
                # 向后找到距离当前位置最近的航点
                new_distance = closest_distance
                new_index = closest_indices[i]
                while new_distance <= closest_distance:
                    closest_distance = new_distance
                    closest_indices[i] = new_index
                    new_index += 1
                    if new_index >= len(waypoints_by_vehicle[i]):
                        break
                    new_distance = np.linalg.norm(np.array([waypoints_by_vehicle[i][new_index][0] - current_x, waypoints_by_vehicle[i][new_index][1] - current_y]))
                # 向前找到距离当前位置最近的航点
                new_distance = closest_distance
                new_index = closest_indices[i]
                while new_distance <= closest_distance:
                    closest_distance = new_distance
                    closest_indices[i] = new_index
                    new_index -= 1
                    if new_index < 0:
                        break
                    new_distance = np.linalg.norm(np.array([waypoints_by_vehicle[i][new_index][0] - current_x, waypoints_by_vehicle[i][new_index][1] - current_y]))
                # 更新航点子集并计算控制命令
                # 初始化局部路径的起始和终止索引(一段距离内的起始航点与结束航点)
                waypoint_subset_first_index = closest_indices[i] - 1 if closest_indices[i] - 1 >= 0 else 0    # 表示当前车辆最接近航点的前一个航点索引
                waypoint_subset_last_index = closest_indices[i]                                               # 表示最接近车辆当前位置的航点索引
                total_distance_ahead = 0
                while total_distance_ahead < INTERP_LOOKAHEAD_DISTANCE:
                    total_distance_ahead += np.linalg.norm(np.array([waypoints_by_vehicle[i][waypoint_subset_last_index][0] - waypoints_by_vehicle[i][waypoint_subset_last_index - 1][0], waypoints_by_vehicle[i][waypoint_subset_last_index][1] - waypoints_by_vehicle[i][waypoint_subset_last_index - 1][1]]))
                    waypoint_subset_last_index += 1
                    if waypoint_subset_last_index >= len(waypoints_by_vehicle[i]):
                        waypoint_subset_last_index = len(waypoints_by_vehicle[i]) - 1
                        break

                # 指定一段距离内包括平滑后的航点
                new_waypoints = smoothed_waypoints_by_vehicle[i][
                                interp_hash_by_vehicle[i][waypoint_subset_first_index]:interp_hash_by_vehicle[i][waypoint_subset_last_index] + 1]

                controllers[i].update_waypoints(new_waypoints)
                # 更新控制器中的车辆状态
                controllers[i].update_values(current_x, current_y, current_yaw, current_speed, current_timestamp, 1, new_distance)
                # 计算控制命令
                controllers[i].update_controls()

                # 获取控制命令
                cmd_throttle, cmd_steer, cmd_brake = controllers[i].get_commands()
                # 记录控制误差并发送控制命令
                cte_histories[i].append(controllers[i].get_crosstrack_error(current_x, current_y, new_waypoints))
                he_histories[i].append(controllers[i].get_heading_error(new_waypoints, current_yaw))
                latency_histories[i].append(controllers[i]._latency)

                send_control_command(vehicle, throttle=cmd_throttle, steer=cmd_steer, brake=cmd_brake)

                dist_to_last_waypoint = np.linalg.norm(
                    np.array([waypoints_by_vehicle[i][-1][0] - current_x, waypoints_by_vehicle[i][-1][1] - current_y]))
            if dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT and vehicle is not None:
                reached_the_end[i] = True
                vehicle.destroy()
                vehicles[i] = None
        if all(reached_the_end):
            break

    try:
        for i in range(NUM_VEHICLES):
            if reached_the_end[i]:
                print(f"\n车辆{i + 1}到达了路径终点。记录结果...")
            else:
                print(f"\n车辆{i + 1}超过了仿真时间。记录结果...")
            # 保存车辆轨迹数据
            write_trajectory_file(x_histories[i], y_histories[i], speed_histories[i], time_histories[i], i)
            write_error_log(cte_histories[i], he_histories[i], i)
            write_latency_log(latency_histories[i], i)

            # 生成轨迹图（绘制车辆实际行驶轨迹和预期路径，观察分析车辆是否准确跟随路径。）
            waypoints_x = [wp[0] for wp in waypoints_by_vehicle[i]]
            waypoints_y = [wp[1] for wp in waypoints_by_vehicle[i]]
            plot_trajectory(x_histories[i], y_histories[i], waypoints_x, waypoints_y, i)
            # 生成误差图（横向误差和航向误差，在同一个图中，分析车辆的控制精度）
            plot_errors(cte_histories[i], he_histories[i], time_histories[i], i)
            # 生成速度图（显示车辆随时间的变化的情况，方便分析加速和减速行为。）
            plot_speed(speed_histories[i], time_histories[i], i)
            # 生成延迟图（系统延迟随时间变化的图，帮助分析系统的实时性表现）
            plot_latency(latency_histories[i], time_histories[i], i)
    finally:
        cleanup_resources(world)


def main():
    """
    Main function
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    argparser.add_argument(
        '-lat_ctrl', '--Lateral-Controller',
        metavar='LATERAL CONTROLLER',
        dest='lateral_controller',
        choices={'BangBang', 'PID', 'PurePursuit', 'Stanley', 'POP'},
        default='POP',
        help='Select Lateral Controller')
    argparser.add_argument(
        '-lon_ctrl', '--Longitudinal-Controller',
        metavar='LONGITUDINAL CONTROLLER',
        dest='longitudinal_controller',
        choices={'PID', 'ALC'},
        default='ALC',
        help='Select Longitudinal Controller')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    try:
        # 开始运行轨迹跟踪
        exec_waypoint_nav_demo(args)
        print('\nSimulation Complete')
        return
    except carla.TCPConnectionError as error:
        logging.error(error)
        time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt Detected...\nTerminating Simulation')
