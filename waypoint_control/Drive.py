#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import random
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

matplotlib.use('agg')
import configparser
import warnings

warnings.filterwarnings("ignore")

# Local level imports
import Controller

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
# import Live_Plotter as lv   # Custom live plotting library
import carla

"""
Configurable Parameters
"""
ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 5.00   # simulator seconds (time before controller start)
TOTAL_RUN_TIME         = 200.00 # simulator seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER     = 300    # number of frames to buffer after total runtime
NUM_PEDESTRIANS        = 0      # total number of pedestrians to spawn
NUM_VEHICLES           = 3     # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0      # seed for pedestrian spawn randomizer
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

# Path interpolation parameterse
INTERP_MAX_POINTS_PLOT = 10  # number of points used for displaying
# lookahead path
INTERP_LOOKAHEAD_DISTANCE = 20  # lookahead in meters
INTERP_DISTANCE_RES = 0.01  # distance between interpolated points

# Controller output directory
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/Results/'


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


def add_vehicle(vehicles, sim_time, waypoints_by_vehicle, vehicle_ids, vehicle_bp, world, args, controllers, _map,
                x_histories, y_histories, yaw_histories, time_histories, speed_histories, cte_histories,
                he_histories, latency_histories, reached_the_end, closest_indices, last_location):

    for vehicle_id, waypoints in waypoints_by_vehicle.items():
        if waypoints:
            first_waypoint_time = waypoints[0][2]
            if sim_time >= first_waypoint_time and vehicle_id not in vehicle_ids:
                location = carla.Location(x=waypoints[0][0], y=waypoints[0][1], z=1)
                world.debug.draw_string(location, str(vehicle_id), life_time=1000, color=carla.Color(255, 0, 0))
                waypoint = _map.get_waypoint(location)
                yaw = waypoint.transform.rotation.yaw
                trans = carla.Transform(location, carla.Rotation(yaw=yaw))
                # 生成车辆
                vehicle = world.try_spawn_actor(random.choice(vehicle_bp), trans)
                if vehicle is None:
                    continue
                # 添加控制器
                controller = Controller.Controller(waypoints, args.lateral_controller, args.longitudinal_controller)
                controllers.append(controller)

                vehicles.append(vehicle)
                vehicle_ids.append(vehicle_id)
                x_histories.append([])
                y_histories.append([])
                yaw_histories.append([])
                time_histories.append([])
                speed_histories.append([])
                cte_histories.append([])
                he_histories.append([])
                latency_histories.append([])
                reached_the_end.append(False)
                closest_indices.append(0)
                last_location.append((waypoints[0][0], waypoints[0][1], 2))


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
    z = transform.location.z
    yaw = math.radians(transform.rotation.yaw)
    return (x, y, z, yaw)


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


def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


def store_trajectory_plot(graph, fname):
    """ Store the Resulting Plot """
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)


def write_trajectory_file(x_list, y_list, v_list, t_list, vehicle_id):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, f'Trajectory_{vehicle_id}.csv')  # t (s), x (m), y (m), v (m/s)
    with open(file_name, 'w') as trajectory_file:
        for i in range(len(x_list)):
            trajectory_file.write('%0.3f, %0.3f, %0.3f, %0.3f\n' % (t_list[i], x_list[i], y_list[i], v_list[i]))


def write_error_log(cte_list, he_list, vehicle_id):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, f'Tracking Error Log_{vehicle_id}.csv')  # cte (m), he (rad)
    with open(file_name, 'w') as error_log:
        for i in range(len(cte_list)):
            error_log.write('%0.10f,%0.10f\n' % (cte_list[i], he_list[i]))


def write_latency_log(latency_list, vehicle_id):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, f'Latency Log_{vehicle_id}.csv')  # latency (ms)
    with open(file_name, 'w') as latency_log:
        for i in range(len(latency_list)):
            latency_log.write('%0.10f\n' % (latency_list[i]))


def filter_vehicle_blueprinter(vehicle_blueprints):
    """
    :param vehicle_blueprints: 车辆蓝图
    :return: 过滤自行车后的车辆蓝图
    """
    filtered_vehicle_blueprints = [bp for bp in vehicle_blueprints if 'bike' not in bp.id and
                                   'omafiets' not in bp.id and
                                   'century' not in bp.id and
                                   'vespa' not in bp.id and
                                   'motorcycle' not in bp.id and
                                   'harley' not in bp.id and
                                   'yamaha' not in bp.id and
                                   'kawasaki' not in bp.id and
                                   'mini' not in bp.id]
    return filtered_vehicle_blueprints


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
    _map = world.get_map()

    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    vehicle_bp = filter_vehicle_blueprinter(vehicle_blueprints)
    # 车辆 车辆id 车辆控制器
    vehicles = []
    vehicle_ids = []
    controllers = []

    # 添加航点
    # 从CSV文件中读取路径点数据，并将其转换为NumPy数组，便于进一步处理
    waypoints_file = WAYPOINTS_FILENAME
    waypoints_by_vehicle = {}
    with open(waypoints_file, 'r') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            vehicle_id = int(row[0])
            x = row[1]
            y = row[2]
            t = row[3]
            if vehicle_id not in waypoints_by_vehicle:
                waypoints_by_vehicle[vehicle_id] = []
            waypoints_by_vehicle[vehicle_id].append((x, y, t))

    # 计算单个车辆路径点之间的距离
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
    smoothed_waypoints_by_vehicle = {}         # 存储每辆车平滑后的路径序列
    min_distance = 2 * INTERP_DISTANCE_RES     # 插值的最小距离阈值
    interp_hash_by_vehicle = {}                # 存储原始路径点在插值后路径点列表中的索引
    for vehicle_id, waypoints in waypoints_by_vehicle.items():
        # 单个车辆的轨迹
        waypoints_np = np.array(waypoints)
        # 单个车辆航点之间的距离
        wp_distance = wp_distances_by_vehicle[vehicle_id]
        # 插值
        wp_interp = []                                     # 插值后的路径点列表
        wp_interp_hash = []                                # 原始路径点在插值后的路径点列表中的索引
        interp_counter = 0                                 # 插值计数器
        for i in range(waypoints_np.shape[0] - 1):
            ##################
            # d1  data   d2  #
            # 0    20    21  #
            ##################
            wp_interp.append(list(waypoints_np[i]))        # 首先将当前路径点添加到 wp_interp 和 wp_interp_hash
            wp_interp_hash.append(interp_counter)          # 原始路径点在插值后的路径点列表中的索引
            interp_counter += 1
            if wp_distance[i] < min_distance:
                continue  # 如果距离太小，跳过插值
            # 否则，计算插值点数
            num_pts_to_interp = int(np.floor(wp_distance[i] / float(INTERP_DISTANCE_RES)) - 1)

            # 计算两路径点之间的向量 wp_vector 和单位向量 wp_uvector
            wp_vector = waypoints_np[i+1] - waypoints_np[i]
            wp_uvector = wp_vector / np.linalg.norm(wp_vector)

            # 在两个路径点之间插入若干插值点 ，并将它们添加到 wp_interp
            for j in range(num_pts_to_interp):
                next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                wp_interp.append(list(waypoints_np[i] + next_wp_vector))
                interp_counter += 1

        # 添加最后一个路径点
        wp_interp.append(list(waypoints_np[-1]))
        wp_interp_hash.append(interp_counter)
        smoothed_waypoints_by_vehicle[vehicle_id] = wp_interp
        interp_hash_by_vehicle[vehicle_id] = wp_interp_hash

    # 初始化车辆的初始状态
    x_histories, y_histories, yaw_histories, time_histories, speed_histories = [], [], [], [], []
    cte_histories, he_histories, latency_histories = [], [], []
    reached_the_end = []
    closest_indices = []
    last_location = []
    measurement = world.get_snapshot()
    first_time = measurement.timestamp.elapsed_seconds
    while True:
        current_timestamp = world.get_snapshot().timestamp.elapsed_seconds
        sim_time = current_timestamp - first_time
        add_vehicle(vehicles, sim_time, waypoints_by_vehicle, vehicle_ids, vehicle_bp, world, args, controllers, _map,
                    x_histories, y_histories, yaw_histories, time_histories, speed_histories, cte_histories,
                    he_histories, latency_histories, reached_the_end, closest_indices, last_location)
        world.tick()
        for i, vehicle in enumerate(vehicles):
            dist_to_last_waypoint = 0.0
            if vehicle is not None:
                current_x, current_y, current_z, current_yaw = get_current_pose(vehicle)
                # 销毁异常车辆,包括轨迹异常，行驶异常导致的车辆停止
                # distance_to_last_location = np.linalg.norm(np.array([last_location[i][0] - current_x, last_location[i][1] - current_y, last_location[i][2] - current_z]))
                # if distance_to_last_location < 0.005:
                #     reached_the_end[i] = True
                #     vehicle.destroy()
                #     vehicles[i] = None
                #     continue
                last_location[i] = (current_x, current_y, current_z)
                current_speed = vehicle.get_velocity().length()
                length = -1.5 if args.lateral_controller == 'PurePursuit' else 1.5 if args.lateral_controller in {'BangBang', 'PID', 'Stanley', 'POP'} else 0.0

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

                vehicle_id = vehicle_ids[i]
                # 计算航点中与车辆当前位置最近的航点的距离
                closest_distance = np.linalg.norm(np.array([waypoints_by_vehicle[vehicle_id][closest_indices[i]][0] - current_x, waypoints_by_vehicle[vehicle_id][closest_indices[i]][1] - current_y]))
                # 向前搜索最近的路径点
                new_distance = closest_distance
                new_index = closest_indices[i]
                while new_distance <= closest_distance:
                    closest_distance = new_distance
                    closest_indices[i] = new_index
                    new_index += 1
                    if new_index >= len(waypoints_by_vehicle[vehicle_id]):
                        break
                    new_distance = np.linalg.norm(np.array([waypoints_by_vehicle[vehicle_id][new_index][0] - current_x, waypoints_by_vehicle[vehicle_id][new_index][1] - current_y]))
                # 向后搜索最近的路径点
                new_distance = closest_distance
                new_index = closest_indices[i]
                while new_distance <= closest_distance:
                    closest_distance = new_distance
                    closest_indices[i] = new_index
                    new_index -= 1
                    if new_index < 0:
                        break
                    new_distance = np.linalg.norm(np.array([waypoints_by_vehicle[vehicle_id][new_index][0] - current_x, waypoints_by_vehicle[vehicle_id][new_index][1] - current_y]))

                # 确定路径点子集的起始和结束索引
                waypoint_subset_first_index = closest_indices[i] - 1 if closest_indices[i] - 1 >= 0 else 0
                waypoint_subset_last_index = closest_indices[i]
                # 计算前方路径点的总距离
                total_distance_ahead = 0
                while total_distance_ahead < INTERP_LOOKAHEAD_DISTANCE:
                    total_distance_ahead += np.linalg.norm(np.array([waypoints_by_vehicle[vehicle_id][waypoint_subset_last_index][0] - waypoints_by_vehicle[vehicle_id][waypoint_subset_last_index - 1][0], waypoints_by_vehicle[vehicle_id][waypoint_subset_last_index][1] - waypoints_by_vehicle[vehicle_id][waypoint_subset_last_index - 1][1]]))
                    waypoint_subset_last_index += 1
                    if waypoint_subset_last_index >= len(waypoints_by_vehicle[vehicle_id]):
                        waypoint_subset_last_index = len(waypoints_by_vehicle[vehicle_id]) - 1
                        break
                # 获取新的路径点子集
                new_waypoints = smoothed_waypoints_by_vehicle[vehicle_id][interp_hash_by_vehicle[vehicle_id][waypoint_subset_first_index]:interp_hash_by_vehicle[vehicle_id][waypoint_subset_last_index] + 1]
                controllers[i].update_waypoints(new_waypoints)  # 更新控制器的路径点
                # 更新控制器的状态值
                controllers[i].update_values(current_x, current_y, current_yaw, current_speed, current_timestamp, 1, new_distance)
                # 计算控制命令
                controllers[i].update_controls()
                cmd_throttle, cmd_steer, cmd_brake = controllers[i].get_commands()
                # 记录横向误差和纵向误差
                cte_histories[i].append(controllers[i].get_crosstrack_error(current_x, current_y, new_waypoints))
                he_histories[i].append(controllers[i].get_heading_error(new_waypoints, current_yaw))
                # 记录延迟
                latency_histories[i].append(controllers[i]._latency)
                # 发送控制命令
                send_control_command(vehicle, throttle=cmd_throttle, steer=cmd_steer, brake=cmd_brake)
                # 计算到最后一个路径点的距离
                dist_to_last_waypoint = np.linalg.norm(np.array([waypoints_by_vehicle[vehicle_id][-1][0] - current_x, waypoints_by_vehicle[vehicle_id][-1][1] - current_y]))
            if dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT and vehicle is not None:
                reached_the_end[i] = True
                vehicle.destroy()
                vehicles[i] = None

        if reached_the_end and all(reached_the_end):
            break
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

    while True:

        # 开始运行轨迹跟踪
        exec_waypoint_nav_demo(args)
        print('\nSimulation Complete')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt Detected...\nTerminating Simulation')