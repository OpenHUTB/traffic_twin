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
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import queue
import threading

warnings.filterwarnings("ignore")

# Local level imports
import Controller

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import carla

"""
Configurable Parameters
"""
ITER_FOR_SIM_TIMESTEP = 10  # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 2.00  # simulator seconds (time before controller start)
TOTAL_RUN_TIME = 200.00  # simulator seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER = 300  # number of frames to buffer after total runtime
NUM_PEDESTRIANS = 0  # total number of pedestrians to spawn
NUM_VEHICLES = 3  # total number of vehicles to spawn
SEED_PEDESTRIANS = 0  # seed for pedestrian spawn randomizer
SEED_VEHICLES = 0  # seed for vehicle spawn randomizer
MAX_SIMULATION_FRAMES = 700  # maximum number of simulation frames

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

WAYPOINTS_FILENAME = 'vehicleWaypoints.txt'  # waypoint file to load
PEDESTRIAN_WAYPOINTS_FILENAME = 'personWaypoints.txt'  # 行人轨迹文件
DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0  # some distance from last position before simulation ends

# Path interpolation parameters
INTERP_MAX_POINTS_PLOT = 10  # number of points used for displaying
INTERP_LOOKAHEAD_DISTANCE = 20  # lookahead in meters
INTERP_DISTANCE_RES = 0.01  # distance between interpolated points

# Controller output directory
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/Results/'


@dataclass
class VehicleData:
    """车辆数据类"""
    id: int
    actor: carla.Vehicle
    controller: Any
    waypoints: List[Tuple[float, float, float]]
    x_history: List[float]
    y_history: List[float]
    yaw_history: List[float]
    time_history: List[float]
    speed_history: List[float]
    cte_history: List[float]
    he_history: List[float]
    latency_history: List[float]
    reached_end: bool = False
    closest_index: int = 0
    last_location: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class PedestrianData:
    """行人数据类"""
    id: int
    actor: carla.Walker
    controller: carla.WalkerAIController
    trajectory: List[Dict[str, float]]
    current_waypoint_idx: int = 0
    completed: bool = False
    spawn_time: float = 0.0


class SimulationManager:
    """仿真管理器"""

    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.map = None
        self.original_settings = None

        # 数据存储
        self.vehicles: Dict[int, VehicleData] = {}
        self.pedestrians: Dict[int, PedestrianData] = {}

        # 轨迹数据
        self.vehicle_waypoints: Dict[int, List[Tuple[float, float, float]]] = {}
        self.pedestrian_trajectories: Dict[int, List[Dict[str, float]]] = {}

        # 状态
        self.sim_running = False
        self.sim_start_time = 0.0
        self.frame_count = 0

    def connect_to_carla(self) -> bool:
        """连接到Carla服务器"""
        try:
            print(f"Connecting to CARLA server at {self.args.host}:{self.args.port}...")
            self.client = carla.Client(self.args.host, self.args.port)
            self.client.set_timeout(30.0)
            self.world = self.client.get_world()

            # 保存原始设置
            self.original_settings = self.world.get_settings()

            print(f"Connected to CARLA server. Map: {self.world.get_map().name}")
            return True

        except Exception as e:
            print(f"Failed to connect to CARLA: {e}")
            return False

    def setup_simulation(self) -> bool:
        """设置仿真环境"""
        try:
            # 设置天气
            weather = carla.WeatherParameters.ClearNoon
            if SIMWEATHER == WEATHERID["CLEARNOON"]:
                weather = carla.WeatherParameters.ClearNoon
            elif SIMWEATHER == WEATHERID["CLOUDYNOON"]:
                weather = carla.WeatherParameters.CloudyNoon

            self.world.set_weather(weather)
            self.map = self.world.get_map()

            # 设置同步模式
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            settings.no_rendering_mode = False

            self.world.apply_settings(settings)

            # 设置交通管理器
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_random_device_seed(0)

            print("Simulation setup completed")
            return True

        except Exception as e:
            print(f"Failed to setup simulation: {e}")
            return False

    def load_vehicle_waypoints(self) -> bool:
        """加载车辆轨迹点"""
        try:
            print(f"Loading vehicle waypoints from {WAYPOINTS_FILENAME}...")

            if not os.path.exists(WAYPOINTS_FILENAME):
                print(f"Error: Waypoints file not found: {WAYPOINTS_FILENAME}")
                return False

            with open(WAYPOINTS_FILENAME, 'r') as f:
                reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
                for row in reader:
                    if len(row) < 4:
                        continue

                    vehicle_id = int(row[0])
                    x = float(row[1])
                    y = float(row[2])
                    t = float(row[3])

                    if vehicle_id not in self.vehicle_waypoints:
                        self.vehicle_waypoints[vehicle_id] = []

                    self.vehicle_waypoints[vehicle_id].append((x, y, t))

            # 按时间排序
            for vehicle_id in self.vehicle_waypoints:
                self.vehicle_waypoints[vehicle_id].sort(key=lambda wp: wp[2])

            print(f"Loaded waypoints for {len(self.vehicle_waypoints)} vehicles")
            return True

        except Exception as e:
            print(f"Failed to load vehicle waypoints: {e}")
            return False

    def load_pedestrian_trajectories(self) -> bool:
        """加载行人轨迹"""
        try:
            print(f"Loading pedestrian trajectories from {PEDESTRIAN_WAYPOINTS_FILENAME}...")

            if not os.path.exists(PEDESTRIAN_WAYPOINTS_FILENAME):
                print(f"Warning: Pedestrian trajectories file not found: {PEDESTRIAN_WAYPOINTS_FILENAME}")
                return False  # 没有行人轨迹程序中断

            with open(PEDESTRIAN_WAYPOINTS_FILENAME, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 4:
                        continue

                    try:
                        person_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        timestamp = float(parts[3])

                        if person_id not in self.pedestrian_trajectories:
                            self.pedestrian_trajectories[person_id] = []

                        self.pedestrian_trajectories[person_id].append({
                            'x': x,
                            'y': y,
                            'timestamp': timestamp
                        })
                    except ValueError:
                        continue

            # 按时间排序
            for person_id in self.pedestrian_trajectories:
                self.pedestrian_trajectories[person_id].sort(key=lambda wp: wp['timestamp'])

            print(f"Loaded trajectories for {len(self.pedestrian_trajectories)} pedestrians")
            return True

        except Exception as e:
            print(f"Failed to load pedestrian trajectories: {e}")
            return False

    def get_ground_height(self, x: float, y: float) -> float:
        """获取地面高度"""
        try:
            # 创建raycast查询
            start_location = carla.Location(x=x, y=y, z=1000.0)
            end_location = carla.Location(x=x, y=y, z=-1000.0)

            # 执行raycast
            hit = self.world.cast_ray(start_location, end_location)

            if hit:
                return hit.location.z + 0.5  # 稍微抬高一点

            # 如果没有命中，尝试使用waypoint
            waypoint = self.map.get_waypoint(carla.Location(x=x, y=y, z=0))
            if waypoint:
                return waypoint.transform.location.z + 0.5

            # 默认高度
            return 1.0

        except Exception as e:
            print(f"Error getting ground height at ({x}, {y}): {e}")
            return 1.0

    def spawn_vehicle(self, vehicle_id: int, waypoints: List[Tuple[float, float, float]]) -> Optional[VehicleData]:
        """生成车辆"""
        try:
            # 获取车辆蓝图
            blueprint_library = self.world.get_blueprint_library()

            # 过滤车辆蓝图
            vehicle_bps = [
                bp for bp in blueprint_library.filter('vehicle.*')
                if not any(excluded in bp.id for excluded in [
                    'vehicle.micro.microlino', 'vehicle.mini.cooper_s_2021',
                    'vehicle.nissan.patrol_2021', 'vehicle.carlamotors.carlacola',
                    'vehicle.carlamotors.european_hgv', 'vehicle.carlamotors.firetruck',
                    'vehicle.tesla.cybertruck', 'vehicle.ford.ambulance',
                    'vehicle.mercedes.sprinter', 'vehicle.volkswagen.t2',
                    'vehicle.volkswagen.t2_2021', 'vehicle.mitsubishi.fusorosa',
                    'vehicle.harley-davidson.low_rider', 'vehicle.kawasaki.ninja',
                    'vehicle.vespa.zx125', 'vehicle.yamaha.yzf',
                    'vehicle.bh.crossbike', 'vehicle.diamondback.century',
                    'vehicle.gazelle.omafiets'
                ])
            ]

            if not vehicle_bps:
                print(f"Error: No valid vehicle blueprints found")
                return None

            # 选择随机车辆类型
            vehicle_bp = random.choice(vehicle_bps)

            # 设置生成位置为第一个轨迹点
            x, y, t = waypoints[0]
            z = self.get_ground_height(x, y)

            # 获取最近的waypoint以获得正确的朝向
            location = carla.Location(x=x, y=y, z=z)
            carla_waypoint = self.map.get_waypoint(location)

            if carla_waypoint:
                yaw = carla_waypoint.transform.rotation.yaw
            else:
                yaw = 0.0

            # 创建transform
            transform = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(yaw=yaw)
            )

            # 生成车辆
            print(f"Spawning vehicle {vehicle_id} at ({x:.2f}, {y:.2f}, {z:.2f})")
            vehicle = self.world.try_spawn_actor(vehicle_bp, transform)

            if vehicle is None:
                print(f"Failed to spawn vehicle {vehicle_id}")
                return None

            # 创建控制器
            controller = Controller.Controller(
                waypoints,
                self.args.lateral_controller,
                self.args.longitudinal_controller
            )

            # 设置车辆属性
            vehicle.set_autopilot(False)

            # 创建车辆数据对象
            vehicle_data = VehicleData(
                id=vehicle_id,
                actor=vehicle,
                controller=controller,
                waypoints=waypoints,
                x_history=[],
                y_history=[],
                yaw_history=[],
                time_history=[],
                speed_history=[],
                cte_history=[],
                he_history=[],
                latency_history=[]
            )

            print(f"Vehicle {vehicle_id} spawned successfully")
            return vehicle_data

        except Exception as e:
            print(f"Error spawning vehicle {vehicle_id}: {e}")
            return None

    def spawn_pedestrian(self, person_id: int, trajectory: List[Dict[str, float]]) -> Optional[PedestrianData]:
        """生成行人"""
        try:
            # 获取行人蓝图
            blueprint_library = self.world.get_blueprint_library()
            walker_bps = blueprint_library.filter('walker.pedestrian.*')

            if not walker_bps:
                print(f"Error: No pedestrian blueprints found")
                return None

            # 根据ID选择确定的行人类型
            bp_index = person_id % len(walker_bps)
            walker_bp = walker_bps[bp_index]

            # 设置生成位置
            start_waypoint = trajectory[0]
            x, y = start_waypoint['x'], start_waypoint['y']
            z = self.get_ground_height(x, y)

            # 创建transform
            transform = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(yaw=0.0)
            )

            # 生成行人
            print(f"Spawning pedestrian {person_id} at ({x:.2f}, {y:.2f}, {z:.2f})")
            walker = self.world.try_spawn_actor(walker_bp, transform)

            if walker is None:
                print(f"Failed to spawn pedestrian {person_id}")
                return None

            # 创建AI控制器
            controller_bp = blueprint_library.find('controller.ai.walker')
            controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)

            # 设置行人属性
            walker.set_simulate_physics(True)

            # 创建行人数据对象
            pedestrian_data = PedestrianData(
                id=person_id,
                actor=walker,
                controller=controller,
                trajectory=trajectory
            )

            print(f"Pedestrian {person_id} spawned successfully")
            return pedestrian_data

        except Exception as e:
            print(f"Error spawning pedestrian {person_id}: {e}")
            return None

    def update_vehicle(self, vehicle_data: VehicleData, current_time: float) -> bool:
        """更新车辆状态"""
        try:
            if vehicle_data.reached_end:
                return True

            vehicle = vehicle_data.actor

            # 获取车辆当前位置
            transform = vehicle.get_transform()
            x = transform.location.x
            y = transform.location.y
            z = transform.location.z
            yaw = math.radians(transform.rotation.yaw)

            # 获取车辆速度
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

            # 更新历史数据
            vehicle_data.x_history.append(x)
            vehicle_data.y_history.append(y)
            vehicle_data.yaw_history.append(yaw)
            vehicle_data.time_history.append(current_time)
            vehicle_data.speed_history.append(speed)

            # 在控制器启动前等待
            if current_time < WAIT_TIME_BEFORE_START:
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.steer = 0.0
                control.brake = 1.0
                vehicle.apply_control(control)
                return False

            # 调整控制器时间
            controller_time = current_time - WAIT_TIME_BEFORE_START

            # 找到最近的路径点
            waypoints = vehicle_data.waypoints
            closest_index = vehicle_data.closest_index

            # 向前搜索最近的路径点
            min_dist = float('inf')
            search_range = min(10, len(waypoints) - closest_index)

            for i in range(closest_index, min(closest_index + search_range, len(waypoints))):
                wx, wy, wt = waypoints[i]
                dist = math.sqrt((wx - x) ** 2 + (wy - y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    vehicle_data.closest_index = i

            # 向后搜索最近的路径点
            for i in range(max(0, vehicle_data.closest_index - 5), vehicle_data.closest_index):
                wx, wy, wt = waypoints[i]
                dist = math.sqrt((wx - x) ** 2 + (wy - y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    vehicle_data.closest_index = i

            # 更新控制器
            controller = vehicle_data.controller

            # 设置lookahead路径点
            lookahead_start = max(0, vehicle_data.closest_index - 1)
            lookahead_end = min(len(waypoints), vehicle_data.closest_index + 20)

            if lookahead_end > lookahead_start:
                lookahead_waypoints = waypoints[lookahead_start:lookahead_end]
                controller.update_waypoints(lookahead_waypoints)

            # 更新控制器状态
            controller.update_values(x, y, yaw, speed, controller_time, 1, min_dist)

            # 计算控制命令
            controller.update_controls()
            throttle, steer, brake = controller.get_commands()

            # 记录误差
            vehicle_data.cte_history.append(controller.get_crosstrack_error(x, y, waypoints))
            vehicle_data.he_history.append(controller.get_heading_error(waypoints, yaw))
            vehicle_data.latency_history.append(controller._latency if hasattr(controller, '_latency') else 0.0)

            # 应用控制
            control = carla.VehicleControl()
            control.throttle = max(0.0, min(1.0, throttle))
            control.steer = max(-1.0, min(1.0, steer))
            control.brake = max(0.0, min(1.0, brake))

            vehicle.apply_control(control)

            # 检查是否到达终点
            last_wp = waypoints[-1]
            dist_to_end = math.sqrt((last_wp[0] - x) ** 2 + (last_wp[1] - y) ** 2)

            if dist_to_end < DIST_THRESHOLD_TO_LAST_WAYPOINT:
                print(f"Vehicle {vehicle_data.id} reached destination")
                vehicle_data.reached_end = True
                return True

            return False

        except Exception as e:
            print(f"Error updating vehicle {vehicle_data.id}: {e}")
            return False

    def update_pedestrian(self, ped_data: PedestrianData, current_time: float) -> bool:
        """更新行人状态"""
        try:
            if ped_data.completed:
                return True

            trajectory = ped_data.trajectory

            # 如果轨迹为空或当前时间早于第一个轨迹点
            if not trajectory or current_time < trajectory[0]['timestamp']:
                return False

            # 如果当前时间晚于最后一个轨迹点
            if current_time >= trajectory[-1]['timestamp']:
                ped_data.completed = True
                return True

            # 找到当前时间对应的轨迹点
            idx = ped_data.current_waypoint_idx

            # 如果当前索引已经超出范围
            if idx >= len(trajectory):
                ped_data.completed = True
                return True

            # 找到包含当前时间的轨迹点区间
            for i in range(idx, len(trajectory) - 1):
                t1 = trajectory[i]['timestamp']
                t2 = trajectory[i + 1]['timestamp']

                if t1 <= current_time <= t2:
                    # 线性插值
                    alpha = (current_time - t1) / (t2 - t1) if t2 > t1 else 0.0
                    x1, y1 = trajectory[i]['x'], trajectory[i]['y']
                    x2, y2 = trajectory[i + 1]['x'], trajectory[i + 1]['y']

                    target_x = x1 + (x2 - x1) * alpha
                    target_y = y1 + (y2 - y1) * alpha
                    target_z = self.get_ground_height(target_x, target_y)

                    # 计算朝向
                    dx = target_x - x1
                    dy = target_y - y1
                    yaw = math.degrees(math.atan2(dy, dx)) if dx != 0 or dy != 0 else 0.0

                    # 设置目标位置
                    target_location = carla.Location(x=target_x, y=target_y, z=target_z)

                    # 使用控制器移动到目标位置
                    ped_data.controller.start()
                    ped_data.controller.go_to_location(target_location)

                    # 更新朝向
                    transform = ped_data.actor.get_transform()
                    transform.location = target_location
                    transform.rotation.yaw = yaw
                    ped_data.actor.set_transform(transform)

                    ped_data.current_waypoint_idx = i
                    break

            return False

        except Exception as e:
            print(f"Error updating pedestrian {ped_data.id}: {e}")
            return False

    def interpolate_waypoints(self, waypoints: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """对路径点进行插值"""
        if not waypoints or len(waypoints) < 2:
            return waypoints

        interpolated = []

        for i in range(len(waypoints) - 1):
            x1, y1, t1 = waypoints[i]
            x2, y2, t2 = waypoints[i + 1]

            # 添加原始点
            interpolated.append((x1, y1, t1))

            # 计算距离
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 如果需要插值
            if dist > INTERP_DISTANCE_RES:
                num_points = int(dist / INTERP_DISTANCE_RES)

                for j in range(1, num_points):
                    alpha = j / num_points
                    x = x1 + (x2 - x1) * alpha
                    y = y1 + (y2 - y1) * alpha
                    t = t1 + (t2 - t1) * alpha
                    interpolated.append((x, y, t))

        # 添加最后一个点
        interpolated.append(waypoints[-1])

        return interpolated

    def run_simulation(self):
        """运行仿真"""
        print("\n" + "=" * 60)
        print("Starting simulation with vehicle and pedestrian control")
        print("=" * 60)

        try:
            # 设置仿真开始时间
            self.sim_start_time = time.time()
            self.sim_running = True

            # 初始化统计
            completed_vehicles = 0
            completed_pedestrians = 0

            # 主仿真循环
            for frame in range(MAX_SIMULATION_FRAMES):
                if not self.sim_running:
                    break

                # 计算当前仿真时间
                current_time = frame * 0.05  # 假设20 FPS，每帧0.05秒

                # 1. 生成车辆（根据时间）
                for vehicle_id, waypoints in list(self.vehicle_waypoints.items()):
                    if vehicle_id not in self.vehicles and waypoints:
                        spawn_time = waypoints[0][2]

                        if current_time >= spawn_time:
                            # 对路径点进行插值
                            interpolated_waypoints = self.interpolate_waypoints(waypoints)

                            # 生成车辆
                            vehicle_data = self.spawn_vehicle(vehicle_id, interpolated_waypoints)

                            if vehicle_data:
                                self.vehicles[vehicle_id] = vehicle_data
                                print(f"Spawned vehicle {vehicle_id} at time {current_time:.2f}s")

                # 2. 生成行人（根据时间）
                for person_id, trajectory in list(self.pedestrian_trajectories.items()):
                    if person_id not in self.pedestrians and trajectory:
                        spawn_time = trajectory[0]['timestamp']

                        if current_time >= spawn_time:
                            # 生成行人
                            ped_data = self.spawn_pedestrian(person_id, trajectory)

                            if ped_data:
                                self.pedestrians[person_id] = ped_data
                                ped_data.spawn_time = current_time
                                print(f"Spawned pedestrian {person_id} at time {current_time:.2f}s")

                # 3. 更新所有车辆
                for vehicle_id, vehicle_data in list(self.vehicles.items()):
                    if not vehicle_data.reached_end:
                        completed = self.update_vehicle(vehicle_data, current_time)
                        if completed:
                            completed_vehicles += 1

                # 4. 更新所有行人
                for person_id, ped_data in list(self.pedestrians.items()):
                    if not ped_data.completed:
                        completed = self.update_pedestrian(ped_data, current_time)
                        if completed:
                            completed_pedestrians += 1

                # 5. 执行world tick
                self.world.tick()
                self.frame_count += 1

                # 6. 打印进度
                if frame % 100 == 0:
                    total_vehicles = len(self.vehicle_waypoints)
                    total_pedestrians = len(self.pedestrian_trajectories)

                    print(f"\nFrame {frame}, Time: {current_time:.2f}s")
                    print(f"Vehicles: {completed_vehicles}/{total_vehicles} completed")
                    print(f"Pedestrians: {completed_pedestrians}/{total_pedestrians} completed")

                # 7. 检查是否所有车辆和行人都已完成
                if (completed_vehicles >= len(self.vehicle_waypoints) and
                        completed_pedestrians >= len(self.pedestrian_trajectories)):
                    print("\nAll vehicles and pedestrians have completed their trajectories!")
                    break

                # 小延迟，避免过快的循环
                time.sleep(0.001)

            # 仿真结束
            self.sim_running = False

            # 打印最终统计
            print("\n" + "=" * 60)
            print("Simulation Completed")
            print("=" * 60)
            print(f"Total frames: {self.frame_count}")
            print(f"Total simulation time: {self.frame_count * 0.05:.2f}s")
            print(f"Vehicles spawned: {len(self.vehicles)}")
            print(f"Pedestrians spawned: {len(self.pedestrians)}")

            # 计算性能指标
            self.calculate_metrics()

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            print(f"\nError during simulation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def calculate_metrics(self):
        """计算性能指标"""
        try:
            print("\n计算性能指标...")

            # 计算车辆跟踪误差
            total_cte = 0.0
            total_he = 0.0
            count = 0

            for vehicle_data in self.vehicles.values():
                if vehicle_data.cte_history:
                    total_cte += sum(vehicle_data.cte_history) / len(vehicle_data.cte_history)
                    total_he += sum(vehicle_data.he_history) / len(vehicle_data.he_history)
                    count += 1

            if count > 0:
                avg_cte = total_cte / count
                avg_he = total_he / count
                print(f"\n车辆追踪指标:")
                print(f"  平均横穿航迹误差: {avg_cte:.4f} m")
                print(f"  平均航向误差: {avg_he:.4f} rad")

            # 计算行人完成率
            if self.pedestrian_trajectories:
                completed = sum(1 for p in self.pedestrians.values() if p.completed)
                total = len(self.pedestrian_trajectories)
                completion_rate = completed / total * 100 if total > 0 else 0

                print(f"\n行人指标:")
                print(f"  完成率: {completion_rate:.1f}% ({completed}/{total})")

            print("\n指标计算完成")

        except Exception as e:
            print(f"指标计算错误: {e}")

    def cleanup(self):
        """清理资源"""
        print("\nCleaning up resources...")

        # 停止仿真
        self.sim_running = False

        # 销毁行人
        for person_id, ped_data in list(self.pedestrians.items()):
            try:
                if ped_data.controller:
                    ped_data.controller.stop()
                    ped_data.controller.destroy()

                if ped_data.actor:
                    ped_data.actor.destroy()

                print(f"  Destroyed pedestrian {person_id}")
            except Exception as e:
                print(f"  Error destroying pedestrian {person_id}: {e}")

        # 销毁车辆
        for vehicle_id, vehicle_data in list(self.vehicles.items()):
            try:
                if vehicle_data.actor:
                    vehicle_data.actor.destroy()

                print(f"  Destroyed vehicle {vehicle_id}")
            except Exception as e:
                print(f"  Error destroying vehicle {vehicle_id}: {e}")

        # 恢复原始设置
        try:
            if self.original_settings and self.world:
                self.world.apply_settings(self.original_settings)

            # 禁用交通管理器同步模式
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(False)

            print("Restored original CARLA settings")
        except Exception as e:
            print(f"Error restoring settings: {e}")

        print("Cleanup completed")


def main():
    """主函数"""
    global WAYPOINTS_FILENAME, PEDESTRIAN_WAYPOINTS_FILENAME, SIMWEATHER
    parser = argparse.ArgumentParser(description='CARLA Vehicle and Pedestrian Control Simulation')

    parser.add_argument(
        '--host',
        default='localhost',
        help='IP of the host server (default: localhost)'
    )
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=2000,
        help='TCP port to listen to (default: 2000)'
    )
    parser.add_argument(
        '-lat_ctrl', '--lateral-controller',
        choices=['BangBang', 'PID', 'PurePursuit', 'Stanley', 'POP'],
        default='PID',
        help='Lateral controller type'
    )
    parser.add_argument(
        '-lon_ctrl', '--longitudinal-controller',
        choices=['PID', 'ALC'],
        default='ALC',
        help='Longitudinal controller type'
    )
    parser.add_argument(
        '--vehicle-file',
        default=WAYPOINTS_FILENAME,
        help='Vehicle waypoints file'
    )
    parser.add_argument(
        '--pedestrian-file',
        default=PEDESTRIAN_WAYPOINTS_FILENAME,
        help='Pedestrian trajectories file'
    )
    parser.add_argument(
        '--map',
        help='CARLA map name (e.g., Town01, Town02)'
    )
    parser.add_argument(
        '--weather',
        choices=list(WEATHERID.keys()),
        default='CLEARNOON',
        help='Weather condition'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=log_level
    )

    # 设置全局参数

    WAYPOINTS_FILENAME = args.vehicle_file
    PEDESTRIAN_WAYPOINTS_FILENAME = args.pedestrian_file

    if args.weather in WEATHERID:
        SIMWEATHER = WEATHERID[args.weather]

    print("=" * 60)
    print("CARLA Vehicle and Pedestrian Control Simulation")
    print("=" * 60)
    print(f"Host: {args.host}:{args.port}")
    print(f"Lateral Controller: {args.lateral_controller}")
    print(f"Longitudinal Controller: {args.longitudinal_controller}")
    print(f"Vehicle File: {WAYPOINTS_FILENAME}")
    print(f"Pedestrian File: {PEDESTRIAN_WAYPOINTS_FILENAME}")
    print(f"Weather: {args.weather}")
    print("=" * 60)

    try:
        # 创建仿真管理器
        sim_manager = SimulationManager(args)

        # 1. 连接到CARLA
        if not sim_manager.connect_to_carla():
            print("Failed to connect to CARLA. Exiting.")
            return

        # 2. 加载指定地图
        if args.map:
            print(f"Loading map: {args.map}")
            sim_manager.world = sim_manager.client.load_world(args.map)

        # 3. 设置仿真环境
        if not sim_manager.setup_simulation():
            print("Failed to setup simulation. Exiting.")
            return

        # 4. 加载轨迹数据
        if not sim_manager.load_vehicle_waypoints():
            print("Failed to load vehicle waypoints. Exiting.")
            return

        if not sim_manager.load_pedestrian_trajectories():
            print("Failed to load pedestrian trajectories. Exiting.")
            return

        # 5. 运行仿真
        sim_manager.run_simulation()

        print("\nSimulation finished successfully!")

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nProgram terminated.")


if __name__ == '__main__':
    main()