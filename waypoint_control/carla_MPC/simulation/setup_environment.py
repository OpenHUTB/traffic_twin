import sys
import random
import math
import os
import numpy as np
import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from interpolate import calc_spline_course_carla
from simulation.mpc_utils import calc_speed_profile
from simulation import config

def setup_environment():
    """
    设置CARLA环境，包含生成车辆和规划路径
    返回：
        world: CARLA世界对象
        map: CARLA地图对象
        ego: 自主车辆对象
        cx, cy, cyaw: 参考轨迹
        sp: 速度曲线
    """
    # 初始化CARLA客户端
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
    except Exception as e:
        print(f"客户端设置错误: {e}")
        sys.exit(1)
    
    # 设置起点和终点
    spawn_points = world.get_map().get_spawn_points()
    start_point = 87
    destination = 70
    start_loc = spawn_points[start_point].location
    dest_loc = spawn_points[destination].location
    
    print(f"起点位置: {start_loc}")
    print(f"终点位置: {dest_loc}")
    
    # 使用CARLA的全局路径规划器生成路线
    sample_res = 2
    grp = GlobalRoutePlanner(map, sample_res)
    way_points = grp.trace_route(start_loc, dest_loc)
    target_speed = 10  # 目标速度，m/s
    
    # 提取路径点坐标
    waypoints_x = []
    waypoints_y = []
    for this_wp, _ in way_points:
        world.debug.draw_string(this_wp.transform.location, 'O', draw_shadow=False,
                               color=carla.Color(r=0, g=255, b=0), life_time=100.0, persistent_lines=True)
        waypoints_x.append(this_wp.transform.location.x)
        waypoints_y.append(this_wp.transform.location.y)
    
    # 使用三次样条拟合轨迹
    cx, cy, cyaw, ck, s = calc_spline_course_carla(
        waypoints_x, 
        waypoints_y, 
        yaw=math.radians(way_points[0][0].transform.rotation.yaw), 
        ds=0.5
    )
    
    # 生成速度配置
    sp = calc_speed_profile(cx, cy, cyaw, target_speed)
    
    # 生成车辆
    vehicle_bp = blueprint_library.filter('vehicle.*')[3]
    first_wp = way_points[0][0]
    start_transform = carla.Transform(start_loc, first_wp.transform.rotation)
    ego = world.spawn_actor(vehicle_bp, start_transform)
    
    # 设置固定的帧率
    settings = world.get_settings()
    settings.fixed_delta_seconds = config.DT  # 设置模拟步长
    settings.synchronous_mode = True  # 启用同步模式
    world.apply_settings(settings)
    
    # 更新车辆轮距参数
    config.WB = ego.bounding_box.extent.y * 2 - 0.2
    print(f"车辆轮距: {config.WB}")
    
    return world, map, ego, cx, cy, cyaw, sp 