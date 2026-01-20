from unicodedata import category

import carla
import argparse
import scipy.io
import json
import numpy as np
import os
import random
import time
from fastdtw import fastdtw
from config import IntersectionConfig, town_configurations
from global_route_planner import GlobalRoutePlanner


def set_destination(start_location, end_location, _map, global_route):
    # 将起始点位置转换为对应的航点。
    start_waypoint = _map.get_waypoint(start_location)
    end_waypoint = _map.get_waypoint(end_location)
    # trace_route() 会返回一条完整的路径（由多个航点和道路选项组成的列表）。
    route_trace = trace_route(start_waypoint, end_waypoint, global_route)
    return route_trace


def trace_route(start_waypoint, end_waypoint, global_router):
    """
    Calculates the shortest route between a starting and ending waypoint.
    利用全局规划器计算从起点 start_waypoint 到终点 end_waypoint 的最短路径。
        :param start_waypoint (carla.Waypoint): initial waypoint
        :param end_waypoint (carla.Waypoint): final waypoint
    """
    start_location = start_waypoint.transform.location
    end_location = end_waypoint.transform.location
    # 返回值：路径规划结果，通常是一个由航点（carla.Waypoint）和对应的道路选项（RoadOption）组成的列表，代表从起点到终点的导航路径。
    return global_router._trace_route(start_location, end_location)


def interpolate_trajectory(trajectory, target_length, i, k):
    l = len(trajectory)
    n = target_length
    # 计算需要删除或增加的点的数量
    num_to_remove_or_add = abs(len(trajectory) - target_length)
    trajectory = np.array(trajectory)
    add_num = 0
    while len(trajectory) < target_length:
        new_points = []
        for i in range(len(trajectory) - 1):
            # 添加当前航点
            new_points.append(trajectory[i])
            # 插入当前航点和下一个航点的平均值
            new_points.append((trajectory[i] + trajectory[i + 1]) / 2)
            add_num += 1
            if add_num == num_to_remove_or_add:
                new_points.extend(trajectory[i + 1:])
                break
        else:
            # 添加最后一个航点
            new_points.append(trajectory[-1])
        trajectory = np.array(new_points)

    # 如果 trajectory 长度超过 target_length
    while len(trajectory) > target_length:
        current_len = len(trajectory)
        to_remove = current_len - target_length

        # 关键改进：动态调整步长
        step = max(1, round(current_len / to_remove))

        # 均匀选择要删除的点（保留首尾）
        remove_indices = list(range(step // 2, current_len, step))[:to_remove]
        trajectory = np.delete(trajectory, remove_indices, axis=0)

    return trajectory


def single_trajectory(vehicle_path, _map, final_vehicle_path, final_timestamp_list, timestamp_list, all_traj):
    # 添加轨迹
    unique_trajectory = vehicle_path[0]
    final_vehicle_path.append(unique_trajectory)

    # 添加时间
    unique_time = timestamp_list[0]
    final_timestamp_list.append(unique_time)

    # 初始化结果列表
    result = []
    # 遍历轨迹和时间列表
    for t_sublist, traj_sublist in zip(final_timestamp_list, final_vehicle_path):
        for t, tra in zip(t_sublist, traj_sublist):
            # 将时间和坐标组合成一个新的列表，并添加到结果列表中
            combine = [tra[0], tra[1], tra[2], t]
            result.append(combine)
    all_traj.append(result)


# def check_pedestrian_valid_area(location, _map):
#     """
#     检查行人是否在合理的活动区域
#     """
#     waypoint = _map.get_waypoint(location)
#
#     # 1. 检查是否在人行道上
#     if waypoint.lane_type == carla.LaneType.Sidewalk:
#         return True
#
#     # 2. 检查是否在十字路口（行人过马路）
#     if waypoint.is_junction:
#         junction_location = waypoint.get_junction()
#         distance_to_junction = location.distance(junction_location.bounding_box.location)
#         if distance_to_junction < 10:  # 在十字路口20米范围内
#             return True
#
#     return False


def is_valid_pedestrian_location(location, _map, max_distance=2.0):
    """
    验证位置是否适合行人
    """
    # 尝试获取人行道航点
    sidewalk_wp = _map.get_waypoint(
        location,
        lane_type=carla.LaneType.Sidewalk,
        project_to_road=False   # 不投影到道路
    )

    if sidewalk_wp:
        distance = location.distance(sidewalk_wp.transform.location)
        if distance <= max_distance:
            return True, distance, "sidewalk"

    # 检查是否是行人可以行走的特殊区域类型
    terrain_wp = _map.get_waypoint(
        location,
        lane_type=carla.LaneType.Any,  # 使用Any获取任何类型的航点
        project_to_road=False
    )

    # 检查是否是适合行人的地面类型
    if terrain_wp:
        distance = location.distance(terrain_wp.transform.location)

        # 检查是否属于行人可通行的区域类型
        if terrain_wp.lane_type in [
            carla.LaneType.Sidewalk,  # 人行道
            carla.LaneType.Parking,  # 停车场
            carla.LaneType.Border,  # 道路边缘
            carla.LaneType.Shoulder  # 路肩
        ]:
            if distance <= max_distance:
                return True, distance

    # 检查是否在十字路口范围内（行人过马路）
    road_wp = _map.get_waypoint(location)
    if road_wp and road_wp.is_junction:
        # 检查是否在十字路口范围内
        junction = road_wp.get_junction()
        if junction:
            distance_to_junction = location.distance(junction.bounding_box.location)
            junction_radius = max(junction.bounding_box.extent.x,
                                  junction.bounding_box.extent.y)

            # 十字路口加上适当缓冲区作为行人过街区域
            if distance_to_junction <= junction_radius + 5.0:   # 留出5米缓冲区
                return True, distance_to_junction, "junction_crossing"

    return False, None, "invalid"

def create_pedestrian_move(start_location, end_location, world, totaltime):
    """创建并控制行人从起点走到终点"""

    # 选择行人蓝图
    walker_bp = random.choice(
        world.get_blueprint_library().filter('walker.pedestrian.*')
    )

    # 设置行人属性
    walker_bp.set_attribute('is_invincible', 'false')

    # 生成行人
    transform = carla.Transform(start_location, carla.Rotation())
    walker = world.try_spawn_actor(walker_bp, transform)

    if walker is None:
        print("无法生成行人")
        return None
    else:
        print("行人生成成功")

    # 创建AI控制器
    controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)

    # 判断速度是否合理
    position_history = []
    current_location = walker.get_location()
    distance = current_location.distance(end_location)
    init_distance = distance
    # print("总距离为:", distance)
    speed = distance / totaltime
    if speed > 6:
        print("速度不合理，其值为: ", speed)
        return position_history
    else:
        print("速度合理，其值为", speed)

    # 开始移动
    controller.start()
    controller.go_to_location(end_location)
    controller.set_max_speed(speed)  # 设置速度

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True  # 开启同步模式
    settings.fixed_delta_seconds = 0.05  # 固定时间步长 0.05 秒
    world.apply_settings(settings)

    try:
        # 等待行人到达目的地
        while True:
            world.tick()

            # 每0.05秒记录一次位置
            current_location = walker.get_location()
            distance = current_location.distance(end_location)
            position_history.append(current_location)

            if distance < 1:  # 当距离小于1米时认为到达
                print("行人已到达目的地!")
                controller.stop()
                break

            if distance > (init_distance + 3): # 当距离越来越远时认为路径错误
                print("路径错误")
                controller.stop()
                position_history.clear()
                break

            # time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        # 清理
        controller.stop()
        walker.destroy()
        controller.destroy()
        # 恢复原始设置
        world.apply_settings(original_settings)

    return position_history


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-t', '--town',
        metavar='TOWN',
        default='Town10HD_Opt',
        choices=town_configurations.keys(),  # 限制用户只能输入已定义的城镇名
        help='Name of the town to use (e.g., Town01, Town10HD_Opt)'
    )
    args = argparser.parse_args()
    np.random.seed(42)
    random.seed(42)
    # 连接到Carla服务器
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    # 重新加载地图，重置仿真时间
    world = client.load_world(args.town, True)
    _map = world.get_map()
    # start_location = carla.Location(x=-41.668877, y=48.905540, z=0.600000)
    # end_location = carla.Location(x=74.798752, y=28.343533, z=0.600000)
    sampling_resolution = 1.5
    global_router = GlobalRoutePlanner(_map, sampling_resolution)

    # path = set_destination(start_location, end_location, _map, global_router)  # waypoint list

    # for ind, spawn_point in enumerate(path):
    #    world.debug.draw_string(spawn_point.transform.location, str(ind), life_time=1000, color=carla.Color(255, 0, 0))
    # 读取所有路口的轨迹文件
    data = scipy.io.loadmat('traj.mat')
    traj = data['traj']
    vehicle_traj = []  # 保存车辆轨迹
    person_traj = []  # 保存行人轨迹
    # 遍历外层的 traj cell 数组
    # traj_cell 是一个 1xN 的 cell 数组
    for i in range(len(traj[0])):      # 遍历外层的 1xN cell 数组
        inner_cell = traj[0][i][0]     # 获取第 i 个 cell
        vehicle_paths = []
        timestamp_list = []
        person_paths = []
        person_timestamp_list = []
        for j in range(len(inner_cell)):    # 遍历内层的 cell 数组
            struct = inner_cell[j]          # 获取第 j 个结构体
            category = struct[0][0][5]
            print(category)
            if category == 'person':
                positions = struct[0][0][2]  # 获取单个的轨迹
                timestamp = struct[0][0][4]  # 获取对应轨迹中航点的时间戳0
                _timestamp = [round(item[0], 2) for item in timestamp]
                person_paths.append(positions)
                person_timestamp_list.append(_timestamp)
            else:
                positions = struct[0][0][2]  # 获取单个的轨迹
                timestamp = struct[0][0][4]  # 获取对应轨迹中航点的时间戳0
                _timestamp = [round(item[0], 2) for item in timestamp]
                vehicle_paths.append(positions)
                timestamp_list.append(_timestamp)
            # positions = struct[0][0][2]     # 获取单个的轨迹
            # timestamp = struct[0][0][4]     # 获取对应轨迹中航点的时间戳0
            # _timestamp = [round(item[0], 2) for item in timestamp]
            # vehicle_paths.append(positions)
            # timestamp_list.append(_timestamp)

        # 排除非车辆轨迹，多目标跟踪和再识别的误差导致轨迹可能是非道路上的点
        # dis_trajectory = vehicle_paths[0][0]
        # dis_location = carla.Location(x=dis_trajectory[0], y=dis_trajectory[1], z=dis_trajectory[2])
        # nearest_waypoint = _map.get_waypoint(dis_location)
        # distance = dis_location.distance(nearest_waypoint.transform.location)
        # 计算 location 与最近的 waypoint 的距离，如果大于一定的阈值，则判断该轨迹是错误的
        # if distance >= 5:
        #     continue

        if not vehicle_paths:
            # 排除非行人轨迹，
            dis_trajectory = person_paths[0][0]
            dis_location = carla.Location(x=dis_trajectory[0], y=dis_trajectory[1], z=dis_trajectory[2])
            nearest_waypoint = _map.get_waypoint(dis_location)
            distance = dis_location.distance(nearest_waypoint.transform.location)
            # 计算 location 与最近的 waypoint 的距离，如果小于一定的阈值，则判断该轨迹是错误的
            is_valid, adistance, location_type = is_valid_pedestrian_location(dis_location, _map, 2.0)
            if distance < 4 and is_valid :
                continue

            # 存储行人最终的轨迹和时间
            final_person_path = []
            final_person_timestamp_list = []

            length = len(person_paths)
            # 处理只有一个路口出现该行人的情况
            # 在这个人的轨迹的末尾插值
            if length < 2:
                single_trajectory(person_paths, _map, final_person_path, final_person_timestamp_list, person_timestamp_list,
                                  person_traj)
                continue
            person_path = []
            for vehicle_p in person_paths:
                if len(vehicle_p) > 40:
                    trimmed_traj = vehicle_p[20:-20]
                    person_path.append(trimmed_traj)
                else:
                    person_path.append(vehicle_p)

            is_exception = False
            # 同一行人在多个路口出现，遍历 person_path，处理每一对相邻的轨迹
            for k in range(len(person_path) - 1):
                # 获取当前轨迹和下一段轨迹
                current_trajectory = person_path[k]
                next_trajectory = person_path[k + 1]

                # 获取当前轨迹的末尾位置
                start_loc = current_trajectory[-1]
                start_location = carla.Location(x=start_loc[0], y=start_loc[1], z=start_loc[2])
                # 获取下一段轨迹的开头位置
                end_loc = next_trajectory[0]
                end_location = carla.Location(x=end_loc[0], y=end_loc[1], z=end_loc[2])
                # 使用导航算法生成新轨迹
                # 处理无法到达的轨迹
                # end_waypoint = _map.get_waypoint(end_location)
                # start_waypoint = _map.get_waypoint(start_location)
                totaltime = person_timestamp_list[k + 1][0]-person_timestamp_list[k][-1]
                try:
                    position = create_pedestrian_move(start_location, end_location, totaltime)
                except Exception as e:
                    is_exception = True
                    if k > 0:
                        # 添加下一段轨迹和时间
                        final_person_path.append(person_path[k])
                        final_person_timestamp_list.append(person_timestamp_list[k])
                    else:
                        # 生成第一段轨迹的时候，无法到达，那么就当成只有一段轨迹处理，转换成 length < 2 的情况
                        single_trajectory(person_path, _map, final_person_path, final_person_timestamp_list, person_timestamp_list,
                                          person_traj)
                    break
                # 将 waypoint list 转换成[[x, y, z], [x, y, z], ...]
                trajectory = []
                for waypoint in position:
                    # 获取 waypoint 的位置
                    location = waypoint
                    # 提取 x, y, z 值
                    x = location.x
                    y = location.y
                    z = location.z
                    # 添加到轨迹列表中
                    trajectory.append([x, y, z])

                # 生成新轨迹对应的时间戳
                # 当前轨迹的最后一个时间戳
                current_end_time = person_timestamp_list[k][-1]
                # 下一段轨迹的第一个时间戳
                next_start_time = person_timestamp_list[k + 1][0]
                # 生成新轨迹的时间戳，间隔为 0.05
                _new_times = np.arange(round(current_end_time, 2) + 0.05, round(next_start_time, 2), 0.05)
                new_times = np.around(_new_times, decimals=2)
                # 如果轨迹数 > 时间数，删除两个点之间的点，从前往后，直到长度等于new_times的长度
                # 如果轨迹数 < 时间数, 则在两个点之间插值，从前往后，直到长度等于new_times的长度
                # 如果轨迹数 = 时间数，则不处理
                # 获取 new_times 的长度
                target_length = len(new_times)
                trajectory_len = len(trajectory)
                # 处理轨迹长度 == 1的时候
                if trajectory_len == 1:
                    _traj = trajectory[0]
                    loc = carla.Location(x=_traj[0], y=_traj[1], z=_traj[2])
                    waypoint = _map.get_waypoint(loc)
                    road_path = waypoint.next(0.2)
                    way_p = road_path[0]
                    location = way_p.transform.location
                    # 提取 x, y, z 值
                    x = location.x
                    y = location.y
                    z = location.z
                    # 添加到轨迹列表中
                    trajectory.append([x, y, z])

                # 如果 trajectory 的长度小于 new_times，进行插值
                in_value_trajectory = interpolate_trajectory(trajectory, target_length, i, k)

                # 将当前轨迹和新生成的轨迹添加到 final_person_path
                final_person_path.append(current_trajectory)
                final_person_path.append(in_value_trajectory)

                # 将当前时间戳和新生成的时间戳添加到 final_person_timestamp_list
                final_person_timestamp_list.append(person_timestamp_list[k])
                final_person_timestamp_list.append(new_times)

            if not is_exception:
                # 添加最后一段轨迹和时间戳
                final_person_path.append(person_path[-1])
                final_person_timestamp_list.append(person_timestamp_list[-1])
                result = []
                # 遍历轨迹和时间列表
                for t_sublist, traj_sublist in zip(final_person_timestamp_list, final_person_path):
                    for t, tra in zip(t_sublist, traj_sublist):
                        # 将时间和坐标组合成一个新的列表，并添加到结果列表中
                        combine = [tra[0], tra[1], tra[2], t]
                        result.append(combine)
                person_traj.append(result)
        else:
            # # 排除非车辆轨迹，多目标跟踪和再识别的误差导致轨迹可能是非道路上的点
            # dis_trajectory = vehicle_paths[0][0]
            # dis_location = carla.Location(x=dis_trajectory[0], y=dis_trajectory[1], z=dis_trajectory[2])
            # nearest_waypoint = _map.get_waypoint(dis_location)
            # distance = dis_location.distance(nearest_waypoint.transform.location)
            # # 计算 location 与最近的 waypoint 的距离，如果大于一定的阈值，则判断该轨迹是错误的
            # if distance >= 4:
            #     continue

            # 存储车辆最终的轨迹和时间
            final_vehicle_path = []
            final_timestamp_list = []

            length = len(vehicle_paths)
            # 处理只有一个路口出现该车辆的情况
            # 在这辆车的轨迹的末尾插值
            if length < 2:
                single_trajectory(vehicle_paths, _map, final_vehicle_path, final_timestamp_list, timestamp_list, vehicle_traj)
                continue
            vehicle_path = []
            for vehicle_p in vehicle_paths:
                if len(vehicle_p) > 40:
                    trimmed_traj = vehicle_p[20:-20]
                    vehicle_path.append(trimmed_traj)
                else:
                    vehicle_path.append(vehicle_p)

            is_exception = False
            # 同一车辆在多个路口出现，遍历 vehicle_path，处理每一对相邻的轨迹
            for k in range(len(vehicle_path) - 1):
                # 获取当前轨迹和下一段轨迹
                current_trajectory = vehicle_path[k]
                next_trajectory = vehicle_path[k + 1]

                # 获取当前轨迹的末尾位置
                start_loc = current_trajectory[-1]
                start_location = carla.Location(x=start_loc[0], y=start_loc[1], z=start_loc[2])
                # 获取下一段轨迹的开头位置
                end_loc = next_trajectory[0]
                end_location = carla.Location(x=end_loc[0], y=end_loc[1], z=end_loc[2])
                # 使用导航算法生成新轨迹
                # 处理无法到达的轨迹
                end_waypoint = _map.get_waypoint(end_location)
                start_waypoint = _map.get_waypoint(start_location)
                try:
                    interval_trajectory = set_destination(start_waypoint.transform.location, end_waypoint.transform.location, _map, global_router)
                except Exception as e:
                    is_exception = True
                    if k > 0:
                        # 添加下一段轨迹和时间
                        final_vehicle_path.append(vehicle_path[k])
                        final_timestamp_list.append(timestamp_list[k])
                    else:
                        # 生成第一段轨迹的时候，无法到达，那么就当成只有一段轨迹处理，转换成 length < 2 的情况
                        single_trajectory(vehicle_path, _map, final_vehicle_path, final_timestamp_list, timestamp_list, vehicle_traj)
                    break
                # 将 waypoint list 转换成[[x, y, z], [x, y, z], ...]
                trajectory = []
                for waypoint in interval_trajectory:
                    # 获取 waypoint 的位置
                    location = waypoint.transform.location
                    # 提取 x, y, z 值
                    x = location.x
                    y = location.y
                    z = location.z
                    # 添加到轨迹列表中
                    trajectory.append([x, y, z])

                # 生成新轨迹对应的时间戳
                # 当前轨迹的最后一个时间戳
                current_end_time = timestamp_list[k][-1]
                # 下一段轨迹的第一个时间戳
                next_start_time = timestamp_list[k + 1][0]
                # 生成新轨迹的时间戳，间隔为 0.05
                _new_times = np.arange(round(current_end_time, 2) + 0.05, round(next_start_time, 2), 0.05)
                new_times = np.around(_new_times, decimals=2)
                # 如果轨迹数 > 时间数，删除两个点之间的点，从前往后，直到长度等于new_times的长度
                # 如果轨迹数 < 时间数, 则在两个点之间插值，从前往后，直到长度等于new_times的长度
                # 如果轨迹数 = 时间数，则不处理
                # 获取 new_times 的长度
                target_length = len(new_times)
                trajectory_len = len(trajectory)
                # 处理轨迹长度 == 1的时候
                if trajectory_len == 1:
                    _traj = trajectory[0]
                    loc = carla.Location(x=_traj[0], y=_traj[1], z=_traj[2])
                    waypoint = _map.get_waypoint(loc)
                    road_path = waypoint.next(0.2)
                    way_p = road_path[0]
                    location = way_p.transform.location
                    # 提取 x, y, z 值
                    x = location.x
                    y = location.y
                    z = location.z
                    # 添加到轨迹列表中
                    trajectory.append([x, y, z])

                # 如果 trajectory 的长度小于 new_times，进行插值
                in_value_trajectory = interpolate_trajectory(trajectory, target_length, i, k)

                # 将当前轨迹和新生成的轨迹添加到 final_vehicle_path
                final_vehicle_path.append(current_trajectory)
                final_vehicle_path.append(in_value_trajectory)

                # 将当前时间戳和新生成的时间戳添加到 final_timestamp_list
                final_timestamp_list.append(timestamp_list[k])
                final_timestamp_list.append(new_times)

            if not is_exception:
                # 添加最后一段轨迹和时间戳
                final_vehicle_path.append(vehicle_path[-1])
                final_timestamp_list.append(timestamp_list[-1])
                result = []
                # 遍历轨迹和时间列表
                for t_sublist, traj_sublist in zip(final_timestamp_list, final_vehicle_path):
                    for t, tra in zip(t_sublist, traj_sublist):
                        # 将时间和坐标组合成一个新的列表，并添加到结果列表中
                        combine = [tra[0], tra[1], tra[2], t]
                        result.append(combine)
                vehicle_traj.append(result)

    # 读取groundtruth
    data_truth = scipy.io.loadmat('pedestrian_ground_truth.mat')
    traj_truth = data_truth['pedestrian_cells']
    all_ground_truth = []  # 保存所有的车辆轨迹
    for i in range(len(traj_truth[0])):
        struct = traj_truth[0][i]    # 获取第 i 个 struct
        positions = struct[0][0][1]     # 获取单个的轨迹
        array_3d = np.array(positions)  # 形状为(N,3,1)
        converted_data = array_3d.squeeze(axis=2).tolist()  # 移除最后一个维度
        all_ground_truth.append(converted_data)
        # all_ground_truth.append(array_3d)
    # 评估真实轨迹与跟踪轨迹
    # 保存对齐后的真实轨迹
    # 初始化累加器
    total_MPE = 0.0
    total_MaxPE = 0.0
    total_MFPE = 0.0
    total_overlap = 0.0
    valid_trajectories = 0  # 有效轨迹计数

    # for ge_traj in vehicle_traj:
    for ge_traj in person_traj:
        max_overlap = -1
        best_truth_traj = None
        track_points = np.array([[point[0], point[1]] for point in ge_traj])

        for truth_traj in all_ground_truth:
            truth_points = np.array([[point[0], point[1]] for point in truth_traj])

            min_length = min(len(track_points), len(truth_points))
            aligned_truth = truth_points[:min_length]
            aligned_track = track_points[:min_length]

            distance, _ = fastdtw(aligned_truth, aligned_track)
            max_distance = np.max(np.linalg.norm(aligned_truth - aligned_track, axis=1)) * min_length
            overlap_ratio = 1 - (distance / max_distance) if max_distance > 0 else 0
            overlap = max(0, min(1, overlap_ratio))

            if overlap > max_overlap:
                max_overlap = overlap
                best_truth_traj = truth_points

        if best_truth_traj is not None:
            min_len = min(len(track_points), len(best_truth_traj))
            truth_pts = best_truth_traj[:min_len]
            track_pts = track_points[:min_len]

            errors = np.linalg.norm(truth_pts - track_pts, axis=1)

            # 累加各项指标
            total_MPE += np.mean(errors)
            total_MaxPE += np.max(errors)
            total_MFPE += errors[-1] if len(errors) > 0 else 0
            total_overlap += max_overlap
            valid_trajectories += 1

    # 计算全局平均值
    if valid_trajectories > 0:
        avg_MPE = total_MPE / valid_trajectories
        avg_MaxPE = total_MaxPE / valid_trajectories
        avg_MFPE = total_MFPE / valid_trajectories
        avg_overlap = total_overlap / valid_trajectories

        print("\n==== 全局平均指标 ====")
        print(f"平均MPE: {avg_MPE:.4f} 米")
        print(f"平均MaxPE: {avg_MaxPE:.4f} 米")
        print(f"平均MFPE: {avg_MFPE:.4f} 米")
        print(f"平均重合度: {avg_overlap:.2%}")
    else:
        print("警告: 没有有效的轨迹匹配")


    # 保存全部行人轨迹到personwaypoint.txt
    with open('personWaypoints.txt', 'w') as file:

        # 遍历每个人的轨迹
        for person_index, tr in enumerate(person_traj):
            # 遍历轨迹中的每个数据点
            for point_index, (x, y, z, t) in enumerate(tr):
                # 使用数据点在轨迹内部的索引作为下标（但这里z值被省略了，如果您需要它，请添加）
                output_string = f"{person_index} {x} {y} {t}"
                # 写入文件，但不是在最后一个数据点后写入换行符
                if point_index != len(tr) - 1 or person_index != len(person_traj) - 1:
                    file.write(output_string + '\n')
                else:
                    file.write(output_string)  # 最后一个数据点，不添加换行符
    if os.path.exists('personWaypoints.txt'):
        print("✅ 文件已生成！路径:", os.path.abspath('personWaypoints.txt'))
    else:
        print("❌ 文件未生成！可能原因：写入失败或路径错误")

    # 将保存的轨迹按时间排序
    # 读取文件
    with open('personWaypoints.txt', 'r') as file:
        lines = file.readlines()

    sorted_lines = sorted(lines, key=lambda x: float(x.strip().split()[-1]))
    # 如果需要将排序后的结果写入新文件
    with open('personWaypoints.txt', 'w') as file:
        for line in sorted_lines:
            file.write(line)



    # 保存全部车辆轨迹到vehiclewaypoint.txt
    with open('vehicleWaypoints.txt', 'w') as file:

        # 遍历每辆车的轨迹
        for vehicle_index, tr in enumerate(vehicle_traj):
            # 遍历轨迹中的每个数据点
            for point_index, (x, y, z, t) in enumerate(tr):
                # 使用数据点在轨迹内部的索引作为下标（但这里z值被省略了，如果您需要它，请添加）
                output_string = f"{vehicle_index} {x} {y} {t}"
                # 写入文件，但不是在最后一个数据点后写入换行符
                if point_index != len(tr) - 1 or vehicle_index != len(vehicle_traj) - 1:
                    file.write(output_string + '\n')
                else:
                    file.write(output_string)  # 最后一个数据点，不添加换行符
    if os.path.exists('vehicleWaypoints.txt'):
        print("✅ 文件已生成！路径:", os.path.abspath('vehicleWaypoints.txt'))
    else:
        print("❌ 文件未生成！可能原因：写入失败或路径错误")

    # 将保存的轨迹按时间排序
    # 读取文件
    with open('vehicleWaypoints.txt', 'r') as file:
        lines = file.readlines()

    # 将每一行拆分为列表，并提取最后一列时间作为排序依据
    # 使用 lambda 函数对每一行的最后一列（索引为 -1）进行排序
    sorted_lines = sorted(lines, key=lambda x: float(x.strip().split()[-1]))
    # 如果需要将排序后的结果写入新文件
    with open('vehicleWaypoints.txt', 'w') as file:
        for line in sorted_lines:
            file.write(line)


if __name__ == '__main__':
    main()
