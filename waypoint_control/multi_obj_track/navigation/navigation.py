import carla
import argparse
import scipy.io
import numpy as np
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
    args = argparser.parse_args()

    # 连接到Carla服务器
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    _map = world.get_map()
    # start_location = carla.Location(x=-41.668877, y=48.905540, z=0.600000)
    # end_location = carla.Location(x=74.798752, y=28.343533, z=0.600000)
    sampling_resolution = 0.735
    global_router = GlobalRoutePlanner(_map, sampling_resolution)

    # path = set_destination(start_location, end_location, _map, global_router)  # waypoint list

    # for ind, spawn_point in enumerate(path):
    #    world.debug.draw_string(spawn_point.transform.location, str(ind), life_time=1000, color=carla.Color(255, 0, 0))
    # 读取所有路口的轨迹文件
    data = scipy.io.loadmat('traj.mat')
    traj = data['traj']

    # 遍历外层的 traj cell 数组
    # traj_cell 是一个 1xN 的 cell 数组
    for i in range(traj.shape[1]):  # 遍历外层的 1xN cell 数组
        inner_cell = traj[0, i]  # 获取第 i 个 cell
        vehicle_path = []
        timestamp_list = []
        for j in range(inner_cell.shape[1]):  # 遍历内层的 cell 数组
            struct = inner_cell[0, j]  # 获取第 j 个结构体
            positions = struct['wrl_pos'][0, 0]  # 获取单个的轨迹
            timestamp = struct['timestamp'][0, 0]  # 获取对应轨迹中航点的时间戳
            vehicle_path.append(positions)
            timestamp_list.append(timestamp)

        # 存储最终的轨迹和时间
        final_vehicle_path = []
        final_timestamp_list = []

        length = len(vehicle_path)
        # 处理只有一个路口出现该车辆的情况
        # 在这辆车的轨迹的末尾插值
        if length < 2:
            unique_trajectory = vehicle_path[1]
            uni_end_loc = unique_trajectory[-1]
            end_location = carla.Location(x=uni_end_loc[0], y=uni_end_loc[1], z=uni_end_loc[2])
            # 返回距离 end_location 最近的 waypoint
            nearest_waypoint = _map.get_waypoint(end_location)
            road_path = nearest_waypoint.next_until_lane_end(0.5)
            ###############
            # 插入时间和轨迹 #
            ###############

        # 遍历 vehicle_path，处理每一对相邻的轨迹
        for k in range(len(vehicle_path) - 1):
            # 获取当前轨迹和下一段轨迹
            current_trajectory = vehicle_path[k]
            next_trajectory = vehicle_path[k + 1]
            # 获取当前轨迹的末尾位置
            end_loc = current_trajectory[-1]
            end_location = carla.Location(x=end_loc[0], y=end_loc[1], z=end_loc[2])
            # 获取下一段轨迹的开头位置
            start_loc = next_trajectory[0]
            start_location = carla.Location(x=start_loc[0], y=start_loc[1], z=start_loc[2])
            # 使用导航算法生成新轨迹
            interval_trajectory = set_destination(start_location, end_location, _map, global_router)

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
            new_times = np.arange(current_end_time + 0.05, next_start_time, 0.05)

            while len(trajectory) < len(new_times):
                # 减少采样分辨率以增加航点密度
                global_router._sampling_resolution -= 0.005
                # 重新生成航点
                interval_trajectory = set_destination(start_location, end_location, _map, global_router)
                # 更新trajectory为新的航点列表
                trajectory = interval_trajectory
                # 检查是否满足条件
                if len(trajectory) >= len(new_times):
                    break  # 如果航点数量已经足够，则退出循环

            # 将当前轨迹和新生成的轨迹添加到 final_vehicle_path
            final_vehicle_path.append(current_trajectory)
            final_vehicle_path.append(trajectory)

            # 将当前时间戳和新生成的时间戳添加到 final_timestamp_list
            final_timestamp_list.append(timestamp_list[k])
            final_timestamp_list.append(new_times)

        # 添加最后一段轨迹和时间戳
        final_vehicle_path.append(vehicle_path[-1])
        final_timestamp_list.append(timestamp_list[-1])

        ###############
        # 保存全部轨迹 #
        ###############


if __name__ == '__main__':
    main()
