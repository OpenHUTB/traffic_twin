import carla
import argparse
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
    start_location = carla.Location(x=-41.668877, y=48.905540, z=0.600000)
    end_location = carla.Location(x=74.798752, y=28.343533, z=0.600000)
    sampling_resolution = 2.0
    global_router = GlobalRoutePlanner(_map, sampling_resolution)
    path = set_destination(start_location, end_location, _map, global_router)
    for ind, spawn_point in enumerate(path):
        world.debug.draw_string(spawn_point.transform.location, str(ind), life_time=1000, color=carla.Color(255, 0, 0))


if __name__ == '__main__':
    main()
