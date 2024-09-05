import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import argparse
from numpy import random


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
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        default=20,
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=20,
        type=int,
        help='Set the seed for pedestrians module')
    args = argparser.parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False

    # 随机选择车辆和蓝图时
    random.seed(args.seed if args.seed is not None else 22)
    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(args.tm_port)
        # 车辆的路线
        traffic_manager.set_random_device_seed(20)

        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        settings.synchronous_mode = True

        settings.fixed_delta_seconds = 0.05

        spawn_points = world.get_map().get_spawn_points()
        for ind, spawn_point in enumerate(spawn_points):
            ############################
            # spawn_point是transform对象#
            ############################
            # print(spawn_point)
            world.debug.draw_string(spawn_point.location, str(ind), life_time=1000, color=carla.Color(255, 0, 0))
        # vehicle_bp = world.get_blueprint_library().find('vehicle.audi.a2')
        vehicle_bp = world.get_blueprint_library().filter('*vehicle*')

        car_bp = [bp for bp in vehicle_bp if 'audi' in bp.id.lower()]
        # for bp in world.get_blueprint_library().filter('*vehicle*'):
        #     print(bp.id)
        # transform = carla.Transform(carla.Location(x=4624, y=561,z=0.3), carla.Rotation(yaw=180))
        vehicle_1 = world.try_spawn_actor(random.choice(car_bp), spawn_points[279])
        # # print("yaw1:", vehicle_1.get_transform().rotation.yaw)
        vehicle_2 = world.try_spawn_actor(random.choice(car_bp), spawn_points[235])
        vehicle_3 = world.try_spawn_actor(random.choice(car_bp), spawn_points[226])
        vehicle_4 = world.try_spawn_actor(random.choice(car_bp), spawn_points[51])
        # print("yaw2:", vehicle_2.get_transform().rotation.yaw)
        vehicle_1.set_autopilot()
        vehicle_2.set_autopilot()
        vehicle_3.set_autopilot()
        vehicle_4.set_autopilot()
        #    intersection_location = carla.Location(x=-47, y=20, z=2)
        # print(vehicle_1.id)
        # print(vehicle_2.id)
        # print(spawn_points[0])
        # print(spawn_points[93])
        # print(spawn_points[99])
        # world.debug.draw_string(carla.Location(x=-47, y=20, z=2), str('ss'), life_time=1000, color=carla.Color(255, 0, 0))
        initial_simulation_time = world.get_snapshot().timestamp.elapsed_seconds
        # print(spawn_points[230])
        # print(spawn_points[293])
        # print(spawn_points[294])
        # spectator = world.get_spectator()
        # print(spectator.get_transform())
        i = 0
        while True:

            # print("yaw1:", vehicle_1.get_transform().rotation.yaw)
            # 获取车辆的位置
            location1 = vehicle_1.get_location()
            x_1 = location1.x
            y_1 = location1.y

            location2 = vehicle_2.get_location()
            x_2 = location2.x
            y_2 = location2.y

            location3 = vehicle_3.get_location()
            x_3 = location3.x
            y_3 = location3.y

            location4 = vehicle_4.get_location()
            x_4 = location4.x
            y_4 = location4.y

            # 获取当前仿真时间
            current_simulation_time = world.get_snapshot().timestamp.elapsed_seconds
            # 计算相对于初始时间的经过时间
            elapsed_time = current_simulation_time - initial_simulation_time
            if i % 50 == 0:
                # # 打印车辆位置和仿真时间
                print(f"{0}, {x_1}, {y_1}, {elapsed_time}")
                print(f"{1}, {x_2}, {y_2}, {elapsed_time}")
                print(f"{2}, {x_3}, {y_3}, {elapsed_time}")
                print(f"{3}, {x_4}, {y_4}, {elapsed_time}")
            world.tick()
            i += 1
    finally:

        if not args.asynch and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
