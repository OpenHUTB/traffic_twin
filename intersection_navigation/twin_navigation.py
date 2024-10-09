"""
    版本1.0：
        不考虑车辆生成先后顺序，先默认车辆是在同一时刻生成，再各自设置
    目标车辆的终点位置。
"""
import carla
import time
import csv
import math
import mysql.connector
import sqlite3
RES_DRIVING = 20.0  # 判断车辆还有下一个路口航点
WAYPOINTS_FILENAME = 'Waypoints.txt'

from agents.navigation.behavior_agent import BehaviorAgent

# def confirm_is_to_destination(current_location, destination):
#     x1, y1, z1 = current_location.x, current_location.y, current_location.z
#     x2, y2, z2 = destination.x, destination.y, destination.z
#     dis_start_to_destination = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
#     return dis_start_to_destination


def setting_config(settings, world):
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)


def sort_spawn_time(intersection_time):
    # 按照 list[0] 时间排序
    sorted_intersection_time = dict(sorted(intersection_time.items(), key=lambda item: item[1][0]))
    return sorted_intersection_time


def query_id(intersection, cursor):
    query = '''
                  SELECT IntersectionID FROM intersections
                  WHERE IntersectionName = %s
                  '''
    # print(intersection)
    cursor.execute(query, (intersection,))
    res = cursor.fetchall()
    return str(res[0][0])


def query_location(waypoint, cursor):
    # 根据真实数据中的路口、车道、方向，在指定位置生成车辆
    # 设置查询条件
    intersection = waypoint[0]
    lane = waypoint[1]
    direction = waypoint[2]
    intersection_id = query_id(intersection, cursor)
    # 查询数据库，找到车辆在对应路口、车道、方向的生成位置
    # 执行查询
    # print(f"Intersection ID: {intersection_id}, type: {type(intersection_id)}")
    # print(f"Lane: {lane}, type: {type(lane)}")
    # print(f"Direction: {direction}, type: {type(direction)}")

    query = '''
               SELECT x, y, z, yaw FROM zhongdianruanjianyuan
               WHERE IntersectionID = %s AND Lane = %s AND DirectionID = %s
               '''
    cursor.execute(query, (intersection_id, lane, direction))
    # 获取所有结果
    results = cursor.fetchall()
    print(intersection_id)
    print(lane)
    print(direction)
    print(results)
    loc_x, loc_y, loc_z, ro_yam = results[0]
    # 将 Decimal 转换为 float
    loc_x = float(loc_x)
    loc_y = float(loc_y)
    loc_z = float(loc_z)
    ro_yaw = float(ro_yam)
    # print(f"{type(loc_x)}")
    # print(loc_x)
    # print(f"{type(ro_yaw)}")
    # print(ro_yaw)
    location = carla.Location(x=loc_x, y=loc_y, z=loc_z)
    return location, ro_yaw


def clear_source(vehicle_id, waypoints_by_vehicle, intersection_time, vehicle_list, vehicle_waypoint_offset, vehicle_plate_list):
    del waypoints_by_vehicle[vehicle_id]
    del intersection_time[vehicle_id]
    del vehicle_list[vehicle_id]
    del vehicle_waypoint_offset[vehicle_id]
    vehicle_plate_list.remove(vehicle_id)


def main():
    # 连接到carla服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    world = client.get_world()
    origin_settings = world.get_settings()

    # 获取车辆蓝图库
    blueprint_library = world.get_blueprint_library()

    # 保存全部车辆
    vehicle_list = {}
    # 保存全部agent
    agent_list = {}
    # 保存全部车辆的车牌编号
    vehicle_plate_list = []
    # 保存车辆当前在第几个航点位置
    vehicle_waypoint_offset = {}
    # 销毁列表，存储要销毁的车辆id
    destroy_list = []

    try:
        # 设置
        settings = world.get_settings()
        setting_config(settings, world)

        waypoints_file = WAYPOINTS_FILENAME
        waypoints_by_vehicle = {}
        intersection_time = {}

        # 读取车辆数据
        with open(waypoints_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                vehicle_id = int(row[0])   # int
                t = row[2]                 # float
                inter = row[4]             # str
                lane = int(row[5]) + 1     # int
                direct = row[6]            # char

                # 将路口、车道和方向存储在 waypoints_by_vehicle 中，编号相同的数据存为列表
                if vehicle_id in waypoints_by_vehicle:
                    waypoints_by_vehicle[vehicle_id].append([inter, lane, direct])
                else:
                    waypoints_by_vehicle[vehicle_id] = [[inter, lane, direct]]

                # 将时间存储在 intersection_time 中，编号相同的数据存为列表
                if vehicle_id in intersection_time:
                    intersection_time[vehicle_id].append(t)
                else:
                    intersection_time[vehicle_id] = [t]

        # 按车辆起点位置的时间排序
        sorted_intersection_time = sort_spawn_time(intersection_time)
        # start_time = time.time()  # 记录当前真实时间

        # 连接到 MySQL 数据库
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='12345',
            database='test'
        )
        # 连接到 SQLite 数据库
        # conn = sqlite3.connect('carla_vehicles.db')
        # 创建一个游标对象
        cursor = conn.cursor()

        # 先生成全部车辆
        # 获取第一个路点生成车辆
        for vehicle_id, waypoints in waypoints_by_vehicle.items():
            first_waypoint = waypoints[0]
            location, ro_yaw = query_location(first_waypoint, cursor)

            # print(f"{type(ro_yaw)}")
            # print(ro_yaw)
            # 生成车辆
            transform = carla.Transform(location, carla.Rotation(yaw=ro_yaw))
            vehicle_bp = blueprint_library.find('vehicle.audi.tt')
            vehicle_bp.set_attribute('color', '100, 250, 250')
            vehicle = world.spawn_actor(vehicle_bp, transform)

            # 刷新生成车辆
            world.tick()
            # 添加车辆
            vehicle_list[vehicle_id] = vehicle
            vehicle_plate_list.append(vehicle_id)

            # 创建代理
            agent = BehaviorAgent(vehicle, behavior='normal')
            agent_list[vehicle_id] = agent

            second_waypoint = waypoints[1]
            location, ro_yam = query_location(second_waypoint, cursor)
            # 先行驶到第二个航点
            agent.set_destination(location)
            vehicle_waypoint_offset[vehicle_id] = 2  # 表示车辆在第一个位置

        while True:

            # # 模拟和真实时间一致
            # current_time = time.time()
            # elapsed_time = current_time - start_time  # 计算经过的真实时间
            # if elapsed_time >= 1:  # 每经过1秒推进仿真
            #     world.tick()
            #     start_time = time.time()  # 重置计时器
            time.sleep(0.05)
            world.tick()

            for vehicle_id, agent in agent_list.items():

                # 注意：先判断车辆到达了最后一个航点，选择将车辆销毁
                if agent.done() and vehicle_waypoint_offset[vehicle_id] >= len(waypoints_by_vehicle[vehicle_id]):
                    vehicle = vehicle_list[vehicle_id]
                    vehicle.destroy()

                    # 清空该车所占资源
                    # clear_source(vehicle_id, waypoints_by_vehicle, intersection_time, vehicle_list,
                    #              vehicle_waypoint_offset, vehicle_plate_list)
                    # 不再更换终点坐标，结束当前循环
                    continue
                # 判断车辆是否到达终点，更换车辆终点
                if agent.done() and vehicle_waypoint_offset[vehicle_id] < len(waypoints_by_vehicle[vehicle_id]):
                    # 取出下一个航点的真实数据路口、车道、方向的列表
                    waypoint = waypoints_by_vehicle[vehicle_id][vehicle_waypoint_offset[vehicle_id]]
                    location, ro_yam = query_location(waypoint, cursor)

                    # 重新设置agent的终点
                    agent.set_destination(location)
                    vehicle_waypoint_offset[vehicle_id] += 1

                control = agent.run_step(debug=True)
                vehicle = vehicle_list[vehicle_id]
                vehicle.apply_control(control)

    finally:
        world.apply_settings(origin_settings)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
