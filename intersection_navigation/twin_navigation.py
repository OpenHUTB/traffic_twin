"""
    版本5.0：
       1.解决了部分车辆闯红灯问题
       2.删除了销毁非正常等待车辆模块
       3.减少局部规划得次数来解决卡顿问题
       4.解决车辆随意变道问题
"""
import carla
import time
import csv
import random
import math
import cProfile
import mysql.connector

from datetime import datetime
OFFSET_DISTANCE = 3          # 生成车辆失败时，将 location 后移3m
RES_SPAWN = 3                # 低距离生成车辆，避免车辆重复生成在一个地方
TIMEOUT_VALUE = 100          # 设置等待超时时间为10秒
LOWEST_SPEED = 0.1           # 设置车辆等待超时的最低速度
WAYPOINTS_FILENAME = 'Waypoints.txt'
FRAME_INTERVAL = 2           # 定义全局帧间隔

from agents.navigation.behavior_agent import BehaviorAgent
from twin_accuracy_evaluator import *


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
    # print(intersection_id)
    # print(lane)
    # print(direction)
    # print(results)
    loc_x, loc_y, loc_z, ro_yam = results[0]
    # 将 Decimal 转换为 float
    loc_x = float(loc_x)
    loc_y = float(loc_y)
    loc_z = float(loc_z) - RES_SPAWN
    ro_yaw = float(ro_yam)
    location = carla.Location(x=loc_x, y=loc_y, z=loc_z)
    return location, ro_yaw


def clear_source(need_to_destroy, waypoints_by_vehicle, intersection_time, vehicle_list, vehicle_waypoint_offset,
                 vehicle_plate_list, agent_list):
    for vehicle_id in need_to_destroy:
        del waypoints_by_vehicle[vehicle_id]
        del intersection_time[vehicle_id]
        del vehicle_list[vehicle_id]
        del vehicle_waypoint_offset[vehicle_id]
        del agent_list[vehicle_id]
        vehicle_plate_list.remove(vehicle_id)


# current_time 是str类型：20240603101544
def get_timestamp(current_time):
    # 将字符串转换为 datetime 对象
    target_date = datetime.strptime(current_time, "%Y%m%d%H%M%S")
    # 获取从1970年1月1日到这个时间的时间戳,float类型
    timestamp = target_date.timestamp()
    return int(timestamp)


def generate_random_vehicle_color():
    # 随机生成颜色值
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    random_color = f'{r}, {g}, {b}'
    return random_color


def is_front_vehicle_waiting_for_red_light(vehicle, max_distance=10, max_angle=30.0, recursion_depth=0, max_recursion_depth=3):
    """
    递归检测当前车辆同一车道上正前方是否有车辆在等待红灯，模仿 _affected_by_traffic_light 的逻辑。

    :param vehicle: 当前车辆
    :param max_distance: 检测前方车辆的最大距离
    :param max_angle: 前方车辆与当前车辆的最大角度差，确保只检测正前方的车辆
    :param recursion_depth: 当前递归深度，用于限制递归层次，避免无限递归
    :param max_recursion_depth: 最大递归深度
    :return: 如果前方车辆在等待红灯，返回 True，否则返回 False
    """
    if recursion_depth >= max_recursion_depth:
        return False

    # 获取世界中所有的车辆
    vehicle_list = vehicle.get_world().get_actors().filter('vehicle.*')
    ego_location = vehicle.get_location()
    ego_transform = vehicle.get_transform()
    ego_forward_vector = ego_transform.get_forward_vector()

    # 获取当前车辆所在的航点
    ego_waypoint = vehicle.get_world().get_map().get_waypoint(ego_location)

    for target_vehicle in vehicle_list:
        if target_vehicle.id == vehicle.id:
            continue

        # 获取目标车辆的航点，确保目标车辆与当前车辆在同一个车道上
        target_location = target_vehicle.get_location()
        target_waypoint = vehicle.get_world().get_map().get_waypoint(target_location)

        # 检查目标车辆是否位于同一车道上
        if ego_waypoint.road_id == target_waypoint.road_id and ego_waypoint.lane_id == target_waypoint.lane_id:
            distance = ego_location.distance(target_location)

            # 检测前方车辆的距离
            if distance < max_distance:
                # 确保目标车辆位于当前车辆的前方
                relative_location = target_location - ego_location
                forward_dot_product = ego_forward_vector.x * relative_location.x + ego_forward_vector.y * relative_location.y

                # 计算当前车辆与前方车辆的角度差
                angle = math.degrees(math.acos(forward_dot_product / (relative_location.length() * ego_forward_vector.length())))

                if forward_dot_product > 0 and angle < max_angle:  # 确保前方车辆在当前车辆前面，且角度在允许范围内
                    # 模仿 _affected_by_traffic_light 的逻辑，判断前方车辆的交通灯状态
                    traffic_light = target_vehicle.get_traffic_light()
                    if traffic_light is not None:
                        # 确保交通灯是红灯并且在合理的距离内
                        if traffic_light.get_state() == carla.TrafficLightState.Red:
                            traffic_light_location = traffic_light.get_location()
                            trigger_wp = vehicle.get_world().get_map().get_waypoint(traffic_light_location)

                            # 判断红灯是否在距离内并且在前方
                            if trigger_wp.transform.location.distance(target_location) < max_distance:
                                ve_dir = target_waypoint.transform.get_forward_vector()
                                wp_dir = trigger_wp.transform.get_forward_vector()
                                dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

                                if dot_ve_wp > 0:  # 车辆方向与交通灯方向一致，确保在正前方
                                    return True

                        else:
                            # 如果前方车辆不是在等红灯，递归检测前方车辆的前方是否有在等红灯的车辆
                            return is_front_vehicle_waiting_for_red_light(target_vehicle, max_distance, max_angle, recursion_depth + 1, max_recursion_depth)

    return False


def spawn_vehicle(vehicle_id, waypoints_by_vehicle, vehicle_waypoint_offset, cursor, filtered_vehicle_blueprints,
                  world, vehicle_list, vehicle_plate_list, agent_list):
    first_waypoint = waypoints_by_vehicle[vehicle_id][0]
    location, ro_yaw = query_location(first_waypoint, cursor)
    # 生成第一辆车
    transform = carla.Transform(location, carla.Rotation(yaw=ro_yaw))
    random_color = generate_random_vehicle_color()
    vehicle_bp = random.choice(filtered_vehicle_blueprints)
    # vehicle_bp.set_attribute('color', random_color)
    # 车辆生成失败时返回None
    vehicle = world.try_spawn_actor(vehicle_bp, transform)
    # 生成车辆位置被占，报碰撞错误，往后面移动3m重新生成车辆，直到生成成功
    while vehicle is None:
        transform.location -= transform.get_forward_vector() * OFFSET_DISTANCE
        vehicle = world.try_spawn_actor(vehicle_bp, transform)

    # 刷新生成车辆
    world.tick()
    # 添加车辆
    vehicle_list[vehicle_id] = vehicle
    vehicle_plate_list.append(vehicle_id)

    # 创建代理
    agent = BehaviorAgent(vehicle, behavior='normal')
    agent.frame_counter = 0           # 初始化帧计数器
    agent_list[vehicle_id] = agent

    # 初始化 last_action_time 为当前时间
    agent.last_action_time = time.time()

    second_waypoint = waypoints_by_vehicle[vehicle_id][1]
    location, ro_yam = query_location(second_waypoint, cursor)
    # 先行驶到第二个航点
    agent.set_destination(location)
    vehicle_waypoint_offset[vehicle_id] = 2  # 表示车辆在第二个路口位置


def main():
    # 连接到carla服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    world = client.get_world()
    origin_settings = world.get_settings()

    # 获取车辆蓝图库
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    # 过滤掉自行车
    filtered_vehicle_blueprints = [bp for bp in vehicle_blueprints if 'bike' not in bp.id and
                                   'omafiets' not in bp.id and
                                   'century' not in bp.id and
                                   'vespa' not in bp.id and
                                   'motorcycle' not in bp.id and
                                   'harley' not in bp.id and
                                   'yamaha' not in bp.id and
                                   'kawasaki' not in bp.id]

    # 保存全部车辆
    vehicle_list = {}
    # 保存全部agent
    agent_list = {}
    # 保存全部车辆的车牌编号
    vehicle_plate_list = []
    # 保存车辆当前在第几个航点位置
    vehicle_waypoint_offset = {}
    # 保存已销毁的车辆ID
    destroyed_vehicle_id = []
    # 保存当前要销毁的车辆
    need_to_destroy = []
    # 保存每辆车的生成和销毁时间（系统时间戳）
    vehicle_lifetimes = {}
    # 判断当前时刻是否需要销毁车辆
    is_to_destroy = False

    try:
        # 设置
        settings = world.get_settings()

        setting_config(settings, world)

        waypoints_file = WAYPOINTS_FILENAME
        waypoints_by_vehicle = {}
        intersection_time = {}

        # 后续还得嵌入while true循环, 边读边跑程序
        # 读取车辆数据,从文本读数据默认是字符串类型
        with open(waypoints_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                vehicle_id = int(row[0])   # int
                t = row[2]                 # str
                inter = row[4]             # str
                lane = int(row[5]) + 1     # int
                direct = row[6]            # str

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

        # 按车辆起点位置的时间排序,另外python3.6以后字典是有序的
        sorted_intersection_time = sort_spawn_time(intersection_time)

        # 提取车辆经过的中间路口，并用ID表示
        vehicle_middle_junctions = extract_middle_junction_ids(waypoints_by_vehicle)
        vehicle_actual_junctions = {}
        # 将 intersection_time 拆分为两部分 vehicle_id 和 起始时间start_time
        vehicle_id_list = []
        start_time = []

        # 遍历字典，提取车辆ID和经过第一个路口的时间
        for vehicle_id, times in sorted_intersection_time.items():
            vehicle_id_list.append(vehicle_id)  # 添加车辆id到列表
            start_time.append(times[0])  # 提取并添加第一个路口的时间

        # 连接到 MySQL 数据库
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='12345',
            database='test'
        )
        # 创建一个游标对象
        cursor = conn.cursor()

        # 默认运行代码则开始最早的一辆车或者一批车辆的孪生
        # 先生成最早的一辆车或者一批车辆，并设置路口终点
        # 从起始时间考虑，判断第一批有几辆车
        earliest_vehicle_num = start_time.count(start_time[0])
        # 根据 earliest_vehicle_num 来决定循环的次数
        count = 0
        for vehicle_id in vehicle_id_list:
            # 记录车辆生成的系统时间
            vehicle_lifetimes[vehicle_id] = (int(time.time()), None)
            spawn_vehicle(vehicle_id, waypoints_by_vehicle, vehicle_waypoint_offset, cursor, filtered_vehicle_blueprints,
                          world, vehicle_list, vehicle_plate_list, agent_list)
            count += 1
            # 最早一批次车辆生成完毕, 结束循环
            if earliest_vehicle_num == count:
                break

        # 记录最早批次生成时间戳,2024年678月
        first_stamp = get_timestamp(start_time[0])
        # 得到当前系统的时间戳，只考虑整数部分(2024年10月)
        first_system_stamp = int(time.time())
        # 记录下次该生成车辆在 vehicle_id_list 中的下标
        next_vehicle_num = earliest_vehicle_num
        while len(vehicle_list) > 0:  # 当车辆未全部销毁时继续循环
            num = 0
            # time.sleep(0.02)
            world.tick()
            # 接下来陆续按时间的先后顺序生成车辆
            # 下一批生成车辆的时间戳 - 上一批生成车辆的时间戳
            if next_vehicle_num < len(start_time):
                this_stamp = get_timestamp(start_time[next_vehicle_num])
                during_to_next_vehicle = this_stamp - first_stamp
                # 当前系统时间戳 - 上次生成车辆的系统时间戳
                this_system_stamp = int(time.time())
                during_system_time = this_system_stamp - first_system_stamp

                # 已经过了相同的时间, 该生成下一批次车辆
                if during_system_time == during_to_next_vehicle:
                    num = start_time.count(start_time[next_vehicle_num])
                    index = 0
                    # 生成车辆
                    for vehicle_id in vehicle_id_list[next_vehicle_num:]:
                        vehicle_lifetimes[vehicle_id] = (int(time.time()), None)
                        spawn_vehicle(vehicle_id, waypoints_by_vehicle, vehicle_waypoint_offset, cursor,
                                      filtered_vehicle_blueprints, world, vehicle_list, vehicle_plate_list, agent_list)
                        index += 1
                        if index == num:
                            break
                    next_vehicle_num += num
                    first_stamp = this_stamp
                    first_system_stamp = this_system_stamp

            # 更换车辆的终点坐标
            for vehicle_id, agent in agent_list.items():

                vehicle = vehicle_list[vehicle_id]
                # speed = vehicle.get_velocity().length()  # 获取车辆当前速度

                # 注意：先判断车辆到达了最后一个航点，选择将车辆销毁
                if agent.done() and vehicle_waypoint_offset[vehicle_id] >= len(waypoints_by_vehicle[vehicle_id]):

                    # 还得加个判断车辆是否已经销毁，否则，会重复销毁，报错
                    # RuntimeError: trying to operate on a destroyed actor; an actor's function was called,
                    # but the actor is already destroyed.
                    if vehicle_id not in destroyed_vehicle_id:
                        vehicle = vehicle_list[vehicle_id]
                        # 销毁车辆
                        vehicle.destroy()
                        vehicle_lifetimes[vehicle_id] = (vehicle_lifetimes[vehicle_id][0], int(time.time()))
                        is_to_destroy = True
                        need_to_destroy.append(vehicle_id)
                        destroyed_vehicle_id.append(vehicle_id)

                    # 不再更换终点坐标, 结束当前循环
                    continue

                # 检查车辆是否相互等待并判断超时
                # 如果车辆不在等待红灯，检查是否低速并可能死锁
                # if speed < LOWEST_SPEED:  # 速度接近于0
                #     elapsed_time = time.time() - agent.last_action_time
                #
                #     # 只有当前方车辆不在等待红灯时，才进行超时检查
                #     if elapsed_time > TIMEOUT_VALUE:
                #         print(f"车辆 {vehicle_id} 超时等待，正在销毁...")
                #         vehicle.destroy()
                #         destroyed_vehicle_id.append(vehicle_id)
                #         is_to_destroy = True
                #         need_to_destroy.append(vehicle_id)
                #         continue
                # else:
                #     # 车辆处于移动状态时重置等待时间
                #     agent.last_action_time = time.time()

                # 判断车辆是否到达终点，更换车辆终点
                if agent.done() and vehicle_waypoint_offset[vehicle_id] < len(waypoints_by_vehicle[vehicle_id]):
                    # 取出下一个航点的真实数据路口、车道、方向的列表
                    waypoint = waypoints_by_vehicle[vehicle_id][vehicle_waypoint_offset[vehicle_id]]
                    location, ro_yam = query_location(waypoint, cursor)

                    # 重新设置agent的终点
                    agent.set_destination(location)
                    vehicle_waypoint_offset[vehicle_id] += 1
                if agent.frame_counter % FRAME_INTERVAL == 0:
                    control = agent.run_step(debug=True)
                    vehicle = vehicle_list[vehicle_id]
                    vehicle.apply_control(control)
                # 增加帧计数器
                agent.frame_counter += 1

            if is_to_destroy:
                # 清空已销毁车辆所占资源
                clear_source(need_to_destroy, waypoints_by_vehicle, intersection_time, vehicle_list,
                             vehicle_waypoint_offset, vehicle_plate_list, agent_list)

            # 清空 need_to_destroy
            need_to_destroy.clear()
            is_to_destroy = False

            # 追踪车辆经过的实际路口
            vehicle_actual_junctions = track_vehicle_actual_junctions(vehicle_list, threshold=20.0)

        # 计算车辆生命周期差异
        time_differences = calculate_vehicle_lifespan(vehicle_lifetimes, intersection_time)
        # 将结果保存为csv
        save_lifespan_to_csv(time_differences, csv_filename='vehicle_lifespan_differences.csv')
        # 显示结果图
        plot_lifespan_differences_line(time_differences, image_filename='vehicle_lifespan_differences.png', limit=100)

        # 比较vehicle_middle_junctions 和 vehicle_actual_junctions 来评估孪生路径误差
        path_errors = evaluate_path_error(vehicle_middle_junctions, vehicle_actual_junctions)
        # 生成图表
        plot_path_errors(path_errors)

    finally:
        world.apply_settings(origin_settings)


if __name__ == '__main__':
    try:
        main()
        # cProfile.run('main()')
    except KeyboardInterrupt:
        print(' - Exited by user.')
