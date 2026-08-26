"""
    收集点云数据集
    1. 这里不需要过滤遮挡情况：裁剪的同时已经将点少于50的边界框过滤了
    2. 边界框需要处理，z轴需要调大一点
    3. 收集的数据集尽量在3000个左右
    4. 同步保存数据
"""
import time
import numpy as np
import carla
import os
import cv2
import random
import scipy.io
import math
from queue import Queue
from queue import Empty
from scipy.spatial.transform import Rotation as R
relativePose_lidar_to_egoVehicle = [0, 0, 1.3, 0, 0, 0, 0, 0, 0]
LIDAR_RANGE = 51.2   # 筛选可视距离雷达的车辆和行人
POINT_SAVE_TIME = 3000  # 保存数据数量


# 创建保存雷达数据的文件夹
def create_radar_folder():
    folder_name = f"train_data/train_model/points"
    # 检查文件夹是否已存在，若不存在则创建
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    return folder_name


# 创建保存标签数据的文件夹
def create_label_folder():
    folder_name = f"train_data/train_model/label"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    return folder_name


def recognize_vehicle_class(vehicle):
    blueprint = vehicle.type_id.lower()  # 获取车辆的蓝图名称并转换为小写
    # 定义需要识别为卡车的特定蓝图ID
    Truck_blueprints = [
        'vehicle.carlamotors.carlacola',
        'vehicle.carlamotors.european_hgv',
        'vehicle.tesla.cybertruck',
        'vehicle.carlamotors.firetruck',
        'vehicle.mitsubishi.fusorosa'
    ]
    # 检查蓝图名称是否在卡车列表中
    if blueprint in Truck_blueprints:
        return 'Truck'
    else:
        return "Car"


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


def save_point_label(world, location, lidar_to_world_inv, time_stamp, current_frame, lidar_yaw):
    # 获取雷达检测范围内的全部车辆和行人
    # 获取附近的所有车辆和行人
    vehicle_list = world.get_actors().filter("*vehicle*")
    pedestrian_list = world.get_actors().filter("*walker*")
    # 筛选出距离雷达小于 50 米的车辆和行人
    # def dist(v):
    #     return v.get_location().distance(location)
    # def dist(p):
    #     return p.get_location().distance(location)

    def dist(actor):
        return actor.get_location().distance(location)

    # 筛选出距离小于 LIDAR_RANGE 的车辆和行人
    vehicle_list = list(filter(lambda actor: dist(actor) < LIDAR_RANGE, vehicle_list))
    pedestrian_list = list(filter(lambda actor: dist(actor) < LIDAR_RANGE, pedestrian_list))
    # vehicle_list = list(filter(lambda v: dist(v) < LIDAR_RANGE, vehicle_list))
    # pedestrian_list = list(filter(lambda p: dist(p) < LIDAR_RANGE, pedestrian_list))
    # 按方向过滤车辆
    # vehicle_list = filter_vehicle_by_direction(vehicle_list, lidar_yaw, location, angle_tolerance=15, distance_threshold=30)

    car_labels = []  # Car 标签列表
    truck_labels = []  # Truck 标签列表
    pedestrian_labels = []  # Pedestrian 标签列表

    # 获取标签NX9
    for vehicle in vehicle_list:
        bounding_box = vehicle.bounding_box
        bbox_z = bounding_box.location.z
        location = vehicle.get_transform().location
        rotation = vehicle.get_transform().rotation
        bounding_box_location = np.array([location.x, location.y, bbox_z, 1])
        # 使用逆变换矩阵将位置从世界坐标系转换到雷达坐标系
        bounding_box_location_lidar = lidar_to_world_inv @ bounding_box_location  # 矩阵乘法
        bounding_box_location_lidar = bounding_box_location_lidar[:3]  # 去掉齐次坐标部分，得到三维坐标

        # 获取边界框的宽长高
        bounding_box_extent = bounding_box.extent
        length = 2 * bounding_box_extent.x
        width = 2 * bounding_box_extent.y
        height = 2 * bounding_box_extent.z

        bounding_box_rotation = np.array([rotation.yaw, rotation.pitch, rotation.roll])
        # 将 Euler 角（pitch, yaw, roll）转换为旋转矩阵（3x3）
        rotation_matrix_world = R.from_euler('zyx', bounding_box_rotation, degrees=True).as_matrix()
        # 使用逆变换矩阵将位置从世界坐标系转换到雷达坐标系
        rotation_matrix_lidar = lidar_to_world_inv[:3, :3] @ rotation_matrix_world
        rotation_lidar = R.from_matrix(rotation_matrix_lidar)
        euler_angles_lidar = rotation_lidar.as_euler('zyx', degrees=False)
        # 输出转换后的 pitch, yaw, roll
        yaw_lidar, pitch_lidar, roll_lidar = euler_angles_lidar
        # 构造标签数据（Nx9 格式）

        # 判断车辆的类别（Car, Truck）
        category = recognize_vehicle_class(vehicle)

        label = [
            bounding_box_location_lidar[0],  # x
            bounding_box_location_lidar[1],  # y
            bounding_box_location_lidar[2],  # z ,需要把z替换成bounding_box.z
            length,
            width,
            height,
            # pitch_lidar,  # pitch
            # roll_lidar,  # roll
            yaw_lidar,  # yaw
            category
        ]
        # # 判断车辆的类别（Car, Truck）
        # category = recognize_vehicle_class(vehicle)
        # 根据类别保存标签
        if category == "Car":
            car_labels.append(label)
        elif category == "Truck":
            truck_labels.append(label)


    # 获取行人标签
    step = 2  # 每隔1个元素遍历（步长为2）
    for pedestrian in pedestrian_list[1::step]:
    # for pedestrian in pedestrian_list:
        bounding_box = pedestrian.bounding_box

        bbox_z = bounding_box.location.z
        location = pedestrian.get_transform().location
        rotation = pedestrian.get_transform().rotation
        bounding_box_location = np.array([location.x, location.y, bbox_z, 1])
        # 使用逆变换矩阵将位置从世界坐标系转换到雷达坐标系
        bounding_box_location_lidar = lidar_to_world_inv @ bounding_box_location  # 矩阵乘法
        bounding_box_location_lidar = bounding_box_location_lidar[:3]  # 去掉齐次坐标部分，得到三维坐标


        # 获取边界框的宽长高
        bounding_box_extent = bounding_box.extent
        length = 2 * bounding_box_extent.x
        width = 2 * bounding_box_extent.y
        height = 2 * bounding_box_extent.z

        bounding_box_rotation = np.array([rotation.yaw, rotation.pitch, rotation.roll])
        # 将 Euler 角（pitch, yaw, roll）转换为旋转矩阵（3x3）
        rotation_matrix_world = R.from_euler('zyx', bounding_box_rotation, degrees=True).as_matrix()
        # 使用逆变换矩阵将位置从世界坐标系转换到雷达坐标系
        rotation_matrix_lidar = lidar_to_world_inv[:3, :3] @ rotation_matrix_world
        rotation_lidar = R.from_matrix(rotation_matrix_lidar)
        euler_angles_lidar = rotation_lidar.as_euler('zyx', degrees=False)
        # 输出转换后的 pitch, yaw, roll
        yaw_lidar, pitch_lidar, roll_lidar = euler_angles_lidar

        # 构造标签数据（Nx9 格式）
        label = [
            bounding_box_location_lidar[0],  # x
            bounding_box_location_lidar[1],  # y
            bounding_box_location_lidar[2] + height / 2,  # z ,需要把z替换成bounding_box.z
            length,
            width,
            height,
            # pitch_lidar,  # pitch
            # roll_lidar,  # roll
            yaw_lidar,  # yaw
            'Pedestrian'
        ]
        pedestrian_labels.append(label)  # 行人标签直接保存，无需分类


    # # 将 Car ， Truck 和 Pedestrian 数据转换为 NumPy 数组
    # car_labels = np.array(car_labels, dtype=object)
    # truck_labels = np.array(truck_labels, dtype=object)
    # pedestrian_labels = np.array(pedestrian_labels, dtype=object)
    # # 构造 MATLAB 格式的表格
    # label_data = {
    #     "Time": time_stamp,
    #     "Car": car_labels,  # Car 标签
    #     "Truck": truck_labels,  # Truck 标签
    #     "Pedestrian": pedestrian_labels  # Pedestrian 标签
    # }

    # 将所有类别的标签合并到一个列表中
    all_labels = []

    # 处理Car标签
    if len(car_labels) > 0:
        for label in car_labels:
            if len(label) >= 7:
                # 格式化数值为两位小数
                formatted_label = []
                for i, value in enumerate(label):
                    if i < 7:  # 前7个是数值
                        formatted_label.append(f"{float(value):.2f}")  # 格式化为两位小数
                    else:  # 第8个及以后是类别名称
                        formatted_label.append(str(value))

                # 如果只有7个字段，添加类别名
                if len(formatted_label) == 7:
                    formatted_label.append("Vehicle")

                all_labels.append(formatted_label)

    # 处理Truck标签
    if len(truck_labels) > 0:
        for label in truck_labels:
            if len(label) >= 7:
                formatted_label = []
                for i, value in enumerate(label):
                    if i < 7:
                        formatted_label.append(f"{float(value):.2f}")
                    else:
                        formatted_label.append(str(value))

                if len(formatted_label) == 7:
                    formatted_label.append("Truck")

                all_labels.append(formatted_label)

    # 处理Pedestrian标签
    if len(pedestrian_labels) > 0:
        for label in pedestrian_labels:
            if len(label) >= 7:
                formatted_label = []
                for i, value in enumerate(label):
                    if i < 7:
                        formatted_label.append(f"{float(value):.2f}")
                    else:
                        formatted_label.append(str(value))

                if len(formatted_label) == 7:
                    formatted_label.append("Pedestrian")

                all_labels.append(formatted_label)

    return all_labels
    # return label_data


# 定义回调函数来保存雷达点云数据
def save_radar_data(radar_data, world, location, lidar_to_world_inv, lidar_yaw, sensor_queue):
    # 时间戳
    timestamp = world.get_snapshot().timestamp.elapsed_seconds
    # 获取当前帧编号
    current_frame = radar_data.frame
    # 保存车辆和行人标签
    # label_data = save_point_label(world, location, lidar_to_world_inv, timestamp, current_frame, lidar_yaw)
    all_labels = save_point_label(world, location, lidar_to_world_inv, timestamp, current_frame, lidar_yaw)

    # 保存点云数据
    # 获取雷达数据并将其转化为numpy数组
    points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (len(points) // 4, 4))
    # location = points[:, :3]
    # 提取 points 的前四列
    location = points[:, :4]
    # 将 location 转换为 float64（即 double 类型）
    location = location.astype(np.float32)
    # intensity = points[:, 3].reshape(-1, 1).astype(np.float64)  # 获取强度数据（第四通道）
    # intensity_scaled = np.round(intensity * 255).astype(np.uint8)
    # count = location.shape[0]
    # 计算 x 的范围
    # x_limits = [np.min(location[:, 0]), np.max(location[:, 0])]  # x 轴的最小值和最大值
    # y_limits = [np.min(location[:, 1]), np.max(location[:, 1])]  # y 轴的最小值和最大值
    # z_limits = [np.min(location[:, 2]), np.max(location[:, 2])]  # z 轴的最小值和最大值

    # 如果 timestamp 是单个值，创建重复的数组
    # if np.isscalar(timestamp):
    #     timestamp_array = np.full((location.shape[0], 1), timestamp, dtype=np.float32)
    # else:
    #     # 如果 timestamp 已经是数组，确保形状正确
    #     timestamp_array = timestamp.reshape(-1, 1)

    # 水平拼接 location 前四列和 timestamp
    # datalog = np.column_stack([location, timestamp_array])
    datalog = location

    # # 创建存储数据的文件夹（每个雷达一个文件夹）
    # radar_folder = create_radar_folder()
    # file_name = os.path.join(radar_folder, f"{current_frame}.mat")
    # LidarData = {
    #     'PointCloud': {
    #         'Location': location,
    #         'Count': count,
    #         'XLimits': x_limits,
    #         'YLimits': y_limits,
    #         'ZLimits': z_limits,
    #         'Color': [],
    #         'Normal': [],
    #         'Intensity': intensity
    #     },
    #     'Timestamp': timestamp,
    #     'Pose': {
    #         'Position': relativePose_lidar_to_egoVehicle[:3],
    #         'Velocity': [0, 0, 0],
    #         'Orientation': [0, 0, 0]
    #     },
    #     'Detections': []
    # }
    # datalog = {
    #     'LidarData': LidarData
    # }
    # 将点云数据保存为 .mat 文件
    # 使用 scipy.io.savemat 保存数据，MATLAB 可以读取的格式
    # scipy.io.savemat(file_name, {'datalog': datalog})
    sensor_queue.put((datalog, all_labels))


def setup_sensors(world, addtion_param, transform, lidar_to_world_inv, data_struct_list):
    lidar = None
    location = carla.Location(x=-46, y=21, z=1)
    lidar_yaw = transform.rotation.yaw
    # 配置LiDAR传感器
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('dropoff_general_rate', '0.1')
    lidar_bp.set_attribute('dropoff_intensity_limit',
                           lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
    lidar_bp.set_attribute('dropoff_zero_intensity',
                           lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])


    for key in addtion_param:
        lidar_bp.set_attribute(key, addtion_param[key])

    # 创建雷达并绑定回调
    lidar = world.spawn_actor(lidar_bp, transform)
    lidar.listen(lambda data: save_radar_data(data, world, location, lidar_to_world_inv, lidar_yaw, data_struct_list))
    return lidar


# 生成自动驾驶车辆
def spawn_autonomous_vehicles(world, tm, num_vehicles=50, random_seed=42):
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)

    vehicle_list = []
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    filter_bike_blueprinter = filter_vehicle_blueprinter(vehicle_blueprints)
    for _ in range(num_vehicles):
        # 随机选择一个位置
        spawn_point = world.get_map().get_spawn_points()
        if len(spawn_point) == 0:
            print("No spawn points available!")
            return []

        # 选择一个随机位置生成车辆
        transform = spawn_point[np.random.randint(len(spawn_point))]
        vehicle_bp = random.choice(filter_bike_blueprinter)
        vehicle = world.try_spawn_actor(vehicle_bp, transform)
        if vehicle is None:
            continue
        # 配置自动驾驶
        vehicle.set_autopilot(True)  # 启动自动驾驶模式
        # 不考虑交通灯
        tm.ignore_lights_percentage(vehicle, 100)
        vehicle_list.append(vehicle)
        print(f"Spawned vehicle: {vehicle.id}")

    return vehicle_list


# 生成随机运动行人
def spawn_autonomous_pedestrians(world, num_pedestrians=120, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    pedestrian_list = []

    # 获取普通行人蓝图（排除特殊类型）
    walker_bps = [
        bp for bp in world.get_blueprint_library().filter('walker.pedestrian*')
        if not bp.id.split('.')[-1] in {'child', 'skeleton'}
    ]


    for _ in range(num_pedestrians):
        # 获取安全生成位置
        spawn_point = None
        for _ in range(3):  # 最多尝试3次
            location = world.get_random_location_from_navigation()
            if location and 0 < location.z < 1.0:
                spawn_point = carla.Transform(location)
                break
        if not spawn_point:
            continue

        # 生成行人
        bp = random.choice(walker_bps)
        pedestrian = world.try_spawn_actor(bp, spawn_point)
        if not pedestrian:
            continue


        # 通过Actor接口启用物理
        try:
            pedestrian.set_simulate_physics(True)
            world.tick()  # 同步模式下必须tick
        except RuntimeError as e:
            print(f"设置物理失败: {e}")
            pedestrian.destroy()
            continue

        controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        controller = world.spawn_actor(controller_bp, carla.Transform(), pedestrian)
        controller.start()  # 启用自动行走
        controller.go_to_location(world.get_random_location_from_navigation())  # 设置目标点

        # 只将行人添加到列表，控制器不保存
        pedestrian_list.append(pedestrian)

        print(f"Spawned pedestrian: {pedestrian.id}")

    return pedestrian_list

# 生成自动驾驶车辆
def spawn_autonomous_vehicles_hutb(world, tm, junction_weights, num_vehicles=50, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    vehicle_list = []

    carla_map = world.get_map()
    filter_bike_blueprinter = filter_vehicle_blueprinter(world.get_blueprint_library().filter('vehicle.*'))
    all_waypoints = carla_map.generate_waypoints(distance=2.0)

    # 提取各个路口外的入口点
    junction_entries = {jid: [] for jid in junction_weights.keys()}
    processed_jids = set()

    for wp in all_waypoints:
        if wp.is_junction and wp.junction_id in junction_weights:
            jid = wp.junction_id
            if jid not in processed_jids:
                processed_jids.add(jid)
                junction = wp.get_junction()
                waypoint_pairs = junction.get_waypoints(carla.LaneType.Driving)
                for entry_wp, _ in waypoint_pairs:
                    outside_wps = entry_wp.previous(15.0)
                    if outside_wps:
                        junction_entries[jid].append(outside_wps[0])

    valid_jids = {jid: w for jid, w in junction_weights.items() if len(junction_entries[jid]) > 0}
    if not valid_jids:
        print("警告: 未检测到有效的路口入口点！")
        return []

    # 按加权计算各路口车辆配额
    total_weight = sum(valid_jids.values())
    normalized_weights = {jid: w / total_weight for jid, w in valid_jids.items()}

    assigned_counts = {}
    remaining_vehicles = num_vehicles
    for jid, weight in normalized_weights.items():
        count = int(num_vehicles * weight)
        assigned_counts[jid] = count
        remaining_vehicles -= count

    if remaining_vehicles > 0:
        assigned_counts[max(normalized_weights, key=normalized_weights.get)] += remaining_vehicles

    # 按配额生成车辆
    for start_jid, count in assigned_counts.items():
        entry_points = junction_entries[start_jid]
        if not entry_points:
            continue

        for _ in range(count):
            start_wp = random.choice(entry_points)
            transform = start_wp.transform
            transform.location.z += 0.5

            vehicle_bp = random.choice(filter_bike_blueprinter)
            vehicle = world.try_spawn_actor(vehicle_bp, transform)

            if vehicle:
                vehicle.set_autopilot(True)
                tm.ignore_lights_percentage(vehicle, 100)
                # 已移除 GlobalRoutePlanner 和 tm.set_path 跨路口导航逻辑
                vehicle_list.append(vehicle)

    print(f"成功按加权在各路口外生成了 {len(vehicle_list)} 辆车（本地行驶，无跨路口导航）。")
    return vehicle_list


def create_pedestrian_generator(world, seed=42):
    # 获取行人蓝图（排除小孩）
    all_walkers = world.get_blueprint_library().filter('walker.pedestrian.*')
    kid_ids = ['walker.pedestrian.0015', 'walker.pedestrian.0016', 'walker.pedestrian.0017']

    adult_bps = [bp for bp in all_walkers if bp.id not in kid_ids]

    # 排序并打乱
    adult_bps.sort(key=lambda x: x.id)
    local_rng = random.Random(seed)
    local_rng.shuffle(adult_bps)

    # 使用 yield 循环输出模型
    for bp in itertools.cycle(adult_bps):
        yield bp


def generate_pedestrian_trajectories(world, junction_weights, num_pedestrians=100, seed=2024):
    """
    按路口权重生成行人的初始位置和目标方向（避开路中央）
    """
    carla_map = world.get_map()
    all_wps = carla_map.generate_waypoints(distance=2.0)

    # 提取各个路口内的点
    junction_wps = {jid: [] for jid in junction_weights.keys()}
    for wp in all_wps:
        if wp.is_junction and wp.junction_id in junction_weights:
            junction_wps[wp.junction_id].append(wp)

    # 寻找人行道点或向外侧偏移
    target_locations = {jid: [] for jid in junction_weights.keys()}
    for jid, wps in junction_wps.items():
        for wp in wps:
            # 尝试获取右侧车道，如果是人行道就使用它
            right_lane = wp.get_right_lane()
            if right_lane and right_lane.lane_type == carla.LaneType.Sidewalk:
                target_locations[jid].append(right_lane.transform.location + carla.Location(z=0.2))
            else:
                # 否则直接向车道右侧垂直偏移 3.5 米，强行移出机动车道
                right_vec = wp.transform.get_right_vector()
                target_locations[jid].append(wp.transform.location + right_vec * 3.5 + carla.Location(z=0.2))

    valid_jids = {jid: w for jid, w in junction_weights.items() if len(target_locations[jid]) > 0}
    if not valid_jids:
        return []

    # 按加权计算各路口行人配额
    total_weight = sum(valid_jids.values())
    assigned_counts = {}
    remaining = num_pedestrians

    for jid, weight in valid_jids.items():
        count = int(num_pedestrians * (weight / total_weight))
        assigned_counts[jid] = count
        remaining -= count

    if remaining > 0:
        assigned_counts[max(valid_jids, key=lambda k: valid_jids[k] / total_weight)] += remaining

    # 4. 生成脚本数据（恢复为起点+终点的形式）
    local_rng = random.Random(seed)
    generated_script = []

    for jid, count in assigned_counts.items():
        locs = target_locations[jid]
        if len(locs) < 2:
            continue

        for _ in range(count):
            spawn_loc = local_rng.choice(locs)
            dest_loc = local_rng.choice(locs)  # 在同路口内随便选一个点作为运动方向

            speed = round(local_rng.uniform(1.1, 1.5), 2)
            generated_script.append({
                "spawn_point": carla.Transform(spawn_loc),
                "destination": dest_loc,
                "speed": speed
            })

    return generated_script


# 生成随机运动行人
def spawn_autonomous_pedestrians_hutb(world, junction_weights, num_pedestrians=100, random_seed=20):
    random.seed(random_seed)
    np.random.seed(random_seed)
    pedestrian_list = []

    auto_pedestrian_script = generate_pedestrian_trajectories(world, junction_weights, num_pedestrians, seed=random_seed)
    pedestrian_gen = create_pedestrian_generator(world, seed=random_seed)

    spawned_info = []
    for script_data in auto_pedestrian_script:
        walker_bp = next(pedestrian_gen)
        walker_actor = world.try_spawn_actor(walker_bp, script_data["spawn_point"])
        if walker_actor:
            pedestrian_list.append(walker_actor)
            spawned_info.append((walker_actor, script_data))

    print(f"成功按权重 Spawn 了 {len(pedestrian_list)} 名行人（避开路中央）。")
    world.tick()

    # 恢复你最开始的机械直线运动控制器逻辑
    for walker_actor, script_data in spawned_info:
        spawn_loc = script_data["spawn_point"].location
        dest_loc = script_data["destination"]
        dx = dest_loc.x - spawn_loc.x
        dy = dest_loc.y - spawn_loc.y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist > 0:
            direction = carla.Vector3D(x=dx / dist, y=dy / dist, z=0.0)
            control = carla.WalkerControl(
                direction=direction,
                speed=script_data["speed"]
            )
            walker_actor.apply_control(control)

    return pedestrian_list


# 主函数
def main():
    # 连接到Carla服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    map_name = 'HutbCarlaCity'
    world = client.load_world(map_name)

    # 仿真设置
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05
    settings.synchronous_mode = True
    world.apply_settings(settings)
    print("Connected to Carla server!")

    # 创建交通管理器
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    camera_dict = {}
    lidar = None
    vehicles = []
    addtion_param = {
        'channels': '128',
        'range': '200',
        'points_per_second': '4000000',
        'rotation_frequency': '20'
    }
    try:
        # 设置随机种子
        random_seed = 20
        # 静止 ego_vehicle 的位置
        ego_transform = carla.Transform(carla.Location(x=-46, y=21, z=1), carla.Rotation(pitch=0, yaw=90, roll=0))
        # 定义 5 个路口 ID
        target_junction_weights = {
            2121: 0.3,
            398: 0.1,
            576: 0.3,
            626: 0.1,
            510: 0.2
        }
        if map_name in ["Town01", "Town10HD_Opt"]:
            # 生成自动驾驶车辆
            vehicles = spawn_autonomous_vehicles(world, tm, num_vehicles=50, random_seed=random_seed)
            # 生成行人
            pedestrians = spawn_autonomous_pedestrians(world, num_pedestrians=100, random_seed=20)
        else:
            # 生成自动驾驶车辆
            vehicles = spawn_autonomous_vehicles_hutb(world, tm, target_junction_weights, num_vehicles=50,random_seed=random_seed)
            # 生成行人
            pedestrians = spawn_autonomous_pedestrians_hutb(world, target_junction_weights, num_pedestrians=100,random_seed=20)
        #启动行人碰撞
        for pedestrian in pedestrians:
            if "walker.pedestrian." in pedestrian.type_id:
                pedestrian.set_collisions(True)
                pedestrian.set_simulate_physics(True)
        # 设置理想化的雷达位置
        lidar_transform = carla.Transform(carla.Location(x=-46, y=21, z=1.8), carla.Rotation(pitch=0, yaw=90, roll=0))
        # 获取雷达到世界的变换矩阵（4x4矩阵）
        lidar_to_world = np.array(lidar_transform.get_matrix())
        lidar_to_world_inv = np.linalg.inv(lidar_to_world)
        sensor_queue = Queue()
        # 启动雷达传感器
        lidar = setup_sensors(world, addtion_param, lidar_transform, lidar_to_world_inv, sensor_queue)
        folder_index = 0

        # 同步保存雷达数据
        for _ in range(POINT_SAVE_TIME):
            world.tick()
            datalog, label = sensor_queue.get(True, 1.0)

            # # 开始保存
            # # 创建存储数据的文件夹（每个雷达一个文件夹）
            # radar_folder = create_radar_folder()
            # file_name = os.path.join(radar_folder, f"{folder_index}.mat")
            # # 使用 scipy.io.savemat 保存数据，MATLAB 可以读取的格式
            # scipy.io.savemat(file_name, {'datalog': datalog})
            #
            # label_folder = create_label_folder()
            # file_name = os.path.join(label_folder, f"{folder_index}.mat")
            # # 保存为 .mat 文件
            # scipy.io.savemat(file_name, {"LabelData": label})
            #
            # time.sleep(0.05)
            # folder_index += 1

            # 生成6位数字的文件名
            file_num = f"{folder_index:06d}"
            # 1. 直接保存 datalog 为 .npy
            radar_folder = create_radar_folder()
            np.save(os.path.join(radar_folder, f"{file_num}.npy"), datalog)
            # 2. 保存 label 为 .txt
            label_folder = create_label_folder()
            with open(os.path.join(label_folder, f"{file_num}.txt"), 'w') as f:

                # 处理不同的数据结构
                if isinstance(label, list):
                    # 检查是否是嵌套列表（多个标签）
                    if label and isinstance(label[0], list):
                        # 多个标签：每行一个标签
                        for label_item in label:
                            line = " ".join(str(item) for item in label_item)
                            f.write(line + "\n")
                    else:
                        # 单个标签：一行
                        line = " ".join(str(item) for item in label)
                        f.write(line + "\n")
                else:
                    # 其他类型（字符串、数字等）
                    f.write(str(label))

            # 3. 每次保存 file_num 到 num.txt，并换行
            with open("num.txt", 'a') as f:  # 'a' 表示追加模式
                f.write(str(file_num) + "\n")  # 添加换行符

            time.sleep(0.05)
            folder_index += 1
        print("Data collection completed!")

        #销毁车辆和雷达传感器
        if lidar is not None:
            lidar.stop()  # 确保停止传感器线程
            lidar.destroy()  # 销毁雷达传感器

        for vehicle in vehicles:
            vehicle.destroy()

        # 销毁所有行人
        for pedestrian in pedestrians:
            pedestrian.destroy()
        # 清空队列
        while not sensor_queue.empty():
            sensor_queue.get()
        # 删除队列引用
        del sensor_queue
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')