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
LIDAR_RANGE = 50   # 筛选可视距离雷达的车辆
POINT_SAVE_TIME = 3000  # 保存数据数量


# 创建保存雷达数据的文件夹
def create_radar_folder():
    folder_name = f"train_data"
    # 检查文件夹是否已存在，若不存在则创建
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    return folder_name


# 创建保存标签数据的文件夹
def create_label_folder():
    folder_name = f"train_data/label"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    return folder_name


def recognize_vehicle_class(vehicle):
    blueprint = vehicle.type_id.lower()  # 获取车辆的蓝图名称并转换为小写
    # 定义需要识别为卡车的特定蓝图ID
    truck_blueprints = [
        'vehicle.carlamotors.carlacola',
        'vehicle.carlamotors.european_hgv',
        'vehicle.tesla.cybertruck',
        'vehicle.carlamotors.firetruck',
        'vehicle.mitsubishi.fusorosa'
    ]
    # 检查蓝图名称是否在卡车列表中
    if blueprint in truck_blueprints:
        return 'truck'
    else:
        return "car"


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
    # 获取雷达检测范围内的全部车辆
    # 获取附近的所有车辆
    vehicle_list = world.get_actors().filter("*vehicle*")

    # 筛选出距离雷达小于 45 米的车辆
    def dist(v):
        return v.get_location().distance(location)
    # 筛选出距离小于 LIDAR_RANGE 的车辆
    vehicle_list = list(filter(lambda v: dist(v) < LIDAR_RANGE, vehicle_list))
    # 按方向过滤车辆
    # vehicle_list = filter_vehicle_by_direction(vehicle_list, lidar_yaw, location, angle_tolerance=15, distance_threshold=30)

    car_labels = []  # car 标签列表
    truck_labels = []  # truck 标签列表
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
        euler_angles_lidar = rotation_lidar.as_euler('zyx', degrees=True)
        # 输出转换后的 pitch, yaw, roll
        yaw_lidar, pitch_lidar, roll_lidar = euler_angles_lidar
        # 构造标签数据（Nx9 格式）

        label = [
            bounding_box_location_lidar[0],  # x
            bounding_box_location_lidar[1],  # y
            bounding_box_location_lidar[2] + 0.3,  # z ,需要把z替换成bounding_box.z
            length,
            width,
            height,
            pitch_lidar,  # pitch
            roll_lidar,  # roll
            yaw_lidar  # yaw
        ]
        # 判断车辆的类别（car, truck）
        category = recognize_vehicle_class(vehicle)
        # 根据类别保存标签
        if category == "car":
            car_labels.append(label)
        elif category == "truck":
            truck_labels.append(label)
    # 将 car 和 truck 数据转换为 NumPy 数组
    car_labels = np.array(car_labels, dtype=object)
    truck_labels = np.array(truck_labels, dtype=object)
    # 构造 MATLAB 格式的表格
    label_data = {
        "Time": time_stamp,
        "car": car_labels,  # car 标签
        "truck": truck_labels  # truck 标签
    }
    # label_folder = create_label_folder()
    # file_name = os.path.join(label_folder, f"{current_frame}.mat")
    # # 保存为 .mat 文件
    # scipy.io.savemat(file_name, {"LabelData": label_data})
    return label_data


# 定义回调函数来保存雷达点云数据
def save_radar_data(radar_data, world, location, lidar_to_world_inv, lidar_yaw, sensor_queue):
    # 时间戳
    timestamp = world.get_snapshot().timestamp.elapsed_seconds
    # 获取当前帧编号
    current_frame = radar_data.frame
    # 保存车辆标签
    label_data = save_point_label(world, location, lidar_to_world_inv, timestamp, current_frame, lidar_yaw)

    # 保存点云数据
    # 获取雷达数据并将其转化为numpy数组
    points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (len(points) // 4, 4))
    location = points[:, :3]
    # 将 location 转换为 float64（即 double 类型）
    location = location.astype(np.float64)
    intensity = points[:, 3].reshape(-1, 1).astype(np.float64)  # 获取强度数据（第四通道）
    # intensity_scaled = np.round(intensity * 255).astype(np.uint8)
    count = location.shape[0]
    # 计算 x 的范围
    x_limits = [np.min(location[:, 0]), np.max(location[:, 0])]  # x 轴的最小值和最大值
    y_limits = [np.min(location[:, 1]), np.max(location[:, 1])]  # y 轴的最小值和最大值
    z_limits = [np.min(location[:, 2]), np.max(location[:, 2])]  # z 轴的最小值和最大值

    # # 创建存储数据的文件夹（每个雷达一个文件夹）
    # radar_folder = create_radar_folder()
    # file_name = os.path.join(radar_folder, f"{current_frame}.mat")
    LidarData = {
        'PointCloud': {
            'Location': location,
            'Count': count,
            'XLimits': x_limits,
            'YLimits': y_limits,
            'ZLimits': z_limits,
            'Color': [],
            'Normal': [],
            'Intensity': intensity
        },
        'Timestamp': timestamp,
        'Pose': {
            'Position': relativePose_lidar_to_egoVehicle[:3],
            'Velocity': [0, 0, 0],
            'Orientation': [0, 0, 0]
        },
        'Detections': []
    }
    datalog = {
        'LidarData': LidarData
    }
    # 将点云数据保存为 .mat 文件
    # 使用 scipy.io.savemat 保存数据，MATLAB 可以读取的格式
    # scipy.io.savemat(file_name, {'datalog': datalog})
    sensor_queue.put((datalog, label_data))


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
def spawn_autonomous_vehicles(world, tm, num_vehicles=70, random_seed=42):
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


# 主函数
def main():
    # 连接到Carla服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

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
        'channels': '64',
        'range': '200',
        'points_per_second': '2200000',
        'rotation_frequency': '20'
    }
    try:
        # 设置随机种子
        random_seed = 20
        # 静止 ego_vehicle 的位置
        ego_transform = carla.Transform(carla.Location(x=-46, y=21, z=1), carla.Rotation(pitch=0, yaw=90, roll=0))
        # 先生成自动驾驶车辆
        vehicles = spawn_autonomous_vehicles(world, tm, num_vehicles=70, random_seed=random_seed)
        # 设置理想化的雷达位置
        lidar_transform = carla.Transform(carla.Location(x=-46, y=21, z=1.8), carla.Rotation(pitch=0, yaw=90, roll=0))
        # 获取雷达到世界的变换矩阵（4x4矩阵）
        lidar_to_world = np.array(lidar_transform.get_matrix())
        lidar_to_world_inv = np.linalg.inv(lidar_to_world)
        sensor_queue = Queue()
        # 启动雷达传感器
        lidar = setup_sensors(world, addtion_param, lidar_transform, lidar_to_world_inv, sensor_queue)
        folder_index = 1
        # 同步保存雷达数据
        for _ in range(POINT_SAVE_TIME):
            world.tick()
            datalog, label = sensor_queue.get(True, 1.0)
            # 开始保存
            # 创建存储数据的文件夹（每个雷达一个文件夹）
            radar_folder = create_radar_folder()
            file_name = os.path.join(radar_folder, f"{folder_index}.mat")
            # 使用 scipy.io.savemat 保存数据，MATLAB 可以读取的格式

            scipy.io.savemat(file_name, {'datalog': datalog})
            label_folder = create_label_folder()
            file_name = os.path.join(label_folder, f"{folder_index}.mat")
            # 保存为 .mat 文件
            scipy.io.savemat(file_name, {"LabelData": label})
            time.sleep(0.05)
            folder_index += 1
        print("Data collection completed!")
        # 销毁车辆和雷达传感器
        if lidar is not None:
            lidar.stop()  # 确保停止传感器线程
            lidar.destroy()  # 销毁雷达传感器

        for vehicle in vehicles:
            vehicle.destroy()
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