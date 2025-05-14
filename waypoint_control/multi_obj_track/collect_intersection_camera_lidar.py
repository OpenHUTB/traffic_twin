"""
     模拟有ego_vehicle的情况：
     1.ego_vehicle没有实体，但是ego_vehicle_transform 与雷达的一致，仅z坐标不同
     2.修改camera_loc 和 lidar_loc 设置在路口中间，先试验理想状态下的轨迹跟踪
     3.保存数据格式为 panda_set 的格式
         camera_folder
         .mat
         .gpsData.mat 自车经纬等信息（可以省去，需要修改helperGenerateEgoTrajectory.m）

     ego_vehicle coordinate  x：前，y：左侧 z:向上
     4.多目标融合检测的实际范围是在45m以内

"""
import time
import numpy as np
import carla
import os
import cv2
import random
import scipy.io
import argparse
from queue import Queue
from queue import Empty
from scipy.spatial.transform import Rotation as R
from config import IntersectionConfig, town_configurations
DATA_MUN = 500
DROP_BUFFER_TIME = 50   # 车辆落地前的缓冲时间，防止车辆还没落地就开始保存图片
FUSION_DETECTION_ACTUAL_DIS = 25  # 多目标跟踪的实际检测距离
WAITE_NEXT_INTERSECTION_TIME = 300  # 等待一定时间后第二路口相机雷达开始记录数据
# 定义全局变量
global_time = 0.0

relativePose_to_egoVehicle = {
       "back_camera": [-7.00, 0.00, 2.62, -180.00, 0.00, 0.00],    # 1
       "front_camera": [7.00, 0.00, 2.62, 0.00, 0.00, 0.00],       # 2
       "right_camera": [0.00, -4.00, 2.62, -90.00, 0.00, 0.00],    # 6
       "front_right_camera": [7.00, -4.00, 2.62, -90.00, 0.00, 0.00],   # 4
       "left_camera": [0.00, 4.00, 2.62, 90.00, 0.00, 0.00],            # 5
       "front_left_camera": [7.00, 4.00, 2.62, 90.00, 0.00, 0.00]   # 3
}
relativePose_lidar_to_egoVehicle = [0, 0, 0.82, 0, 0, 0, 0, 0, 0]

# 相机名称列表
camera_names = [
    'back_camera', 'front_camera', 'front_left_camera',
    'front_right_camera', 'left_camera', 'right_camera'
]


def create_town_folder(town):
    folder_name = f"{town}"
    # 检查文件夹是否已存在，若不存在则创建
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    return folder_name


# 创建保存雷达数据的文件夹
def create_radar_folder(junc, town_folder):
    folder_name = f"{town_folder}/{junc}"
    # 检查文件夹是否已存在，若不存在则创建
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    return folder_name


# 创建保存相机数据的文件夹
def create_camera_folder(camera_id, junc, town_folder):
    folder_name = f"{town_folder}/{junc}/camera/{camera_id}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    return folder_name


def rename_intersection(input_string):
    # 检查输入字符串是否以 'road_intersection_' 开头
    if input_string.startswith('road_intersection_'):
        # 提取数字部分
        num_str = input_string.split('_')[-1]  # 分割字符串并取最后一个部分（假设数字总是在最后）
        # 构建新的字符串
        new_string = f'test_data_junc{num_str}'
        return new_string
    else:
        # 如果输入字符串不符合预期格式，可以返回原字符串或抛出异常
        return input_string  # 这里简单返回原字符串，但实际应用中可能需要更复杂的错误处理


# 保存车辆标签
def save_point_label(world, location, lidar_to_world_inv, time_stamp, all_vehicle_labels):
    # 获取雷达检测范围内的全部车辆
    # 获取附近的所有车辆
    vehicle_list = world.get_actors().filter("*vehicle*")
    pedestrian_list = world.get_actors().filter("*walker*")

    # 筛选出距离雷达小于 45 米的车辆
    # def dist(v):
    #     return v.get_location().distance(location)
    def dist(actor):
        return actor.get_location().distance(location)
    # 筛选出距离小于 LIDAR_RANGE 的车辆
    # vehicle_list = list(filter(lambda v: dist(v) < FUSION_DETECTION_ACTUAL_DIS, vehicle_list))
    vehicle_list = list(filter(lambda actor: dist(actor) < FUSION_DETECTION_ACTUAL_DIS, vehicle_list))
    pedestrian_list = list(filter(lambda actor: dist(actor) < FUSION_DETECTION_ACTUAL_DIS, pedestrian_list))
    vehicle_labels = []  # 车辆标签列表
    pedestrian_labels = []  # 车辆标签列表
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
            bounding_box_location_lidar[2] + 0.3,
            length,
            width,
            height
            # pitch_lidar,  # pitch
            # roll_lidar,  # roll
            # yaw_lidar  # yaw
        ]
        vehicle_id = vehicle.id
        vehicle_labels.append((time_stamp, vehicle_id, label))
    all_vehicle_labels.append(vehicle_labels)

    # 获取行人标签
    step = 2  # 每隔1个元素遍历（步长为2）
    for pedestrian in pedestrian_list[::step]:
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
        euler_angles_lidar = rotation_lidar.as_euler('zyx', degrees=True)
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
            # yaw_lidar  # yaw
        ]
        pedestrian_id = pedestrian.id
        pedestrian_labels.append((time_stamp, pedestrian_id, label))


# 定义函数来保存雷达点云数据
def save_radar_data(radar_data, world, ego_vehicle_transform, actual_vehicle_num, lidar_to_world_inv, all_vehicle_labels, junc, town_folder):
    global global_time
    # 获取当前帧编号
    current_frame = radar_data.frame
    # 时间戳
    # timestamp = world.get_snapshot().timestamp.elapsed_seconds
    timestamp = global_time
    global_time = timestamp + 0.05
    location = ego_vehicle_transform.location
    save_point_label(world, location, lidar_to_world_inv, timestamp, all_vehicle_labels)

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

    # 创建存储数据的文件夹
    radar_folder = create_radar_folder(junc, town_folder)
    file_name = os.path.join(radar_folder, f"{current_frame}.mat")
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

    # 创建CameraData结构体
    camera_data = []
    for i, name in enumerate(camera_names):
        camera_data.append({
            'ImagePath': f"camera/{name}/{current_frame}.jpg",  # 字符串路径
            'Pose': {
                'Position': relativePose_to_egoVehicle[name][:3],  # 单独的struct
                'Velocity': [0, 0, 0],  # 静止速度
                'Orientation': relativePose_to_egoVehicle[name][3:]  # 姿态
            },
            'Timestamp': timestamp,  # 时间戳
            'Detections': []  # 假设是检测框数据
        })

    # 构造 MATLAB 的结构体数组
    # 逐字段提取，确保 MATLAB 能正确识别为 struct array
    CameraData = np.zeros(len(camera_data), dtype=[
        ('ImagePath', 'O'),
        ('Pose', 'O'),
        ('Timestamp', 'float64'),
        ('Detections', 'O')
    ])

    for i, entry in enumerate(camera_data):
        CameraData[i] = (
            entry['ImagePath'],  # 字符串路径
            entry['Pose'],  # Pose 字典会被转换为 MATLAB 的 struct
            entry['Timestamp'],  # 时间戳
            entry['Detections']  # 5x4 矩阵
        )
    datalog = {
        'LidarData': LidarData,
        'CameraData': CameraData  # 使用结构体数组
    }
    vehicle_list = []
    # 保存每一帧融合检测实际范围内的车辆数量
    vehicle_list = world.get_actors().filter("*vehicle*")

    def dist(v):
        return v.get_location().distance(ego_vehicle_transform.location)

    vehicle_list = [v for v in vehicle_list if dist(v) < FUSION_DETECTION_ACTUAL_DIS]
    vehicle_count = len(vehicle_list)
    actual_vehicle_num.append((timestamp, vehicle_count))
    # 将点云数据保存为 .mat 文件
    # 使用 scipy.io.savemat 保存数据，MATLAB 可以读取的格式
    scipy.io.savemat(file_name, {'datalog': datalog})


# 定义函数来保存相机图像数据
def save_camera_data(image_data, camera_id, junc, town_folder):
    current_frame = image_data.frame
    image = np.array(image_data.raw_data)
    image = image.reshape((image_data.height, image_data.width, 4))  # 4th channel is alpha
    image = image[:, :, :3]  # 去掉 alpha 通道，只保留 RGB
    camera_folder = create_camera_folder(camera_id, junc, town_folder)
    file_name = os.path.join(camera_folder, f"{current_frame}.jpg")
    try:
        cv2.imwrite(file_name, image)  # 使用 OpenCV 保存图像
    except Exception as e:
        print(f"Error saving image for frame {current_frame}: {e}")
        return None
    return image


def sensor_callback(sensor_data, sensor_queue, sensor_name):
    sensor_queue.put((sensor_data, sensor_name))


# 记录雷达和相机数据
def setup_sensors(world, addtion_param, sensor_queue, transform, camera_loc):
    lidar = None
    camera_dict = {}
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
    # world.tick()
    # lidar.listen(lambda data: save_radar_data(data, world))
    lidar.listen(lambda data: sensor_callback(data, sensor_queue, "lidar"))

    # 配置相机传感器
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', '90')
    for cam_id, transform in camera_loc.items():
        camera = world.spawn_actor(camera_bp, transform)
        # camera.listen(lambda data, camera_id=cam_id: save_camera_data(data, camera_id))
        camera.listen(lambda data, camera_id=cam_id: sensor_callback(data, sensor_queue, camera_id))
        camera_dict[cam_id] = camera

    return lidar, camera_dict


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


# 生成自动驾驶车辆
def spawn_autonomous_vehicles(world, tm, num_vehicles=70, random_seed=42):
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    tm.set_random_device_seed(random_seed)
    vehicle_list = []
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    filter_vehicle_blueprints = filter_vehicle_blueprinter(vehicle_blueprints)
    # 随机选择一个位置
    spawn_points = world.get_map().get_spawn_points()
    if len(spawn_points) == 0:
        print("No spawn points available!")
        return []

    # 如果蓝图不足，使用颜色来区分
    num_blueprints = len(filter_vehicle_blueprints)
    num_colors = 12
    available_colors = ["255,0,0", "0,255,0", "0,0,255", "255,255,0", "0,255,255", "255,0,255", "128,128,0",
                        "128,0,128", "0,128,128", "255,165,0", "0,255,255", "255,192,203"]
    # 生成车辆
    vehicle_index = 0
    for _ in range(num_vehicles):
        # 选择一个随机位置生成车辆
        transform = spawn_points[np.random.randint(len(spawn_points))]
        # vehicle_bp = random.choice(filter_vehicle_blueprints)
        # 选择蓝图，确保每个蓝图的车辆唯一
        if vehicle_index < num_blueprints:
            vehicle_bp = filter_vehicle_blueprints[vehicle_index]
            vehicle_index += 1
        else:
            # 蓝图用完后，开始使用颜色来区分
            vehicle_bp = filter_vehicle_blueprints[vehicle_index % num_blueprints]
            color = available_colors[vehicle_index % num_colors]
            vehicle_bp.set_attribute('color', color)
            vehicle_index += 1

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
def spawn_autonomous_pedestrians(world, num_pedestrians=100, random_seed=42):
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

        # # 绑定控制器
        # controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        # controller = world.spawn_actor(controller_bp, carla.Transform(), pedestrian)
        # if controller:
        #     controller.start()
        #     controller.go_to_location(world.get_random_location_from_navigation())
        #     pedestrian_list.append((pedestrian, controller))
        # else:
        #     pedestrian.destroy()

        controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        controller = world.spawn_actor(controller_bp, carla.Transform(), pedestrian)
        controller.start()  # 启用自动行走
        controller.go_to_location(world.get_random_location_from_navigation())  # 设置目标点

        # 只将行人添加到列表，控制器不保存
        pedestrian_list.append(pedestrian)

        print(f"Spawned pedestrian: {pedestrian.id}")

    return pedestrian_list


def destroy_actor(lidar, camera_dict, vehicles, sensor_queue):
    if lidar is not None:
        lidar.stop()  # 确保停止传感器线程
        lidar.destroy()  # 销毁雷达传感器

    # 同样处理相机传感器
    for camera_traffic_id, camera in camera_dict.items():
        if camera is not None:
            camera.stop()  # 停止相机传感器
            camera.destroy()  # 销毁相机传感器

    for vehicle in vehicles:
        vehicle.destroy()
    # 清空队列
    while not sensor_queue.empty():
        sensor_queue.get()
    # 删除队列引用
    del sensor_queue


# 主函数
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
        '-n', '--number-of-vehicles',
        metavar='N0',
        default=50,
        type=int,
        help='Number of vehicles (default: 50)')
    argparser.add_argument(
        '-w', '--wait',
        action='store_true',
        default=False,
        help='Whether to wait vehicle reach(default: False)')
    argparser.add_argument(
        '-t', '--town',
        metavar='TOWN',
        default='Town10HD_Opt',
        choices=town_configurations.keys(),  # 限制用户只能输入已定义的城镇名
        help='Name of the town to use (e.g., Town01, Town10HD_Opt)'
    )
    argparser.add_argument(
        '-i', '--intersection',
        metavar='INTERSECTION',
        default='road_intersection_1',  # 默认路口
        help='Name of the intersection within the town (default: road_intersection_1)'
    )
    args = argparser.parse_args()

    # 连接到Carla服务器
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    # 重新加载地图，重置仿真时间
    world = client.load_world(args.town, True)
    # 仿真设置
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05
    settings.synchronous_mode = True
    world.apply_settings(settings)
    # 天气参数
    new_weather = carla.WeatherParameters(
        cloudiness=20.000000,
        precipitation=0.000000,
        precipitation_deposits=0.000000,
        wind_intensity=10.000000,
        sun_azimuth_angle=300.000000,
        sun_altitude_angle=45.000000,
        fog_density=2.000000,
        fog_distance=0.750000,
        fog_falloff=0.100000,
        wetness=0.000000,
        scattering_intensity=1.000000,
        mie_scattering_scale=0.030000,
        rayleigh_scattering_scale=0.033100,
        dust_storm=0.000000)
    world.set_weather(new_weather)
    print("Connected to Carla server!")

    # 创建交通管理器
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    addtion_param = {
        'channels': '64',
        'range': '200',
        'points_per_second': '2200000',
        'rotation_frequency': '20'
    }
    try:
        # 设置随机种子
        random_seed = 20
        intersection_config = town_configurations[args.town][args.intersection]
        ego_transform = intersection_config.ego_vehicle_position
        camera_loc = intersection_config.camera_positions
        # 先生成自动驾驶车辆
        vehicles = spawn_autonomous_vehicles(world, tm, num_vehicles=args.number_of_vehicles, random_seed=random_seed)
        # 生成随机运动行人
        pedestrians = spawn_autonomous_pedestrians(world, num_pedestrians=100, random_seed=20)

        lidar_transform = carla.Transform(
            carla.Location(x=ego_transform.location.x, y=ego_transform.location.y, z=ego_transform.location.z + 0.82),
            ego_transform.rotation)
        # 获取雷达到世界的变换矩阵（4x4矩阵）
        lidar_to_world = np.array(lidar_transform.get_matrix())
        lidar_to_world_inv = np.linalg.inv(lidar_to_world)

        # 对于两个路口的测试，第二个路口需要等待车辆到达后开始记录数据
        # if args.wait:
        #     # 记录第二路口数据时，等待车辆到达后开始记录
        #     for _ in range(WAITE_NEXT_INTERSECTION_TIME):
        #         world.tick()
        #         time.sleep(0.05)
        town_folder = create_town_folder(args.town)
        junc = rename_intersection(args.intersection)
        # 等待车辆落地开始行驶后再开始收集数据集
        for _ in range(DROP_BUFFER_TIME):
            world.tick()
            time.sleep(0.05)
        sensor_queue = Queue()
        # 启动相机、雷达传感器
        lidar, camera_dict = setup_sensors(world, addtion_param, sensor_queue, lidar_transform, camera_loc)
        actual_vehicle_num = []
        all_vehicle_labels = []
        vehicles_traj = {}
        for _ in range(DATA_MUN):
            world.tick()
            actor_list = world.get_actors().filter('vehicle.*')
            for actor in actor_list:
                vehicle_id = actor.id
                location = actor.get_location()
                x = location.x,
                y = location.y,
                z = location.z,
                # 如果该车辆ID不存在于字典中，则初始化一个空列表
                if vehicle_id not in vehicles_traj:
                    vehicles_traj[vehicle_id] = [[x, y, z]]
                else:
                    vehicles_traj[vehicle_id].append([x, y, z])
            # 同步保存多传感器数据
            for _ in range(1 + len(camera_dict)):
                data, sensor_name = sensor_queue.get(True, 1.0)
                if "lidar" in sensor_name:  # lidar数据
                    save_radar_data(data, world, ego_transform, actual_vehicle_num, lidar_to_world_inv, all_vehicle_labels, junc, town_folder)
                else:
                    save_camera_data(data, sensor_name, junc, town_folder)
            # time.sleep(0.05)
        folder_name = f"{town_folder}/{junc}/vehicle_data"
        # 检查文件夹是否已存在，若不存在则创建
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created folder: {folder_name}")
        file_path = os.path.join(folder_name, "vehicle_count.mat")
        # 将时间戳和车辆数量追加保存到txt文件中
        vehicle_data = np.array(actual_vehicle_num)
        # 保存数据为 mat 文件
        scipy.io.savemat(file_path, {"vehicle_data": vehicle_data})

        flattened_data = [item for sublist in all_vehicle_labels for item in sublist]
        processed_data = []

        for entry in flattened_data:
            timestamp, vehicle_id, position_with_dims = entry
            x, y, z, length, width, height = position_with_dims
            position = (x, y, z)
            box = (length, width, height)
            processed_data.append({
                'Time': timestamp,
                'TruthID': vehicle_id,
                'Position': position,
                'Box': box
            })

        truths = np.array(processed_data, dtype=object)
        file_path = os.path.join(folder_name, "truths.mat")
        scipy.io.savemat(file_path, {'truths': truths})

        # 保存全部车辆ground_truth
        ground_truth_file_path = os.path.join(town_folder, "ground_truth.mat")
        # 转换为MATLAB兼容格式
        # 转换为目标结构
        mat_data = []
        for vehicle_id, trajectory in vehicles_traj.items():
            # 创建结构化数组
            vehicle_struct = np.zeros((1,), dtype=[
                ('vehicleID', np.uint32),
                ('wrl_pos', 'O')  # 'O'表示Python对象
            ])

            # 填充数据 - 关键修正点
            vehicle_struct[0]['vehicleID'] = np.uint32(vehicle_id)
            # 确保轨迹是二维数组
            trajectory_array = np.array(trajectory, dtype=np.float64)
            if trajectory_array.ndim == 1:
                trajectory_array = trajectory_array.reshape(-1, 3)
            vehicle_struct[0]['wrl_pos'] = trajectory_array

            mat_data.append(vehicle_struct)

        # 转换为MATLAB兼容的cell数组
        # 关键修正：使用np.empty而不是np.array
        cell_array = np.empty((1, len(mat_data)), dtype=object)
        for i, item in enumerate(mat_data):
            cell_array[0, i] = item

        # 保存为MAT文件
        scipy.io.savemat(ground_truth_file_path,
                         {'vehicle_cells': cell_array},
                         format='5',
                         do_compression=True,
                         long_field_names=True)  # 确保MATLAB兼容性

        destroy_actor(lidar, camera_dict, vehicles, sensor_queue)
    except Exception as e:
        print(f"Error occurred during execution: {e}")
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')