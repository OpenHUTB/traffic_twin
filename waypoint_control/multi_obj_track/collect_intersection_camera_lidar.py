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
import json
import subprocess
import pickle
import struct
from ultralytics import YOLO
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
base_frame = None
# openpcdet进行目标识别所用时间
extra_time = 0

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
def save_point_label(world, location, lidar_to_world_inv, time_stamp, all_vehicle_labels, all_pedestrian_labels):
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
    pedestrian_labels = []  # 行人标签列表
    car_labels = []  # Car 标签列表
    truck_labels = []  # Truck 标签列表
    pedestrian_labelspy = []  # Pedestrian 标签列表
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

        # 判断车辆的类别（Car, Truck）
        category = recognize_vehicle_class(vehicle)

        labelpy = [
            bounding_box_location_lidar[0],  # x
            bounding_box_location_lidar[1],  # y
            bounding_box_location_lidar[2] + 0.3,  # z ,需要把z替换成bounding_box.z
            length,
            width,
            height,
            yaw_lidar,  # yaw
            category
        ]
        # 根据类别保存标签
        if category == "Car":
            car_labels.append(labelpy)
        elif category == "Truck":
            truck_labels.append(labelpy)

        vehicle_id = vehicle.id
        vehicle_labels.append((time_stamp, vehicle_id, label))
    all_vehicle_labels.append(vehicle_labels)

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

        # 构造标签数据（Nx9 格式）
        labelpy = [
            bounding_box_location_lidar[0],  # x
            bounding_box_location_lidar[1],  # y
            bounding_box_location_lidar[2] + height / 2,  # z ,需要把z替换成bounding_box.z
            length,
            width,
            height,
            yaw_lidar,  # yaw
            'Pedestrian'
        ]
        pedestrian_labelspy.append(labelpy)  # 行人标签直接保存，无需分类

        pedestrian_id = pedestrian.id
        pedestrian_labels.append((time_stamp, pedestrian_id, label))
    all_pedestrian_labels.append(pedestrian_labels)

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
    if len(pedestrian_labelspy) > 0:
        for label in pedestrian_labelspy:
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


def send_v2x_message_lidar(lidar_data, sensor, pkl_file_path, junc):
    try:
        # 1. 读取 pkl 文件获取帧 ID
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, list):
            data = data[0]

        # 获取 frame_id 并转为字符串 (如果没有则默认为 "0")
        frame_id = str(data.get('frame_id', '0'))

        # 2. 获取当前时间戳 (保留4位小数即可)

        current_time = f"{time.time() - extra_time:.4f}"

        # 3. 拼接成最简单的纯文本字符串，用逗号隔开
        # 结果类似: "000001,1710660000.1234"
        text_payload = f"{frame_id},{current_time},{junc},点云数据"

        # 4. 没有任何多余操作，直接按照您要求的 utf-8 格式转换并发送！
        msg = carla.CustomV2XBytes()
        msg.set_bytes(bytearray(text_payload, 'utf-8'))
        sensor.send(msg)

        print(f"[V2X] 极简发送成功！")
        print(f"发送的内容: '{text_payload}'")
        print(f"占用字节数: {len(text_payload.encode('utf-8'))} 字节")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[发包报错]: {e}")


# 定义函数来保存雷达点云数据
def save_radar_data(radar_data, world, ego_vehicle_transform, actual_vehicle_num, actual_pedestrian_num,lidar_to_world_inv, all_vehicle_labels, all_pedestrian_labels, junc, town_folder, file_num, sensors, num):
    global global_time
    # 获取当前帧编号
    current_frame = radar_data.frame
    # 时间戳
    # timestamp = world.get_snapshot().timestamp.elapsed_seconds
    timestamp = global_time
    global_time = timestamp + 0.05
    location = ego_vehicle_transform.location
    all_labels = save_point_label(world, location, lidar_to_world_inv, timestamp, all_vehicle_labels, all_pedestrian_labels)

    # sensor = sensors["v2x_point"]
    # pkl_file_path = "/home/yons/traffic_twin/waypoint_control/multi_obj_track/OpenPCDet/output/cfgs/custom_models/pv_rcnn/default/pv_rcnn/default/eval/epoch_no_number/val/default/result.pkl"
    # send_v2x_message_lidar(radar_data, sensor, pkl_file_path, junc)
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
    pedestrian_list = []
    # 保存每一帧融合检测实际范围内的车辆和行人数量
    vehicle_list = world.get_actors().filter("*vehicle*")
    pedestrian_list = world.get_actors().filter("*walker*")

    # def dist(v):
    #     return v.get_location().distance(ego_vehicle_transform.location)
    def dist(actor):
        return actor.get_location().distance(ego_vehicle_transform.location)

    vehicle_list = [actor for actor in vehicle_list if dist(actor) < FUSION_DETECTION_ACTUAL_DIS]
    pedestrian_list = [actor for actor in pedestrian_list if dist(actor) < FUSION_DETECTION_ACTUAL_DIS]
    vehicle_count = len(vehicle_list)
    pedestrian_count = len(pedestrian_list)
    actual_vehicle_num.append((timestamp, vehicle_count))
    actual_pedestrian_num.append((timestamp, pedestrian_count))


    # 将点云数据保存为 .mat 文件
    # 使用 scipy.io.savemat 保存数据，MATLAB 可以读取的格式
    scipy.io.savemat(file_name, {'datalog': datalog})


    # 提取 points 的前四列
    locationpy = points[:, :4]
    # 将 locationpy 转换为 float64（即 double 类型）
    locationpy = locationpy.astype(np.float32)
    # 如果 timestamp 是单个值，创建重复的数组
    if np.isscalar(timestamp):
        timestamp_array = np.full((locationpy.shape[0], 1), timestamp, dtype=np.float32)
    else:
        # 如果 timestamp 已经是数组，确保形状正确
        timestamp_array = timestamp.reshape(-1, 1)

    # 水平拼接 location 前四列和 timestamp
    datalogpy = np.column_stack([locationpy, timestamp_array])

    # 1. 直接保存 datalogpy 为 .npy
    radar_folder = create_radar_folder_py(town_folder, junc)
    np.save(os.path.join(radar_folder, f"{file_num}.npy"), datalogpy)
    # 保存备份用于目标检测
    target_dir = "/home/yons/traffic_twin/waypoint_control/multi_obj_track/OpenPCDet/data/custom/points"
    clear_folder_contents(target_dir)
    np.save(os.path.join(target_dir, f"{file_num}.npy"), datalogpy)
    # 2. 保存 label 为 .txt
    label_folder = create_label_folder(town_folder, junc)
    with open(os.path.join(label_folder, f"{file_num}.txt"), 'w') as f:
        # 处理不同的数据结构
        if isinstance(all_labels, list):
            # 检查是否是嵌套列表（多个标签）
            if all_labels and isinstance(all_labels[0], list):
                # 多个标签：每行一个标签
                for label_item in all_labels:
                    line = " ".join(str(item) for item in label_item)
                    f.write(line + "\n")
            else:
                # 单个标签：一行
                line = " ".join(str(item) for item in all_labels)
                f.write(line + "\n")
        else:
            # 其他类型（字符串、数字等）
            f.write(str(all_labels))

    # 保存备份用于目标检测
    goal_dir = "/home/yons/traffic_twin/waypoint_control/multi_obj_track/OpenPCDet/data/custom/labels"
    clear_folder_contents(goal_dir)
    with open(os.path.join(goal_dir, f"{file_num}.txt"), 'w') as f:
        # 处理不同的数据结构
        if isinstance(all_labels, list):
            # 检查是否是嵌套列表（多个标签）
            if all_labels and isinstance(all_labels[0], list):
                # 多个标签：每行一个标签
                for label_item in all_labels:
                    line = " ".join(str(item) for item in label_item)
                    f.write(line + "\n")
            else:
                # 单个标签：一行
                line = " ".join(str(item) for item in all_labels)
                f.write(line + "\n")
        else:
            # 其他类型（字符串、数字等）
            f.write(str(all_labels))
    # 3. 每次保存 file_num 到 num.txt，并换行
    with open("num.txt", 'a') as f:  # 'a' 表示追加模式
        f.write(str(file_num) + "\n")  # 添加换行符
    # 保存备份用于目标检测
    dir_train = "/home/yons/traffic_twin/waypoint_control/multi_obj_track/OpenPCDet/data/custom/ImageSets/train.txt"
    dir_val = "/home/yons/traffic_twin/waypoint_control/multi_obj_track/OpenPCDet/data/custom/ImageSets/val.txt"
    with open(dir_train, 'w') as f:
        f.write(str(file_num) + "\n")  # 添加换行符
    with open(dir_val, 'w') as f:
        f.write(str(file_num) + "\n")  # 添加换行符

    # # 运行自动化目标检测脚本
    # duration = run_shell_script()
    # global extra_time
    # extra_time += duration

    sensor = sensors["v2x_point"]
    pkl_file_path = "/home/yons/traffic_twin/waypoint_control/multi_obj_track/OpenPCDet/output/cfgs/custom_models/pv_rcnn/default/pv_rcnn/default/eval/epoch_no_number/val/default/result.pkl"
    send_v2x_message_lidar(radar_data, sensor, pkl_file_path, junc)

# 更新目标检测的文件夹
def clear_folder_contents(folder_path):
    # 如果文件夹本来就不存在，直接建一个就行了
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹不存在，已新建: {folder_path}")
        return

    # 如果存在，就遍历里面的所有内容
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # 如果是普通文件或软链接，直接删除
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 等同于 os.remove
            # 如果里面还有子文件夹，用 shutil.rmtree 删掉子文件夹
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'删除 {file_path} 失败。原因: {e}')

    print(f"文件夹内容已清空: {folder_path}")

# 定义函数来保存相机图像数据
def save_camera_data(image_data, camera_id, junc, town_folder, model, sensors, num):
    global base_frame
    current_frame = image_data.frame
    # 如果是第一帧，就把它的 ID 存为基数
    if base_frame is None:
        base_frame = current_frame
        print(f"收到第一帧数据！将原始帧 ID {base_frame} 设置为基数 0。")
    # 计算重置后的当前帧
    normalized_frame = current_frame - base_frame
    frame_str = f"{normalized_frame:06d}"

    image = np.array(image_data.raw_data)
    image = image.reshape((image_data.height, image_data.width, 4))  # 4th channel is alpha
    image = image[:, :, :3]  # 去掉 alpha 通道，只保留 RGB
    # 使用yolov8检测图片
    results = model.predict(source=image, stream=True)
    for r in results:
        # 获取检测框、置信度和类别
        boxes = r.boxes
        for box in boxes:
            # 获取坐标 (x1, y1, x2, y2)
            b = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"检测到: {model.names[cls]}, 置信度: {conf:.2f}")

    if camera_id == "back_camera":
        sensor = sensors["v2x_back"]
    elif camera_id == "front_camera":
        sensor = sensors["v2x_front"]
    elif camera_id == "right_camera":
        sensor = sensors["v2x_front_left"]
    elif camera_id == "front_right_camera":
        sensor = sensors["v2x_front_right"]
    elif camera_id == "left_camera":
        sensor = sensors["v2x_left"]
    elif camera_id == "front_left_camera":
        sensor = sensors["v2x_right"]

    send_v2x_message_camera(sensor, junc, frame_str)

    camera_folder = create_camera_folder(camera_id, junc, town_folder)
    file_name = os.path.join(camera_folder, f"{current_frame}.jpg")
    try:
        cv2.imwrite(file_name, image)  # 使用 OpenCV 保存图像
    except Exception as e:
        print(f"Error saving image for frame {current_frame}: {e}")
        return None
    return image


def send_v2x_message_camera(sensor, junc, frame_id):
    try:
        # 获取当前时间戳 (保留4位小数即可)
        current_time = f"{time.time() - extra_time:.4f}"

        # 拼接成最简单的纯文本字符串，用逗号隔开
        # 结果类似: "000001,1710660000.1234"
        text_payload = f"{frame_id},{current_time},{junc},图片数据"

        # 4. 没有任何多余操作，直接按照您要求的 utf-8 格式转换并发送！
        msg = carla.CustomV2XBytes()
        msg.set_bytes(bytearray(text_payload, 'utf-8'))
        sensor.send(msg)

        print(f"[V2X] 极简发送成功！")
        print(f"发送的内容: '{text_payload}'")
        print(f"占用字节数: {len(text_payload.encode('utf-8'))} 字节")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ [发包报错]: {e}")

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
def spawn_autonomous_vehicles(world, tm, num_vehicles=30, random_seed=42):
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
def spawn_autonomous_pedestrians(world, num_pedestrians=100, random_seed=20):
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

def spawn_v2x_sensors(world, lidar_transform, z_height=2.57):
    sensors = {}  # 字典

    # 用字典直接给坐标命名，明确区分
    coordinates = {
        "v2x_back": (-7.0, 0.0),
        "v2x_front": (7.0, 0.0),
        "v2x_front_left": (7.0, 4.0),
        "v2x_front_right": (7.0, -4.0),
        "v2x_left": (0.0, 4.0),
        "v2x_right": (0.0, -4.0),
        "v2x_point": (0.5, 0.5)
    }

    # 获取传感器蓝图
    bp = world.get_blueprint_library().find('sensor.other.v2x_custom')
    # 定义通信频道
    bp.set_attribute("channel_id", "5")

    for name, (x, y)in coordinates.items():
        # 直接使用原始坐标
        location = carla.Location(x=x, y=y, z=z_height)
        transform = carla.Transform(location)
        # 生成V2X传感器
        sensor = world.spawn_actor(bp, transform)
        # 激活传感器
        sensor.listen(lambda data: do_nothing(data))
        # 将生成的传感器以名字存入字典
        sensors[name] = sensor

    return sensors

def do_nothing(data):
    pass

def spawn_v2x_receiver(world):
    location = carla.Location(x=0, y=0, z=2.62)
    transform = carla.Transform(location, carla.Rotation(yaw=0))

    # 获取传感器蓝图
    bp = world.get_blueprint_library().find('sensor.other.v2x_custom')
    # 定义通信频道
    bp.set_attribute("channel_id", "5")
    # 生成传感器
    receiver = world.spawn_actor(bp, transform)
    receiver.listen(lambda data: _on_v2x_received(data))

    return receiver

def _on_v2x_received(event):
    """
    接收端回调函数：将所有帧的数据保存在同一个文件夹下的独立 txt 中
    """
    if event.get_message_count() == 0:
        return

    for i, custom_data in enumerate(event):
        try:
            # 1. 获取底层数据
            parsed_data = custom_data.get()
            text_payload = ""

            # 2. 智能提取文本内容
            if isinstance(parsed_data, dict):
                payload = parsed_data.get("Message", {}).get("Message", {}).get("Bytes", "")
                if isinstance(payload, (bytes, bytearray)):
                    text_payload = payload.decode('utf-8')
                elif isinstance(payload, str):
                    text_payload = payload
            elif isinstance(parsed_data, (bytes, bytearray)):
                text_payload = parsed_data.decode('utf-8')
            elif isinstance(parsed_data, str):
                text_payload = parsed_data
            else:
                continue

            # 3. 解析逗号分隔的 "帧ID,发送时间"
            if ',' not in text_payload:
                continue

            frame_id_str, send_time_str, JUNC_ID, data_type = text_payload.split(',')
            frame_id = int(frame_id_str)
            send_time = float(send_time_str)

            # 4. 计算当前延迟
            receive_time = time.time() - extra_time
            latency_ms = (receive_time - send_time) * 1000

            print(f"✅ [V2X 接收] 帧 ID: {frame_id:06d} | 延迟: {latency_ms:.2f} ms")

            # 核心保存逻辑：扁平化保存，按帧号命名 txt
            # 直接在总文件夹下生成对应的 txt 文件路径
            BASE_SAVE_DIR = "./v2x_latency_logs"
            txt_file_path = os.path.join(BASE_SAVE_DIR, f"frame_{frame_id:06d}.txt")

            # 追加写入模式 'a'：如果该帧的 txt 文件不存在，会自动创建；
            # 如果已经存在（即收到了同一帧其他车辆发来的数据），则会在下一行继续写入。
            with open(txt_file_path, "a", encoding="utf-8") as f:
                log_line = (f"路口号: {JUNC_ID}, "
                            f"发送时间: {send_time:.6f}, "
                            f"接收时间: {receive_time:.6f}, "
                            f"延迟(ms): {latency_ms:.2f}, "
                            f"数据类型: {data_type}\n")
                f.write(log_line)

        except ValueError:
            pass
        except Exception as e:
            print(f"❌ [解析与保存报错]: {e}")
def destroy_actor(lidar, camera_dict, vehicles, sensor_queue, pedestrians):
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

    for pedestrian in pedestrians:
        pedestrian.destroy()
    # 清空队列
    while not sensor_queue.empty():
        sensor_queue.get()
    # 删除队列引用
    del sensor_queue


# python用法
# 创建保存雷达数据的文件夹
def create_radar_folder_py(town_folder, junc):
    folder_name = os.path.join("train_data", town_folder, junc, "points")
    # 检查文件夹是否已存在，若不存在则创建
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    return folder_name

# def create_radar_folder(junc, town_folder):
#     folder_name = f"{town_folder}/{junc}"
#     # 检查文件夹是否已存在，若不存在则创建
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
#         print(f"Created folder: {folder_name}")
#     return folder_name
# 创建保存标签数据的文件夹
def create_label_folder(town_folder, junc):
    folder_name = os.path.join("train_data", town_folder, junc, "labels")
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


def run_shell_script():
    # 定义脚本的绝对路径
    script_path = "/home/yons/object_detection.sh"

    # 定义工作目录
    work_dir = "/mnt/mydrive/traffic_twin/waypoint_control/multi_obj_track"
    print("开始执行 Shell 脚本...")
    # 记录开始时间 (高精度)
    start_time = time.time()
    try:
        # 使用 subprocess.run 执行脚本
        # cwd=work_dir 确保脚本是在 OpenPCDet 根目录下运行的
        # capture_output=True 可以截获脚本在终端打印的信息
        result = subprocess.run(
            ["bash", script_path],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=True
        )

        print("✅ 脚本执行成功！输出如下：")
        print(result.stdout)
        # 记录结束时间
        end_time = time.time()
        # 计算耗时
        duration = end_time - start_time
        return duration

    except subprocess.CalledProcessError as e:
        print("❌ 脚本执行失败！")
        print(f"错误信息：\n{e.stderr}")

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
        help='Number of vehicles (default: 30)')
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
        default='road_intersection_2',  # 默认路口
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
        'channels': '128',
        'range': '200',
        'points_per_second': '4000000',
        'rotation_frequency': '20'
    }

    # 加载yolo模型
    model = YOLO('yolov8n.pt')

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
        # 启动行人碰撞
        for pedestrian in pedestrians:
            if "walker.pedestrian." in pedestrian.type_id:
                pedestrian.set_collisions(True)
                pedestrian.set_simulate_physics(True)

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
        # 生成并启动V2X数据传输端
        sensors = spawn_v2x_sensors(world, lidar_transform, z_height=2.62)
        # 生成并启动V2X数据收集端
        receiver = spawn_v2x_receiver(world)
        # 定义保存数据的唯一总文件夹
        BASE_SAVE_DIR = "./v2x_latency_logs"
        # 在程序启动时，确保总文件夹存在（如果不存在则创建）
        os.makedirs(BASE_SAVE_DIR, exist_ok=True)
        actual_vehicle_num = []
        actual_pedestrian_num = []
        all_vehicle_labels = []
        all_pedestrian_labels = []
        vehicles_traj = {}
        pedestrians_traj = {}
        folder_index = 0
        # 设置变量
        num = 0
        for _ in range(DATA_MUN):
            world.tick()
            num += 1
            actor_list_vehicle = world.get_actors().filter('vehicle.*')
            for actor in actor_list_vehicle:
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

            actor_list_walker = world.get_actors().filter('walker.*')
            for actor in actor_list_walker:
                pedestrian_id = actor.id
                location = actor.get_location()
                x = location.x,
                y = location.y,
                z = location.z,
                # 如果该行人ID不存在于字典中，则初始化一个空列表
                if pedestrian_id not in pedestrians_traj:
                    pedestrians_traj[pedestrian_id] = [[x, y, z]]
                else:
                    pedestrians_traj[pedestrian_id].append([x, y, z])
            # 同步保存多传感器数据
            file_num = f"{folder_index:06d}"
            # 运行自动化目标检测脚本
            duration = run_shell_script()
            global extra_time
            extra_time += duration
            for _ in range(1 + len(camera_dict)):
                data, sensor_name = sensor_queue.get(True, 1.0)
                if "lidar" in sensor_name:  # lidar数据
                    save_radar_data(data, world, ego_transform, actual_vehicle_num, actual_pedestrian_num, lidar_to_world_inv, all_vehicle_labels, all_pedestrian_labels, junc, town_folder, file_num, sensors, num)
                else:
                    save_camera_data(data, sensor_name, junc, town_folder, model, sensors, num)
            # time.sleep(0.05)
            folder_index += 1


        # 保存车辆数据
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
        ground_truth_file_path = os.path.join(town_folder, "vehicle_ground_truth.mat")
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


        # 保存行人数据
        folder_name = f"{town_folder}/{junc}/pedestrian_data"
        # 检查文件夹是否已存在，若不存在则创建
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created folder: {folder_name}")
        file_path = os.path.join(folder_name, "pedestrian_count.mat")
        # 将时间戳和行人数量追加保存到txt文件中
        pedestrian_data = np.array(actual_vehicle_num)
        # 保存数据为 mat 文件
        scipy.io.savemat(file_path, {"pedestrian_data": pedestrian_data})

        flattened_data = [item for sublist in all_pedestrian_labels for item in sublist]
        processed_data = []

        for entry in flattened_data:
            timestamp, pedestrian_id, position_with_dims = entry
            x, y, z, length, width, height = position_with_dims
            position = (x, y, z)
            box = (length, width, height)
            processed_data.append({
                'Time': timestamp,
                'TruthID': pedestrian_id,
                'Position': position,
                'Box': box
            })

        truths = np.array(processed_data, dtype=object)
        file_path = os.path.join(folder_name, "truths.mat")
        scipy.io.savemat(file_path, {'truths': truths})

        # 保存全部行人ground_truth
        ground_truth_file_path = os.path.join(town_folder, "pedestrian_ground_truth.mat")
        # 转换为MATLAB兼容格式
        # 转换为目标结构
        mat_data = []
        for pedestrian_id, trajectory in pedestrians_traj.items():
            # 创建结构化数组
            pedestrian_struct = np.zeros((1,), dtype=[
                ('pedestrianID', np.uint32),
                ('wrl_pos', 'O')  # 'O'表示Python对象
            ])

            # 填充数据 - 关键修正点
            pedestrian_struct[0]['pedestrianID'] = np.uint32(pedestrian_id)
            # 确保轨迹是二维数组
            trajectory_array = np.array(trajectory, dtype=np.float64)
            if trajectory_array.ndim == 1:
                trajectory_array = trajectory_array.reshape(-1, 3)
            pedestrian_struct[0]['wrl_pos'] = trajectory_array

            mat_data.append(pedestrian_struct)

        # 转换为MATLAB兼容的cell数组
        # 关键修正：使用np.empty而不是np.array
        cell_array = np.empty((1, len(mat_data)), dtype=object)
        for i, item in enumerate(mat_data):
            cell_array[0, i] = item

        # 保存为MAT文件
        scipy.io.savemat(ground_truth_file_path,
                         {'pedestrian_cells': cell_array},
                         format='5',
                         do_compression=True,
                         long_field_names=True)  # 确保MATLAB兼容性


        destroy_actor(lidar, camera_dict, vehicles, sensor_queue, pedestrians)
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