"""
    在Carla中收集训练再识别网络的数据集:
        获取不同的蓝图的车辆连续的图片，每种类型的车辆保存为一个文件夹,
        文件夹是从序号1开始命名，每个文件夹包含数量图片，为.jpeg格式；
        在一个随机的生成点生成车辆，将相机附着在车辆的合适位置，不停的
        保存图片，然后销毁车辆，再重新生成一个不同蓝图的车辆，
        重复使用相机保存图片...

        将相机挂在路边来收集数据
"""
import carla
import random
import os
import time
import numpy as np
import cv2
import scipy.io
from queue import Queue
from queue import Empty
# 定义车辆类别的标签编号（在CARLA中，车辆的类别 ID 通常为10）
CAR_CLASS_ID = 14
TRUCK_CLASS_ID = 15
BUS_CLASS_ID = 16
# 设置参数
DROP_BUFFER_TIME = 60   # 车辆落地前的缓冲时间，防止车辆还没落地就开始保存图片
IMAGES_SAVE_TIME = 20   # 保存图片的数量
OUTPUT_DIR = "./reid_data"  # 数据保存路径
# 相机位置
camera_location = [
                carla.Transform(carla.Location(x=-111, y=-2, z=2.800176), carla.Rotation(yaw=-90)),
                carla.Transform(carla.Location(x=-106, y=-25, z=2.800176), carla.Rotation(yaw=90))
                ]

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# 图像保存函数
def save_image(image, folder_path):
    # 将CARLA图像转换为OpenCV格式
    image_array = np.array(image.raw_data)
    image_array = image_array.reshape((image.height, image.width, 4))
    image_array = image_array[:, :, :3]  # 只取RGB，不包含Alpha通道
    # 获取当前帧编号
    current_frame = image.frame
    # 保存为JPEG格式
    img_path = os.path.join(folder_path, f"{current_frame}.jpeg")
    cv2.imwrite(img_path, image_array)
    time.sleep(0.1)  # 增加延迟，减缓保存速度


# 创建相机传感器
def create_camera_sensor(world, transform):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    # 设置语义分割相机
    segmentation_camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    # 配置相机传感器
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', '90')
    segmentation_camera_bp.set_attribute('image_size_x', '1920')
    segmentation_camera_bp.set_attribute('image_size_y', '1080')
    segmentation_camera_bp.set_attribute('fov', '90')

    camera = world.spawn_actor(camera_bp, transform)
    segmentation_camera = world.spawn_actor(segmentation_camera_bp, transform)
    # 等待传感器初始化
    time.sleep(0.05)
    return camera, segmentation_camera


def sensor_callback(sensor_data, sensor_queue, sensor_name):
    sensor_queue.put((sensor_data, sensor_name))


# 生成车辆并拍摄图像
def generate_vehicle_images(vehicle_type, folder_name, world, spawn_points, tm):
    # 创建文件夹
    folder_path = os.path.join(OUTPUT_DIR, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    spawn_point = spawn_points[13]
    spectator = world.get_spectator()
    cameras = []
    segmentation_cameras = []
    # 生成车辆
    vehicle = world.try_spawn_actor(vehicle_type, spawn_point)
    x = spawn_point.location.x
    y = spawn_point.location.y
    location = carla.Location(x=x, y=y, z=20)
    spectator_transform = carla.Transform(location, carla.Rotation(pitch=-90))
    spectator.set_transform(spectator_transform)
    world.tick()
    # 配置自动驾驶
    vehicle.set_autopilot(True)  # 启动自动驾驶模式
    # 不考虑交通灯
    tm.ignore_lights_percentage(vehicle, 100)
    sensor_queue = Queue()
    # 等待车辆落地开始行驶后再开始收集数据集
    for _ in range(DROP_BUFFER_TIME):
        world.tick()
        time.sleep(0.05)  # 稍微延迟，让车辆有时间落地
    label_dicts = [{},{}]  # 每个视角的标签列表
    pic_index = 0
    for cam_loc in camera_location:
        # 创建相机传感器
        camera, segmentation_camera = create_camera_sensor(world, cam_loc)
        cameras.append(camera)
        segmentation_cameras.append(segmentation_camera)
        pic_index += 1
        camera.listen(lambda image, pic_ind=pic_index: sensor_callback(image, sensor_queue, f"camera{pic_ind}"))
        segmentation_camera .listen(lambda data, pic_ind=pic_index: sensor_callback(data, sensor_queue, f"segmentation{pic_ind}"))
    # 同步收集相机数据
    for _ in range(IMAGES_SAVE_TIME):
        world.tick()
        for _ in range(len(cameras) + len(segmentation_cameras)):
            data, sensor_name = sensor_queue.get(True, 1.0)
            # 根据传感器名称来判断如何保存数据
            if "camera" in sensor_name:  # RGB相机数据
                # 生成相机文件夹路径（根据相机编号）
                camera_index = int(sensor_name.replace("camera", ""))
                camera_folder_path = os.path.join(folder_path, f"camera{camera_index}")
                if not os.path.exists(camera_folder_path):
                    os.makedirs(camera_folder_path)
                # 保存图像
                save_image(data, camera_folder_path)

            elif "segmentation" in sensor_name:  # 语义分割相机数据
                # 根据传感器编号来获取对应的字典
                segmentation_index = int(sensor_name.replace("segmentation", ""))
                label_dict = label_dicts[segmentation_index - 1]
                # 保存标签
                save_label(data, label_dict)
        time.sleep(0.05)

    return vehicle, cameras, segmentation_cameras, sensor_queue, label_dicts, folder_path


def save_label(image, label_dict):
    """
     处理语义分割相机的输出，提取车辆像素并拟合二维框。
     """
    # 确保 label_dict 是一个字典
    if label_dict is None:
        label_dict = {}  # 初始化为空字典
    # 将CARLA图像转换为NumPy数组
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # 语义分割图像为BGRA格式
    segmentation_mask = array[:, :, 2]  # 提取蓝色通道作为分割掩码

    # unique_classes = np.unique(segmentation_mask)  # 获取所有唯一的类别 ID
    # print(f"Unique classes in segmentation mask: {unique_classes}")
    # 获取车辆像素的坐标
    vehicle_pixels = np.column_stack(np.where((segmentation_mask == CAR_CLASS_ID) |
                                              (segmentation_mask == TRUCK_CLASS_ID) |
                                              (segmentation_mask == BUS_CLASS_ID)))
    # 如果没有检测到车辆像素，则返回None
    if vehicle_pixels.size == 0:
        return None
        # 获取当前帧编号
    current_frame = image.frame
    # 获取二维边界框 (x_min, y_min, x_max, y_max)
    x_min = vehicle_pixels[:, 1].min()
    x_max = vehicle_pixels[:, 1].max()
    y_min = vehicle_pixels[:, 0].min()
    y_max = vehicle_pixels[:, 0].max()

    box_width = x_max-x_min
    box_height = y_max-y_min
    # 将标签存储到字典中
    label_dict[current_frame] = [x_min, y_min, box_width, box_height]


def save_label_to_mat(label_dicts, folder_path):
    """
    保存多个视角的 groundtruth 数据为 MATLAB 格式的 .mat 文件。
    :param label_dicts: 包含多个视角 groundtruth 数据的列表，每个元素是一个字典。
    :param folder_path: 保存 .mat 文件的文件夹路径。
    """
    # 遍历每个视角的字典
    for idx, label_dict in enumerate(label_dicts):
        # 构造 MATLAB 格式的表格
        label_data = {
            "Time": [],
            "Label": [],  # car 标签
        }
        # 遍历字典中的键值对
        for current_time, labels in label_dict.items():
            label_data["Time"].append(current_time)  # 添加时间戳
            label_data["Label"].append(labels)  # 添加标签列表
        # 保存为 .mat 文件
        file_name = os.path.join(folder_path, f"{idx+1}.mat")
        scipy.io.savemat(file_name, {"LabelData": label_data})


def destroy_vehicle_sensor(vehicle, cameras, segmentation_cameras, sensor_queue):
    vehicle.destroy()
    for cam in cameras:
        if cam is not None:
            cam.stop()  # 停止相机传感器
            cam.destroy()  # 销毁相机传感器
    for seg_cam in segmentation_cameras:
        if seg_cam is not None:
            seg_cam.stop()  # 停止相机传感器
            seg_cam.destroy()  # 销毁相机传感器
    # 销毁队列
    # 清空队列
    while not sensor_queue.empty():
        sensor_queue.get()
    # 删除队列引用
    del sensor_queue


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
                                   'mini' not in bp.id and
                                   'vehicle.carlamotors.firetruck' not in bp.id and
                                   'vehicle.carlamotors.european_hgv' not in bp.id and
                                   'vehicle.mitsubishi.fusorosa' not in bp.id]
    return filtered_vehicle_blueprints


# 主函数
def main():
    # 连接到CARLA模拟器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    try:
        # 仿真设置
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)
        print("Connected to Carla server!")

        # 创建交通管理器
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        blueprint_library = world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        filter_bike_blueprinter = filter_vehicle_blueprinter(vehicle_blueprints)
        spawn_points = world.get_map().get_spawn_points()

        # 为每个蓝图创建数据集文件夹
        for folder_index in range(len(filter_bike_blueprinter)):
            vehicle_print = filter_bike_blueprinter[folder_index]
            folder_name = f"{folder_index + 1}"
            print(f"Generating images for: {folder_name} ({vehicle_print})")
            vehicle, cameras, segmentation_cameras, sensor_queue, label_dicts, folder_path = generate_vehicle_images(vehicle_print, folder_name, world, spawn_points, tm)
            # 将ground_truth保存为mat文件
            save_label_to_mat(label_dicts, folder_path)
            destroy_vehicle_sensor(vehicle, cameras, segmentation_cameras, sensor_queue)

        print("Data collection completed!")
    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    main()
