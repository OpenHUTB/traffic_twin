"""
    在Carla中收集训练再识别网络的数据集:
        获取不同的蓝图的车辆连续的图片，每种类型的车辆保存为一个文件夹,
        文件夹是从序号1开始命名，每个文件夹包含数量图片，为.jpeg格式；
        在一个随机的生成点生成车辆，将相机附着在车辆的合适位置，不停的
        保存图片，然后销毁车辆，再重新生成一个不同蓝图的车辆，
        重复使用相机保存图片...
"""
import carla
import random
import os
import time
import numpy as np
import cv2

# 设置参数
DROP_BUFFER_TIME = 55   # 车辆落地前的缓冲时间，防止车辆还没落地就开始保存图片
IMAGES_SAVE_TIME = 25   # 保存图片的时间
OUTPUT_DIR = "./reid_data"  # 数据保存路径
# 相机相对于车辆的位置
camera_location_relative_vehicle = [
        carla.Transform(carla.Location(x=4, y=0, z=1),carla.Rotation(yaw=180)),
        carla.Transform(carla.Location(x=0, y=4, z=1), carla.Rotation(yaw=-90)),
        carla.Transform(carla.Location(x=0, y=-4, z=1), carla.Rotation(yaw=90)),
        carla.Transform(carla.Location(x=-4, y=0, z=1), carla.Rotation(yaw=0))
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
    timestamp = int(time.time() * 1000)  # 当前时间的毫秒级时间戳
    # 保存为JPEG格式
    img_path = os.path.join(folder_path, f"{timestamp}.jpeg")
    cv2.imwrite(img_path, image_array)
    time.sleep(0.1)  # 增加延迟，减缓保存速度


# 创建相机传感器
def create_camera_sensor(vehicle, world, transform):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    # 配置相机传感器
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', '90')
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    # 等待传感器初始化
    time.sleep(0.05)
    return camera


# 生成车辆并拍摄图像
def generate_vehicle_images(vehicle_type, folder_name, world, spawn_points, tm):
    # 创建文件夹
    folder_path = os.path.join(OUTPUT_DIR, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 随机生成车辆位置
    spawn_point = random.choice(spawn_points)
    spectator = world.get_spectator()
    cameras = []
    # 生成车辆
    vehicle = world.try_spawn_actor(vehicle_type, spawn_point)
    spawn_point.location.z += 20
    spectator_transform = carla.Transform(spawn_point.location, carla.Rotation(pitch=-90))
    spectator.set_transform(spectator_transform)
    world.tick()
    # 配置自动驾驶
    vehicle.set_autopilot(True)  # 启动自动驾驶模式
    # 不考虑交通灯
    tm.ignore_lights_percentage(vehicle, 100)

    # 等待车辆落地开始行驶后再开始收集数据集
    for _ in range(DROP_BUFFER_TIME):
        world.tick()
        time.sleep(0.05)  # 稍微延迟，让车辆有时间落地

    pic_index = 0
    for cam_loc in camera_location_relative_vehicle:
        # 创建相机传感器
        camera = create_camera_sensor(vehicle, world, cam_loc)
        cameras.append(camera)
        pic_index += 1
        # 创建文件夹
        view_folder_path = os.path.join(folder_path, f"{pic_index}")
        print(view_folder_path)
        if not os.path.exists(view_folder_path):
            os.makedirs(view_folder_path)
        # 获取并且保存图片
        camera.listen(lambda data, camera_path=view_folder_path: save_image(data, camera_path))

    # # 同步收集相机数据
    for _ in range(IMAGES_SAVE_TIME):
        world.tick()
        time.sleep(0.05)
    return vehicle, cameras


def destroy_vehicle_sensor(vehicle, cameras):
    vehicle.destroy()
    for cam in cameras:
        if cam is not None:
            cam.stop()  # 停止相机传感器
            cam.destroy()  # 销毁相机传感器


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
            print(folder_index)
            vehicle_print = filter_bike_blueprinter[folder_index]
            folder_name = f"{folder_index + 1}"
            print(f"Generating images for: {folder_name} ({vehicle_print})")
            vehicle, cameras = generate_vehicle_images(vehicle_print, folder_name, world, spawn_points, tm)
            destroy_vehicle_sensor(vehicle, cameras)

        print("Data collection completed!")
    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    main()
