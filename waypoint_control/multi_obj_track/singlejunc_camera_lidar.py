import time
import numpy as np
import carla
import os
import cv2
import random
import scipy.io

camera_loc = {
       1: carla.Transform(carla.Location(x=-48.539356, y=18.196909, z=5.598385), carla.Rotation(pitch=-8.674802, yaw=-90.430931, roll=0.000077)),
       2: carla.Transform(carla.Location(x=-48.539356, y=21.196909, z=5.598385), carla.Rotation(pitch=-8.674802, yaw=90.111420, roll=0.000077)),
       3: carla.Transform(carla.Location(x=-54.539356, y=13.196909, z=5.598385), carla.Rotation(pitch=-8.674802, yaw=-178.492355, roll=0.000077)),
       4: carla.Transform(carla.Location(x=-54.539356, y=25.196909, z=5.598385), carla.Rotation(pitch=-8.674802, yaw=-178.492355, roll=0.000077)),
       5: carla.Transform(carla.Location(x=-51.539356, y=13.196909, z=5.598385), carla.Rotation(pitch=-8.674802, yaw=-0.282532, roll=0.000077)),
       6: carla.Transform(carla.Location(x=-51.539356, y=25.196909, z=5.598385), carla.Rotation(pitch=-8.674802, yaw=-0.282532, roll=0.000077))
}


# 创建保存雷达数据的文件夹
def create_radar_folder(lidar_id):
    folder_name = f"data/lidar_{lidar_id}_data"
    # 检查文件夹是否已存在，若不存在则创建
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    return folder_name


# 创建保存相机数据的文件夹
def create_camera_folder(camera_id):
    folder_name = f"data/camera_{camera_id}_data"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    return folder_name


# 定义回调函数来保存雷达点云数据
def save_radar_data(radar_data, lidar_id):
    # 获取当前帧编号
    current_frame = radar_data.frame

    # 获取雷达数据并将其转化为numpy数组
    points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (len(points) // 4, 4))

    # 创建存储数据的文件夹（每个雷达一个文件夹）
    radar_folder = create_radar_folder(lidar_id)
    file_name = os.path.join(radar_folder, f"{current_frame}.mat")
    # 将点云数据保存为 .mat 文件
    # 使用 scipy.io.savemat 保存数据，MATLAB 可以读取的格式
    scipy.io.savemat(file_name, {'points': points})
    # 保存点云数据到文件
    np.save(file_name, points)
    return points  # 返回点云数据用于后续处理


# 定义回调函数来保存相机图像数据
def save_camera_data(image_data, camera_id):
    current_frame = image_data.frame
    image = np.array(image_data.raw_data)
    image = image.reshape((image_data.height, image_data.width, 4))  # 4th channel is alpha
    image = image[:, :, :3]  # 去掉 alpha 通道，只保留 RGB
    camera_folder = create_camera_folder(camera_id)
    file_name = os.path.join(camera_folder, f"{current_frame}.png")
    # cv2.imwrite(file_name, image)  # 使用 OpenCV 保存图像
    try:
        cv2.imwrite(file_name, image)  # 使用 OpenCV 保存图像
    except Exception as e:
        print(f"Error saving image for frame {current_frame}: {e}")
        return None
    return image


# 记录雷达和相机数据
def setup_sensors(world, lidar_id, addtion_param):
    lidar = None
    camera_dict = {}
    # 获取所有交通信号灯
    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    for traffic_light in traffic_lights:
        if int(traffic_light.get_opendrive_id()) == lidar_id:
            # 获取交通灯的位置信息
            traffic_transform = traffic_light.get_transform()
            boxes = traffic_light.get_light_boxes()
            transform = carla.Transform(boxes[1].location, traffic_transform.rotation)
            transform.rotation.pitch = -15
            print(transform)
            # 配置LiDAR传感器
            lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('dropoff_general_rate',
                                   lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit',
                                   lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity',
                                   lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

            for key in addtion_param:
                lidar_bp.set_attribute(key, addtion_param[key])

            # 创建雷达并绑定回调
            lidar = world.spawn_actor(lidar_bp, transform)
            world.tick()
            break
    lidar.listen(lambda data: save_radar_data(data, lidar_id))

    # 配置相机传感器
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', '90')
    for cam_id, transform in camera_loc.items():
        camera = world.spawn_actor(camera_bp, transform)
        camera.listen(lambda data: save_camera_data(data, cam_id))
        camera_dict[cam_id] = camera

    return lidar, camera_dict


# 生成自动驾驶车辆
def spawn_autonomous_vehicles(world, tm, num_vehicles=10):
    vehicle_list = []
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    vehicle_bp = random.choice(vehicle_blueprints)
    for _ in range(num_vehicles):
        # 随机选择一个位置
        spawn_point = world.get_map().get_spawn_points()
        if len(spawn_point) == 0:
            print("No spawn points available!")
            return []

        # 选择一个随机位置生成车辆
        transform = spawn_point[np.random.randint(len(spawn_point))]
        vehicle_bp = random.choice(vehicle_blueprints)
        vehicle = world.try_spawn_actor(vehicle_bp, transform)
        if vehicle is None:
            continue
        # 配置自动驾驶
        vehicle.set_autopilot(True)  # 启动自动驾驶模式
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

    radar_dict = {}
    camera_dict = {}

    addtion_param = {
        'channels': '64',
        'range': '100',
        'points_per_second': '250000',
        'rotation_frequency': '20'
    }
    try:

        # 先生成自动驾驶车辆
        vehicles = spawn_autonomous_vehicles(world, tm, num_vehicles=20)

        # 启动雷达传感器
        traffic_id = 951
        radar_dict, camera_dict = setup_sensors(world, traffic_id, addtion_param)

        while True:
            world.tick()
            time.sleep(0.1)

    except Exception as e:
        print(f"Error occurred during execution: {e}")
    finally:
        for lidar_traffic_id, lidar in radar_dict.items():
            if lidar is not None:
                lidar.stop()  # 确保停止传感器线程
                lidar.destroy()  # 销毁雷达传感器

        # 同样处理相机传感器
        for camera_traffic_id, camera in camera_dict.items():
            if camera is not None:
                camera.stop()  # 停止相机传感器
                camera.destroy()  # 销毁相机传感器
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')