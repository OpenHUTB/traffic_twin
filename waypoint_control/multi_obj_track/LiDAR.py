import carla
import random
import numpy as np

# 1. 连接到Carla服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# # 2. 获取蓝图库并创建车辆（可选）
blueprint_library = world.get_blueprint_library()
# vehicle_bp = blueprint_library.find('vehicle.audi.a2')
# spawn_point = world.get_map().get_spawn_points()[0]
# vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 3. 创建LiDAR传感器
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

# 4. 配置LiDAR参数
lidar_bp.set_attribute('channels', '32')  # 通道数
lidar_bp.set_attribute('range', '100')  # 探测范围(米)
lidar_bp.set_attribute('points_per_second', '500000')  # 每秒点数
lidar_bp.set_attribute('rotation_frequency', '10')  # 旋转频率(Hz)
lidar_bp.set_attribute('upper_fov', '10')  # 上视场角
lidar_bp.set_attribute('lower_fov', '-30')  # 下视场角

# 5. 设置LiDAR位置和旋转
# 相对于父对象的位置(x,y,z)和旋转(pitch,yaw,roll)
lidar_transform = carla.Transform(
    carla.Location(x=0.0, y=0.0, z=2.5),  # 位置偏移
    carla.Rotation(pitch=0, yaw=0, roll=0)  # 旋转角度
)

# 6. 附加到车辆（或使用world.spawn_actor直接放置在世界中）
lidar = world.spawn_actor(
    lidar_bp,
    lidar_transform,
    attach_to = None  # 如果不附加到车辆，设置为None
)


# 7. 设置数据回调函数
def lidar_callback(point_cloud):
    # 将原始数据转换为numpy数组
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # 提取XYZ坐标（忽略强度值）
    points = data[:, :-1]
    print(f"Received {len(points)} LiDAR points")


lidar.listen(lidar_callback)

# 运行模拟...
# 完成后记得销毁传感器
# lidar.destroy()