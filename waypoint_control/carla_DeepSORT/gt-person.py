import json
from typing import Any
import torch
import numpy as np
import cv2
from time import perf_counter
import carla
import queue
import random

from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from pascal_voc_writer import Writer

import concurrent.futures
from functools import partial

# Part 1
# image size
image_w = 256 * 4
image_h = 256 * 3

# yolo filtering - 添加行人类别
class_id = [0, 1, 2]  # 0: person, 1: vehicle, 2: car
class_name = {0: 'person', 1: 'vehicle', 2: 'car'}

cfg = get_config()
cfg.merge_from_file('deep_sort/configs/deep_sort.yaml')
deepsort_weights = "deep_sort/deep/checkpoint/ckpt.t7"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 为行人和车辆分别创建 DeepSort 跟踪器
deepsort_person = DeepSort(
    deepsort_weights,
    max_age=70,
    n_init=3)

deepsort_vehicle = DeepSort(
    deepsort_weights,
    max_age=70,
    n_init=3)

# establising Carla connection
client = carla.Client('localhost', 2000)
world = client.get_world()

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Get the world spectator
spectator = world.get_spectator()

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

# vehicle setup
vehicle_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
# vehicle_bp.set_attribute('role_name', 'ego')
# vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# camera setip
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', f'{image_w}')
camera_bp.set_attribute('image_size_y', f'{image_h}')
camera_bp.set_attribute('fov', '110')
fov = 110

# attaching camera
# camera_init_trans = carla.Transform(carla.Location(z=2))
# 路口 1
# camera_init_trans = carla.Transform(carla.Location(x=-46, y=14, z=2.5),carla.Rotation(pitch=0, yaw=90, roll=0))
# 路口 2
# camera_init_trans = carla.Transform(carla.Location(x=104, y=14, z=2.5),carla.Rotation(pitch=0, yaw=90, roll=0))
# 路口 3
# camera_init_trans = carla.Transform(carla.Location(x=-106, y=14, z=2.5),carla.Rotation(pitch=0, yaw=90, roll=0))
# 路口 4
# camera_init_trans = carla.Transform(carla.Location(x=-46, y=-68, z=2.5),carla.Rotation(pitch=0, yaw=90, roll=0))
# 路口 5
camera_init_trans = carla.Transform(carla.Location(x=-50, y=128, z=2.5),
                                                carla.Rotation(pitch=0, yaw=-0, roll=0))

# camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
camera = world.spawn_actor(camera_bp, camera_init_trans)
image_w = camera_bp.get_attribute('image_size_x').as_int()
image_h = camera_bp.get_attribute('image_size_y').as_int()

# auto pilot for ego vehicle
# vehicle.set_autopilot(True)

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()
camera.listen(image_queue.put)


# Part 2

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]]).T
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img


# Remember the edge pairs
edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

# Get the world to camera matrix
world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

# Calculate the camera projection matrix to project from 3D -> 2D
K = build_projection_matrix(image_w, image_h, fov)
K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

# for i in range(30):
#     vehicle_bp = world.get_blueprint_library().filter('vehicle')
#
#     # Exclude bicycle
#     car_bp = [bp for bp in vehicle_bp if int(bp.get_attribute('number_of_wheels')) == 4]
#     npc = world.try_spawn_actor(random.choice(car_bp), random.choice(spawn_points))
#
#     if npc:
#         npc.set_autopilot(True)


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


# def spawn_pedestrians(world, num_pedestrians=150, random_seed=20):
#     """
#     生成随机运动的行人
#     """
#     random.seed(random_seed)
#     np.random.seed(random_seed)
#     pedestrian_list = []
#
#     # 获取普通行人蓝图（排除特殊类型）
#     walker_bps = [
#         bp for bp in world.get_blueprint_library().filter('walker.pedestrian*')
#         if not bp.id.split('.')[-1] in {'child', 'skeleton'}
#     ]
#
#     print(f"Found {len(walker_bps)} pedestrian blueprints")
#
#     spawned_count = 0
#     for _ in range(num_pedestrians):
#         # 获取安全生成位置
#         spawn_point = None
#         for _ in range(10):  # 最多尝试10次
#             location = world.get_random_location_from_navigation()
#             if location and 0 < location.z < 1.0:
#                 spawn_point = carla.Transform(location)
#                 break
#         if not spawn_point:
#             print("Failed to find valid spawn location")
#             continue
#
#         # 生成行人
#         bp = random.choice(walker_bps)
#         pedestrian = world.try_spawn_actor(bp, spawn_point)
#         if not pedestrian:
#             print(f"Failed to spawn pedestrian at {spawn_point.location}")
#             continue
#
#         # 通过Actor接口启用物理
#         try:
#             pedestrian.set_simulate_physics(True)
#             # 在同步模式下，我们会在主循环中调用world.tick()
#         except RuntimeError as e:
#             print(f"Failed to set physics: {e}")
#             pedestrian.destroy()
#             continue
#
#         # 设置AI控制器
#         try:
#             controller_bp = world.get_blueprint_library().find('controller.ai.walker')
#             if controller_bp:
#                 controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=pedestrian)
#                 controller.start()  # 启用自动行走
#
#                 # 设置随机目标点
#                 target_location = world.get_random_location_from_navigation()
#                 if target_location:
#                     controller.go_to_location(target_location)
#
#                 # 只将行人添加到列表，控制器不保存
#                 pedestrian_list.append(pedestrian)
#                 spawned_count += 1
#                 print(f"Successfully spawned pedestrian {spawned_count} with AI controller")
#             else:
#                 print("Failed to find AI controller blueprint")
#                 pedestrian.destroy()
#         except Exception as e:
#             print(f"Failed to setup AI controller: {e}")
#             pedestrian.destroy()
#
#     print(f"Spawned {spawned_count} pedestrians with AI controllers")
#     return pedestrian_list

# 生成随机运动行人
def spawn_autonomous_pedestrians(world, num_pedestrians=150):
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

    return pedestrian_list

# 添加自定义 JSON 编码器来处理 numpy 类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

tm = client.get_trafficmanager(8000)
tm.set_synchronous_mode(True)
random_seed = 20
random.seed(random_seed)
np.random.seed(random_seed)
tm.set_random_device_seed(random_seed)
vehicle_list = []
blueprint_library = world.get_blueprint_library()
vehicle_blueprints = blueprint_library.filter('vehicle.*')
filter_vehicle_blueprints = filter_vehicle_blueprinter(vehicle_blueprints)
# # 随机选择一个位置
# spawn_points = world.get_map().get_spawn_points()

# 如果蓝图不足，使用颜色来区分
num_blueprints = len(filter_vehicle_blueprints)
num_colors = 12
available_colors = ["255,0,0", "0,255,0", "0,0,255", "255,255,0", "0,255,255", "255,0,255", "128,128,0",
                    "128,0,128", "0,128,128", "255,165,0", "0,255,255", "255,192,203"]
# 生成车辆
vehicle_index = 0
for _ in range(50):
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
    # print(f"Spawned vehicle: {vehicle.id}")

# 生成行人
# pedestrian_list = spawn_pedestrians(world, 150)  # 生成50个行人
pedestrian_list = spawn_autonomous_pedestrians(world, 150)


# Retrieve all these type objects

car_objects = world.get_environment_objects(carla.CityObjectLabel.Car)
truck_objects = world.get_environment_objects(carla.CityObjectLabel.Truck)
bus_objects = world.get_environment_objects(carla.CityObjectLabel.Bus)
env_object_ids = []
for obj in (car_objects + truck_objects + bus_objects):
    env_object_ids.append(obj.id)

world.enable_environment_objects(env_object_ids, False)  # Disable all static vehicles

edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]


def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False


def get_vanishing_point(p1, p2, p3, p4):
    k1 = (p4[1] - p3[1]) / (p4[0] - p3[0])
    k2 = (p2[1] - p1[1]) / (p2[0] - p1[0])

    vp_x = (k1 * p3[0] - k2 * p1[0] + p1[1] - p3[1]) / (k1 - k2)
    vp_y = k1 * (vp_x - p3[0]) + p3[1]

    return [vp_x, vp_y]


def clear():
    """destroy all the actors
    """
    settings = world.get_settings()
    settings.synchronous_mode = False  # Disables synchronous mode
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    camera.stop()

    # destroy all npc's
    for npc in world.get_actors().filter('*vehicle*'):
        if npc:
            npc.destroy()

        # destroy all pedestrians
    for pedestrian in world.get_actors().filter('*walker*'):
        if pedestrian:
            pedestrian.destroy()

        # destroy all controllers
    for controller in world.get_actors().filter('*controller*'):
        if controller:
            controller.destroy()

    print("Vehicles and Pedestrians Destroyed.")


# vehicle.set_autopilot(True)
edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

# 轨迹保存数据结构
vehicle_trajectories = {}
person_trajectories = {}

frames_count = 0
annotations = []
sort_final = []

# 颜色映射
color_map = {
    'person': (0, 255, 0),      # 绿色
    'vehicle': (0, 0, 255),     # 红色
    'car': (255, 0, 0)          # 蓝色
}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    for tick in range(500):
        try:
            world.tick()

            # Move the spectator to the top of the vehicle
            # transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4, z=50)),
            #                             carla.Rotation(yaw=-180, pitch=-90))
            # transform = carla.Transform(carla.Location(x=-46, y=14, z=3.6),carla.Rotation(pitch=0, yaw=90, roll=0))
            spectator.set_transform(camera_init_trans)

            # Retrieve and reshape the image
            image = image_queue.get()
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            img = img[:, :, :3]  # 去除alpha通道
            img = np.ascontiguousarray(img, dtype=np.uint8)
            timestamp_sec = image.timestamp

            # Get the camera matrix
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # Get the image frame from the image queue
            frame = np.copy(img)
            frames_count += 1
            ground_truth_annotations = []
            DSort_nnotations = []

            # Perform YOLO object detection
            model = YOLO('weights/best.pt')
            preds = model(frame)

            # 分别处理行人和车辆的检测结果
            person_bbox_xyxy = []
            person_conf_score = []
            person_cls_id = []

            vehicle_bbox_xyxy = []
            vehicle_conf_score = []
            vehicle_cls_id = []

            # Iterate through the detected objects and their bounding boxes
            for box in preds:
                for r in box.boxes.data.tolist():
                    x_min, y_min, x_max, y_max, conf, class_ids = r
                    id = int(class_ids)
                    if id in class_id:
                        if id == 0:  # person
                            person_bbox_xyxy.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                            person_conf_score.append(conf)
                            person_cls_id.append(int(id))
                        else:  # vehicle
                            vehicle_bbox_xyxy.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                            vehicle_conf_score.append(conf)
                            vehicle_cls_id.append(int(id))
                    else:
                        continue

            # 分别使用不同的 DeepSort 跟踪器
            person_outputs = deepsort_person.update(person_bbox_xyxy, person_conf_score, frame)
            vehicle_outputs = deepsort_vehicle.update(vehicle_bbox_xyxy, vehicle_conf_score, frame)

            # 处理行人跟踪结果
            for output in person_outputs:
                x1, y1, x2, y2, track_id = output
                DSort_nnotations.append({
                    "height": int(y2 - y1),
                    "width": int(x2 - x1),
                    "id": "person",
                    "track_id": int(track_id),
                    "y": int(y1),
                    "x": int(x1)
                })

                # 更新行人轨迹
                if track_id not in person_trajectories:
                    person_trajectories[track_id] = []
                person_trajectories[track_id].append({
                    "frame": frames_count,
                    "timestamp": timestamp_sec,
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1)
                })

                # 在图像上绘制行人边界框和ID
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color_map['person'], 2)
                cv2.putText(img, f'Person {track_id}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map['person'], 2)

            # 处理车辆跟踪结果
            for output in vehicle_outputs:
                x1, y1, x2, y2, track_id = output
                DSort_nnotations.append({
                    "height": int(y2 - y1),
                    "width": int(x2 - x1),
                    "id": "vehicle",
                    "track_id": int(track_id),
                    "y": int(y1),
                    "x": int(x1)
                })

                # 更新车辆轨迹
                if track_id not in vehicle_trajectories:
                    vehicle_trajectories[track_id] = []
                vehicle_trajectories[track_id].append({
                    "frame": frames_count,
                    "timestamp": timestamp_sec,
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1)
                })

                # 在图像上绘制车辆边界框和ID
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color_map['vehicle'], 2)
                cv2.putText(img, f'Vehicle {track_id}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map['vehicle'], 2)

            sort_final.append({
                "timestamp": timestamp_sec,
                "num": image.frame,
                "class": "frame",
                "hypotheses": DSort_nnotations
            })

            hypo = [{
                "frames": sort_final,
                "class": "video",
                "filename": "hypo.json"
            }]

            # 处理地面真实标注 - 包括行人和车辆
            vehicles = list(world.get_actors().filter('*vehicle*'))
            walkers = list(world.get_actors().filter('*walker*'))
            all_actors = vehicles + walkers
            for actor in all_actors:
                # 确定类别
                if 'walker' in actor.type_id:
                    obj_class = 'person'
                else:
                    obj_class = 'vehicle'

                bb = actor.bounding_box
                dist = actor.get_transform().location.distance(camera.get_transform().location)

                # Filter for the objects within 50m
                if dist < 50:
                    forward_vec = camera.get_transform().get_forward_vector()
                    ray = actor.get_transform().location - camera.get_transform().location
                    if forward_vec.dot(ray) > 0:
                        verts = [v for v in bb.get_world_vertices(actor.get_transform())]
                        points_image = []

                        for vert in verts:
                            ray0 = vert - camera.get_transform().location
                            cam_forward_vec = camera.get_transform().get_forward_vector()
                            if (cam_forward_vec.dot(ray0) > 0):
                                p = get_image_point(vert, K, world_2_camera)
                            else:
                                p = get_image_point(vert, K_b, world_2_camera)

                            points_image.append(p)

                        x_min, x_max = 10000, -10000
                        y_min, y_max = 10000, -10000

                        for edge in edges:
                            p1 = points_image[edge[0]]
                            p2 = points_image[edge[1]]

                            p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                            p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                            if not p1_in_canvas and not p2_in_canvas:
                                continue

                            p1_temp, p2_temp = (p1.copy(), p2.copy())

                            if not (p1_in_canvas and p2_in_canvas):
                                p = [0, 0]
                                p_in_canvas, p_not_in_canvas = (p1, p2) if p1_in_canvas else (p2, p1)
                                k = (p_not_in_canvas[1] - p_in_canvas[1]) / (p_not_in_canvas[0] - p_in_canvas[0])

                                x = np.clip(p_not_in_canvas[0], 0, image.width)
                                y = k * (x - p_in_canvas[0]) + p_in_canvas[1]

                                if y >= image.height:
                                    p[0] = (image.height - p_in_canvas[1]) / k + p_in_canvas[0]
                                    p[1] = image.height - 1
                                elif y <= 0:
                                    p[0] = (0 - p_in_canvas[1]) / k + p_in_canvas[0]
                                    p[1] = 0
                                else:
                                    p[0] = image.width - 1 if x == image.width else 0
                                    p[1] = y

                                p1_temp, p2_temp = (p, p_in_canvas)

                            x_max = max(p1_temp[0], p2_temp[0], x_max)
                            x_min = min(p1_temp[0], p2_temp[0], x_min)
                            y_max = max(p1_temp[1], p2_temp[1], y_max)
                            y_min = min(p1_temp[1], p2_temp[1], y_min)

                        # 调整过滤条件，对行人使用更小的阈值
                        min_area = 50 if obj_class == 'person' else 100
                        min_width = 10 if obj_class == 'person' else 20

                        if (y_max - y_min) * (x_max - x_min) > min_area and (x_max - x_min) > min_width:
                            if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max),
                                                                                                     image_h, image_w):
                                # 绘制地面真实边界框
                                color = color_map[obj_class]
                                # 确保坐标是整数
                                x_min_int, y_min_int, x_max_int, y_max_int = int(x_min), int(y_min), int(x_max), int(y_max)
                                cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), color, 1)
                                cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), color, 1)
                                cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), color, 1)
                                cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), color, 1)

                            ground_truth_annotations.append({
                                "dco": True,
                                "height": int(y_max - y_min),
                                "width": int(x_max - x_min),
                                "id": obj_class,
                                "y": int(y_min),
                                "x": int(x_min)
                            })
            annotations.append({
                "timestamp": timestamp_sec,
                "num": image.frame,
                "class": "frame",
                "annotations": ground_truth_annotations
            })

            gt_output = [{
                "frames": annotations,
                "class": "video",
                "filename": "groundtruth.json"
            }]

            # 保存轨迹数据 - 使用自定义编码器
            trajectories_data = {
                "vehicle_trajectories": {int(k): v for k, v in vehicle_trajectories.items()},
                "person_trajectories": {int(k): v for k, v in person_trajectories.items()},
                "total_frames": frames_count
            }

            with open('groundtruth.json', 'w') as json_file:
                json.dump(gt_output, json_file, cls=NumpyEncoder)

            with open('hypo.json', 'w') as json_file:
                json.dump(hypo, json_file, cls=NumpyEncoder)

            with open('trajectories.json', 'w') as json_file:
                json.dump(trajectories_data, json_file, indent=2, cls=NumpyEncoder)

            cv2.imshow('Multi-Object Tracking', img)
            print(f"Frame: {frames_count}, Vehicles: {len(vehicle_trajectories)}, Persons: {len(person_trajectories)}")

            if cv2.waitKey(1) == ord('q'):
                clear()
                break

        except KeyboardInterrupt as e:
            clear()
            break

camera.stop()
camera.destroy()
# vehicle.destroy()

# 清理所有actor
for vehicle in vehicle_list:
    if vehicle.is_alive:
        vehicle.destroy()

for pedestrian in pedestrian_list:
    if pedestrian.is_alive:
        pedestrian.destroy()

cv2.destroyAllWindows()