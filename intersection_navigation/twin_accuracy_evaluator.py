"""
        设计评价孪生精准程度：将从以下几个角度来评价孪生的真实性
                    1）到达时间误差：车辆从一个路口到达另一个路口所花费的时间是最直接的评估指标。可
                    以使用平均绝对误差（MAE）或均方误差（MSE）来量化孪生模型与真实世界之间的时间差异。
                    2）路径误差：判断大于3个航点的车辆中间航点是否经过那个路口
                    3)....

        编辑距离（Levenshtein距离）：
                优点：这种方法可以直接量化仿真路径与预期路径的差异。通过计算将仿真路径转化为预期路径所需的最少
                操作次数（如添加、删除、替换路口），能够直观地反映出路径的偏差程度。
                缺点：编辑距离主要关注路径中的离散点（路口），忽略了各个路段之间的实际距离差异。这意味着它无法
                衡量车辆绕路的物理长度，只关注路径的结构变化。

"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from twin_navigation import *
# 定义存储文件的文件夹路径
output_folder = "output_results"
# 创建文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

junction_name_dict = {
    1627: "尖山路与旺龙路交叉口",
    1316: "青山路与旺龙路西交叉口",
    616: "青山路与尖山路交叉口",
    721: "旺龙路与岳麓大道桥下路口",
    201: "望青路与青山路交叉口",
    351: "岳麓大道与麓谷大道交叉口"
}

# 路口ID及其位置
junctions = {
    1627: carla.Location(x=232, y=-520, z=46),  # 尖山路与旺龙路交叉口
    1316: carla.Location(x=-591, y=-198, z=62),  # 青山路与旺龙路西交叉口
    616: carla.Location(x=83, y=48, z=44),   # 青山路与尖山路交叉口
    721:  carla.Location(x=-909, y=347.5, z=43.8),  # 旺龙路与岳麓大道桥下路口
    201: carla.Location(x=839, y=98, z=50.4),   # 望青路与青山路交叉口
    351:  carla.Location(x=877.5, y=680, z=35)   # 岳麓大道与麓谷大道交叉口
}


def calculate_vehicle_lifespan(vehicle_lifetimes, intersection_time):
    """
    计算每辆车的实际生成到销毁时间与 intersection_time 中的时间差异。
    vehicle_lifetimes: dict - 存储车辆生成和销毁的实际时间（系统时间戳） vehicle_id: (start_time, end_time)
    intersection_time: dict - 每辆车在 intersection_time 中记录的航点时间列表
    """
    time_differences = {}

    for vehicle_id, (start_time, end_time) in vehicle_lifetimes.items():
        actual_lifespan = end_time - start_time  # 实际生成到销毁的时间（秒）
        intersection_times = intersection_time.get(vehicle_id, [])
        if len(intersection_times) >= 2:
            first_timestamp = get_timestamp(intersection_times[0])
            last_timestamp = get_timestamp(intersection_times[-1])
            predicted_lifespan = last_timestamp - first_timestamp  # intersection_time 中记录的时间间隔
            time_difference = abs(actual_lifespan - predicted_lifespan)  # 差异
            time_differences[vehicle_id] = time_difference
        #     print(
        #         f"车辆ID: {vehicle_id}, 实际寿命: {actual_lifespan}s, 预测寿命: {predicted_lifespan}s, 差异: {time_difference}s")
        # else:
        #     print(f"车辆ID: {vehicle_id} 的航点时间不足，无法计算差异")

    return time_differences


def save_lifespan_to_csv(time_differences, csv_filename='vehicle_lifespan_differences.csv'):
    """
    将车辆生命周期差异保存为 CSV 文件
    :param time_differences: 车辆生命周期差异字典 {vehicle_id: time_difference, ...}
    :param csv_filename: 输出的 CSV 文件名
    """
    # 确保文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 将输出文件路径和文件名结合起来
    output_path = os.path.join(output_folder, csv_filename)

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Vehicle ID', 'Time Difference (s)'])  # 表头
        for vehicle_id, time_diff in time_differences.items():
            writer.writerow([vehicle_id, time_diff])


def plot_lifespan_differences_line(time_differences, image_filename='vehicle_lifespan_differences.png', limit=100):
    """
    根据车辆生命周期差异生成折线图并保存为图片
    :param time_differences: 车辆生命周期差异字典 {vehicle_id: time_difference, ...}
    :param image_filename: 输出的图像文件名
    :param limit: 要显示的车辆数量上限（过多车辆不适合全部显示）
    """
    # 确保文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 限制显示的车辆数量
    vehicle_ids = list(time_differences.keys())[:limit]
    time_diffs = list(time_differences.values())[:limit]

    # 创建折线图
    plt.figure(figsize=(12, 8))
    plt.plot(vehicle_ids, time_diffs, marker='o', color='blue', linestyle='-')

    # 设置标题和标签
    plt.title(f'Vehicle Lifespan Differences (Top {limit} Vehicles)')
    plt.xlabel('Vehicle ID')
    plt.ylabel('Time Difference (s)')

    # 拼接完整文件路径
    full_image_path = os.path.join(output_folder, image_filename)

    # 保存为图片
    plt.savefig(full_image_path)


def extract_middle_junction_ids(waypoints_by_vehicle):
    """
    提取车辆经过的中间路口，并用ID表示
    :param waypoints_by_vehicle: 字典，保存了每辆车的航点信息 {vehicle_id: [[inter, lane, direct], ...]}
    :return: 字典，返回每辆车经过的中间路口ID {vehicle_id: [junction_id, ...]}
    """
    vehicle_middle_junctions = {}

    for vehicle_id, waypoints in waypoints_by_vehicle.items():
        # 提取中间航点，排除第一个和最后一个
        if len(waypoints) > 2:
            middle_waypoints = waypoints[1:-1]
            middle_junction_ids = []

            for waypoint in middle_waypoints:
                intersection_name = waypoint[0]  # 获取路口名称
                # 根据路口名称查找对应的ID
                junction_id = next((jid for jid, jname in junction_name_dict.items() if jname == intersection_name),
                                   None)
                if junction_id:
                    middle_junction_ids.append(junction_id)

            # 保存该车辆的中间路口ID
            vehicle_middle_junctions[vehicle_id] = middle_junction_ids

    return vehicle_middle_junctions


def track_vehicle_actual_junctions(vehicle_list, threshold=10.0):
    """
    跟踪车辆经过的实际路口ID，并返回一个字典。
    :param vehicle_list: 字典，包含所有车辆对象 {vehicle_id: carla.Vehicle}
    :param threshold: float，判断是否经过路口的距离阈值
    :return: 字典，保存车辆经过的实际路口ID {vehicle_id: [junction_id, ...]}
    """
    vehicle_actual_junctions = {}

    for vehicle_id, vehicle in vehicle_list.items():
        current_location = vehicle.get_location()

        # 初始化该车辆的实际经过路口列表
        if vehicle_id not in vehicle_actual_junctions:
            vehicle_actual_junctions[vehicle_id] = []

        # 检查该车辆是否经过某个路口
        for junction_id, junction_location in junctions.items():
            distance = current_location.distance(junction_location)

            # 如果车辆经过了路口（距离小于阈值），记录该路口ID
            if distance < threshold and junction_id not in vehicle_actual_junctions[vehicle_id]:
                vehicle_actual_junctions[vehicle_id].append(junction_id)

    return vehicle_actual_junctions


def calculate_levenshtein_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # 初始化dp数组的边界值
    for i in range(1, m + 1):
        dp[i][0] = i  # 需要删除i个字符
    for j in range(1, n + 1):
        dp[0][j] = j  # 需要插入j个字符

    # 动态规划计算编辑距离
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0  # 如果两个字符相等，代价为0
            else:
                cost = 1  # 如果两个字符不相等，代价为1

            dp[i][j] = min(
                dp[i - 1][j] + 1,   # 删除操作
                dp[i][j - 1] + 1,   # 插入操作
                dp[i - 1][j - 1] + cost  # 替换操作
            )

    return dp[m][n]


def evaluate_path_error(vehicle_middle_junctions, vehicle_actual_junctions):
    """
    计算每辆车的预期路径与实际路径之间的路径误差（Levenshtein距离）。
    :param vehicle_middle_junctions: 字典，每辆车的预期中间路口 {vehicle_id: [expected_junctions]}
    :param vehicle_actual_junctions: 字典，每辆车的实际经过路口 {vehicle_id: [actual_junctions]}
    :return: 字典，每辆车的路径误差 {vehicle_id: levenshtein_distance}
    """
    path_errors = {}

    for vehicle_id in vehicle_middle_junctions:
        expected_junctions = vehicle_middle_junctions.get(vehicle_id, [])
        actual_junctions = vehicle_actual_junctions.get(vehicle_id, [])

        # 计算Levenshtein距离，即路径误差
        error = calculate_levenshtein_distance(expected_junctions, actual_junctions)
        path_errors[vehicle_id] = error

    return path_errors


def plot_path_errors(path_errors: dict, title: str = "Path Errors for Vehicles",
                     image_filename: str = 'path_errors.png'):
    """
    根据路径错误绘制散点图
    :param path_errors: 车辆路径错误字典 {vehicle_id: error_value}
    :param title: 图表的标题
    :param image_filename: 保存的图像文件名
    """
    # 确保文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 将输出文件路径和文件名结合起来
    image_path = os.path.join(output_folder, image_filename)

    # 获取车辆ID和对应的路径错误
    vehicle_ids = list(path_errors.keys())
    errors = list(path_errors.values())

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制散点图
    plt.scatter(vehicle_ids, errors, color='blue', alpha=0.7)

    # 设置标题和标签
    plt.title(title)
    plt.xlabel('Vehicle ID')
    plt.ylabel('Path Error (Levenshtein Distance)')

    # 保存为图片
    plt.savefig(image_path)

    # 显示图表（可选）
    # plt.show()

