# 评估孪生效果
import math

import numpy as np
from fastdtw import fastdtw


def trajectory_overlap_dtw(truth_trajectory, track_trajectory, threshold=0.5):
    """
    使用动态时间规整（DTW）计算两条轨迹的重合度。

    :param truth_trajectory: 真实轨迹，格式为 [[x1, y1, t1], [x2, y2, t2], ...]。
    :param track_trajectory: 控制轨迹，格式为 [[x1, y1, z1], [x2, y2, z2], ...]。
    :param threshold: 判断重合点的位置误差阈值（默认 0.5 米）。
    :return: 轨迹重合度（0 到 1 之间的值，1 表示完全重合）。
    """
    # 提取轨迹的 x, y 坐标
    truth_points = np.array([[point[0], point[1]] for point in truth_trajectory])
    track_points = np.array([[point[0], point[1]] for point in track_trajectory])

    # 确保两条轨迹长度一致（取较短的长度）
    min_length = min(len(truth_points), len(track_points))
    truth_points = truth_points[:min_length]
    track_points = track_points[:min_length]

    # 使用 DTW 计算轨迹之间的距离
    distance, _ = fastdtw(truth_points, track_points)

    # 计算最大可能距离（假设两条轨迹完全不重合）
    max_distance = np.max(np.linalg.norm(truth_points - track_points, axis=1)) * min_length

    # 计算重合度
    overlap_ratio = 1 - (distance / max_distance)

    # 限制重合度在 [0, 1] 之间
    return max(0, min(1, overlap_ratio))


def trajectory_metrics(truth, track, threshold=0.5):
    """
    计算轨迹的多个指标：
    1. 平均轨迹重合度（Mean Trajectory Overlap Ratio, TOR）
    2. 平均位置误差（Mean Position Error, MPE）
    3. 平均最大位置误差（Mean Maximum Position Error, MeanMaxPE）
    4. 平均终点误差（Mean Final Position Error, MFPE）

    :param truth: 车辆的轨迹字典，键为车辆编号，值为 [[x1, y1, t1], [x2, y2, t2], ...]。
    :param track: 控制车辆所走的轨迹字典，键为车辆编号，值为 [[x1, y1, z1], [x2, y2, z2], ...]。
    :param threshold: 判断重合点的位置误差阈值（默认 0.5 米）。
    :return: 平均轨迹重合度、平均位置误差、平均最大位置误差和平均终点误差的元组 (mean_tor, mean_error, mean_max_error, mean_fpe)。
    """
    total_error = 0.0
    total_max_error = 0.0
    total_tor = 0.0
    total_fpe = 0.0
    num_vehicles = 0
    total_points = 0

    # 遍历 truth 和 track 的键和值
    for (truth_id, truth_trajectory), (track_id, track_trajectory) in zip(truth.items(), track.items()):
        # 确保轨迹长度一致
        min_length = min(len(truth_trajectory), len(track_trajectory))
        if min_length == 0:
            continue  # 如果轨迹为空，跳过

        # 初始化当前车辆的指标
        vehicle_error = 0.0
        vehicle_max_error = 0.0

        # 计算当前车辆的误差
        for i in range(min_length):
            truth_point = truth_trajectory[i]  # 真实轨迹点 [x, y, t]
            track_point = track_trajectory[i]  # 控制轨迹点 [x, y, z]

            # 计算欧氏距离（仅考虑 x 和 y）
            error = math.sqrt((truth_point[0] - track_point[0]) ** 2 + (truth_point[1] - track_point[1]) ** 2)
            vehicle_error += error
            if error > vehicle_max_error:
                vehicle_max_error = error  # 更新当前车辆的最大误差
            total_points += 1

        # 计算当前车辆的轨迹重合度（使用 DTW）
        vehicle_tor = trajectory_overlap_dtw(truth_trajectory, track_trajectory, threshold)

        # 计算当前车辆的终点误差
        truth_end_point = truth_trajectory[-1]  # 真实轨迹的终点 [x, y, t]
        track_end_point = track_trajectory[-1]  # 控制轨迹的终点 [x, y, z]
        fpe = math.sqrt((truth_end_point[0] - track_end_point[0]) ** 2 + (
                    truth_end_point[1] - track_end_point[1]) ** 2)
        total_fpe += fpe

        # 累加当前车辆的指标
        total_error += vehicle_error
        total_max_error += vehicle_max_error
        total_tor += vehicle_tor
        num_vehicles += 1

    # 计算平均指标
    mean_error = total_error / total_points if total_points > 0 else 0.0
    mean_max_error = total_max_error / num_vehicles if num_vehicles > 0 else 0.0
    mean_tor = total_tor / num_vehicles if num_vehicles > 0 else 0.0
    mean_fpe = total_fpe / num_vehicles if num_vehicles > 0 else 0.0

    return mean_tor, mean_error, mean_max_error, mean_fpe


def mean_metrics(lateral_errors, longitudinal_errors, delays):
    """
    计算所有车辆的平均横向误差（Mean Lateral Error, MLE）、平均纵向误差（Mean Longitudinal Error, MLOE）和平均延迟（Mean Delay, MD）。

    :param lateral_errors: 横向误差列表，每个子列表表示一辆车的每一帧的横向误差。
    :param longitudinal_errors: 纵向误差列表，每个子列表表示一辆车的每一帧的纵向误差。
    :param delays: 延迟列表，每个子列表表示一辆车的每一帧的延迟。
    :return: 平均横向误差、平均纵向误差和平均延迟的元组 (mean_lateral_error, mean_longitudinal_error, mean_delay)。
    """
    total_lateral_error = 0.0
    total_longitudinal_error = 0.0
    total_delay = 0.0
    total_frames = 0

    # 遍历每辆车的横向误差、纵向误差和延迟列表
    for lateral_vehicle, longitudinal_vehicle, delay_vehicle in zip(lateral_errors, longitudinal_errors, delays):
        if len(lateral_vehicle) == 0 or len(longitudinal_vehicle) == 0 or len(delay_vehicle) == 0:
            continue  # 如果车辆没有数据，跳过

        # 确保横向误差、纵向误差和延迟的帧数一致
        min_length = min(len(lateral_vehicle), len(longitudinal_vehicle), len(delay_vehicle))
        if min_length == 0:
            continue  # 如果帧数为 0，跳过

        # 累加当前车辆的横向误差、纵向误差和延迟
        total_lateral_error += sum(abs(e) for e in lateral_vehicle[:min_length])
        total_longitudinal_error += sum(abs(e) for e in longitudinal_vehicle[:min_length])
        total_delay += sum(d for d in delay_vehicle[:min_length])
        total_frames += min_length

    # 计算平均横向误差、平均纵向误差和平均延迟
    if total_frames == 0:
        return 0.0, 0.0, 0.0  # 如果没有数据，返回 (0.0, 0.0, 0.0)
    mean_lateral_error = total_lateral_error / total_frames
    mean_longitudinal_error = total_longitudinal_error / total_frames
    mean_delay = total_delay / total_frames
    return mean_lateral_error, mean_longitudinal_error, mean_delay
