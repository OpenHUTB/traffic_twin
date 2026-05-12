import os
import glob
import numpy as np
import scipy.io as sio

# 相机位姿参数
relativePose_to_egoVehicle = {
    "back_camera": [-7.00, 0.00, 2.62, -180.00, 0.00, 0.00],
    "front_camera": [7.00, 0.00, 2.62, 0.00, 0.00, 0.00],
    "front_left_camera": [7.00, 4.00, 2.62, 90.00, 0.00, 0.00],
    "front_right_camera": [7.00, -4.00, 2.62, -90.00, 0.00, 0.00],
    "left_camera": [0.00, 4.00, 2.62, 90.00, 0.00, 0.00],
    "right_camera": [0.00, -4.00, 2.62, -90.00, 0.00, 0.00]
}
camera_names = list(relativePose_to_egoVehicle.keys())
# 雷达位姿
relativePose_lidar_to_egoVehicle = [0, 0, 0.82, 0, 0, 0]

def extract_target_data(data_dict):
    if '数据类型' not in data_dict:
        return None, None
    data_type = data_dict['数据类型']

    try:
        if data_type == 'ptd':
            measurement = np.array([
                float(data_dict['x']), float(data_dict['y']), float(data_dict['z']),
                float(data_dict['l']), float(data_dict['w']), float(data_dict['h']),
                0.0, 0.0, float(data_dict['yaw'])
            ], dtype=np.float32)
            return 'ptd', measurement

        elif data_type == 'img':
            measurement = np.array([
                float(data_dict['x']), float(data_dict['y']),
                float(data_dict['w']), float(data_dict['h'])
            ], dtype=np.float32)

            cam_id = data_dict.get('相机编号', 'unknown')

            # 提取类别
            category = np.array([
                data_dict.get('类别', 'unknown')
            ], dtype=object)

            target_id = int(data_dict.get('编号', -1))

            return 'img', (cam_id, measurement, category, target_id)

    except (KeyError, ValueError):
        pass
    return None, None


def process_single_file(txt_file_path, output_dir, current_time, feat_lookup=None, junction_name=None):
    # 雷达目标存储列表
    ptd_targets = []
    # 为每个相机建立一个独立的边框列表和类别列表
    camera_targets = {name: [] for name in camera_names}
    camera_labels = {name: [] for name in camera_names}
    # 新增编号列表和特征值列表
    camera_ids = {name: [] for name in camera_names}
    camera_features = {name: [] for name in camera_names}

    # 逐行读取与解析
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            data_dict = {}
            for part in line.split(','):
                if ':' in part:
                    key, value = part.split(':', 1)
                    data_dict[key.strip()] = value.strip()

            dtype, target_data = extract_target_data(data_dict)

            # 点云，图片分类装载数据
            if dtype == 'ptd':
                ptd_targets.append(target_data)
            elif dtype == 'img':
                cam_id, measurement, category, target_id = target_data
                if cam_id in camera_targets:
                    camera_targets[cam_id].append(measurement)
                    camera_labels[cam_id].append(category)
                    camera_ids[cam_id].append(target_id)

                    if feat_lookup and junction_name:
                        key = (junction_name, cam_id, target_id)
                        feat = feat_lookup.get(key, [0.0]*24)
                    else:
                        feat = [0.0]*24    # 无特征时填0
                    camera_features[cam_id].append(feat)
                    # camera_features[cam_id].append(feat.tolist())

    # LidarData 结构
    LidarData = {
        'Timestamp': current_time,
        'Pose': {
            'Position': np.array(relativePose_lidar_to_egoVehicle[:3]).reshape(1, 3),
            'Velocity': np.array([0, 0, 0]).reshape(1, 3),
            'Orientation': np.array(relativePose_lidar_to_egoVehicle[3:]).reshape(1, 3)
        },
        'Detections': ptd_targets if ptd_targets else []
    }

    # CameraData 结构
    CameraData = np.zeros(len(camera_names), dtype=[
        ('ImagePath', 'O'),
        ('Pose', 'O'),
        ('Timestamp', 'float32'),
        ('Detections', 'O'),
        ('Category', 'O'),
        ('Features', 'O')
    ])

    # 提取纯帧号
    frame_name = os.path.basename(txt_file_path).replace('.txt', '')

    for i, cam_name in enumerate(camera_names):
        pose_params = relativePose_to_egoVehicle[cam_name]

        # 将每个相机的单独提取出来
        cam_pose = {
            'Position': np.array(pose_params[:3]).reshape(1, 3),
            'Velocity': np.array([0, 0, 0]).reshape(1, 3),
            'Orientation': np.array(pose_params[3:]).reshape(1, 3)
        }

        # 该相机在此帧检测到的目标
        dets = camera_targets[cam_name]
        labels = camera_labels[cam_name]

        CameraData[i] = (
            f"camera/{cam_name}",  # ImagePath
            cam_pose,  # Pose
            current_time,  # Timestamp
            dets if dets else [],  # Detections
            labels if labels else [],  # Category
            feat if feat else[]   # Feat
        )

    # 封装成 datalog
    datalog = {
        'LidarData': LidarData,
        'CameraData': CameraData
    }

    # 保存至 .mat 文件
    mat_filename = frame_name + '.mat'
    mat_file_path = os.path.join(output_dir, mat_filename)
    sio.savemat(mat_file_path, {'datalog': datalog})


def load_features_for_junction(feature_dir, junction_name):
    """
    解析指定路口的全部帧特征文件，返回字典 key=(junc_name, cam_id, target_id) -> 24维特征np.array
    """
    feat_dict = {}
    junction_feat_dir = os.path.join(feature_dir, junction_name)
    if not os.path.exists(junction_feat_dir):
        print(f"特征文件夹不存在 {junction_feat_dir}")
        return feat_dict

    feat_files = sorted(glob.glob(os.path.join(junction_feat_dir, 'frame_*.txt')))
    for fpath in feat_files:
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 找到 "24维特征:" 的位置，把行分成 前半部分(键值对) 和 后半部分(特征列表)
                feature_key = '24维特征:'
                idx = line.find(feature_key)
                if idx == -1:
                    continue

                front_part = line[:idx].strip()
                back_part = line[idx + len(feature_key):].strip()

                # 解析前半部分键值对 (路口号, 相机编号, 编号)
                data = {}
                # 去掉末尾可能的逗号
                if front_part.endswith(','):
                    front_part = front_part[:-1]
                pairs = [p.strip() for p in front_part.split(',')]
                for pair in pairs:
                    if ':' in pair:
                        k, v = pair.split(':', 1)
                        data[k.strip()] = v.strip()

                try:
                    junction_num = int(float(data.get('路口号', '0')))
                    cam_id = data.get('相机编号', '')
                    target_id = int(data.get('编号', -1))
                except (ValueError, KeyError):
                    continue

                # 解析特征列表: 去除方括号，然后按逗号分割
                feature_str = back_part.strip()
                if feature_str.startswith('[') and feature_str.endswith(']'):
                    feature_str = feature_str[1:-1]
                feat_values = [x.strip() for x in feature_str.split(',') if x.strip()]
                if len(feat_values) != 24:
                    print(f"警告：{fpath} 中特征维度不为24，跳过：{line}")
                    continue

                # features = np.array(feat_values, dtype=np.float32)
                features = np.array(feat_values, dtype=np.float32).tolist()

                # 构建字典键，统一使用 "juncX" 格式
                junc_name_key = f'junc{junction_num}'
                # feat_dict[(junc_name_key, cam_id, target_id)] = features
                feat_dict[(junc_name_key, cam_id, target_id)] = features

    return feat_dict

def batch_extract_folder(input_folder, output_folder, feat_lookup=None, junction_name=None):
    os.makedirs(output_folder, exist_ok=True)
    txt_files = glob.glob(os.path.join(input_folder, '*.txt'))
    txt_files.sort()
    current_time = 0

    if not txt_files:
        print(f"在文件夹 '{input_folder}' 中未找到 .txt 文件！")
        return

    for idx, file_path in enumerate(txt_files, 1):
        filename = os.path.basename(file_path)
        try:
            process_single_file(file_path, output_folder, current_time, feat_lookup=feat_lookup, junction_name=junction_name)
        except Exception as e:
            print(f"[{idx}/{len(txt_files)}] ❌ {filename} 失败: {e}")
        current_time += 0.05

    print("完成转换")


if __name__ == '__main__':
    # 设定要处理的路口编号范围 (1 到 5)
    junction_numbers = range(1, 6)
    FEATURE_DIR = './split_data1'

    for i in junction_numbers:
        junction_name = f'junc{i}'

        # 动态生成输入和输出路径
        INPUT_DIR = f'./split_data/{junction_name}'
        OUTPUT_DIR = f'./Town10HD_Opt/test_data_{junction_name}'

        print(f" 正在处理路口: {junction_name}")

        # 检查输入文件夹是否存在
        if not os.path.exists(INPUT_DIR):
            print(f" 找不到输入文件夹 '{INPUT_DIR}'，跳过当前路口。")
            continue

        # 加载该路口的特征字典
        feat_dict = load_features_for_junction(FEATURE_DIR, junction_name)

        # 调用批处理函数
        batch_extract_folder(INPUT_DIR, OUTPUT_DIR, feat_lookup=feat_dict, junction_name=junction_name)

    print("\n 所有指定路口数据提取任务已全部完成！")