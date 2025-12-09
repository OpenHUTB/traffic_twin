import pickle
import numpy as np
import os
from scipy.io import savemat


def save_frames_to_mat(pkl_file, output_dir="mat_results", score_threshold=0.6):
    """
    将每一帧的检测结果保存为.mat文件
    按照car, truck, pedestrian的顺序保存为[x, y, z, l, w, h, 0, 0, theta]格式
    只保存score>threshold的数据

    参数:
        pkl_file: 输入的.pkl文件路径
        output_dir: 输出.mat文件的目录
        score_threshold: 置信度阈值
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载pkl文件
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)

    print(f"总帧数: {len(results)}")
    print(f"置信度阈值: {score_threshold}")
    print(f"输出目录: {output_dir}")

    # 定义类别顺序
    category_order = ['Car', 'Truck', 'Pedestrian']

    for frame_idx, frame_result in enumerate(results):
        # 获取当前帧的数据
        frame_id = frame_result.get('frame_id', f'{frame_idx:06d}')
        boxes_lidar = frame_result.get('boxes_lidar', [])
        scores = frame_result.get('score', [])
        names = frame_result.get('name', [])

        # 如果没有检测对象，跳过
        if len(boxes_lidar) == 0:
            continue

        # 初始化存储所有符合条件对象的列表
        all_detections = []

        # 按照类别顺序处理
        for category in category_order:
            # 找到当前类别的所有检测对象
            for i in range(len(boxes_lidar)):
                # 检查类别是否匹配（忽略大小写）
                if i >= len(names):
                    continue

                current_name = names[i]
                current_score = scores[i] if i < len(scores) else 0.0

                # 检查类别和置信度
                if (current_name.lower() == category.lower() and
                        current_score > score_threshold):
                    box = boxes_lidar[i]
                    # 转换为目标格式: [x, y, z, l, w, h, 0, 0, theta]
                    detection = [
                        float(box[0]),  # x
                        float(box[1]),  # y
                        float(box[2]),  # z
                        float(box[3]),  # l (长度)
                        float(box[4]),  # w (宽度)
                        float(box[5]),  # h (高度)
                        0.0,  # 占位符1
                        0.0,  # 占位符2
                        float(box[6])  # yaw (旋转角)
                    ]
                    all_detections.append(detection)

        # 转换为numpy数组
        if all_detections:
            detections_array = np.array(all_detections, dtype=np.float32)
        else:
            # 如果没有符合条件的检测对象，创建一个空数组
            detections_array = np.zeros((0, 9), dtype=np.float32)

        # 构建输出数据字典（MATLAB兼容格式）
        mat_data = {
            'datapy': {
                'detections': detections_array,  # 主要数据
                'frame_id': frame_id,  # 帧ID
                'num_detections': len(detections_array),  # 检测数量
                'score_threshold': score_threshold  # 使用的阈值
            }
        }

        # 生成输出文件名
        # 使用帧ID作为文件名，如果没有帧ID则使用序号
        if frame_id.isdigit():
            filename = f"{int(frame_id):06d}.mat"
        else:
            filename = f"{frame_id}.mat"

        output_path = os.path.join(output_dir, filename)

        # 保存为.mat文件
        savemat(output_path, mat_data)

        # 每处理100帧打印一次进度
        if (frame_idx + 1) % 100 == 0 or frame_idx == len(results) - 1:
            print(f"已处理 {frame_idx + 1}/{len(results)} 帧")

    print(f"\n处理完成！所有.mat文件已保存到: {output_dir}")

    # 统计信息
    print(f"共处理了 {len(results)} 帧数据")
    print(f"生成的.mat文件保存在: {os.path.abspath(output_dir)}")


def check_mat_files(output_dir="mat_results", max_files=5):
    """
    检查生成的.mat文件内容
    """
    import glob
    from scipy.io import loadmat

    mat_files = sorted(glob.glob(os.path.join(output_dir, "*.mat")))

    if not mat_files:
        print(f"在 {output_dir} 中没有找到.mat文件")
        return

    print(f"\n找到 {len(mat_files)} 个.mat文件")
    print("检查前几个文件的内容:")

    for i, mat_file in enumerate(mat_files[:max_files]):
        print(f"\n{'=' * 60}")
        print(f"文件 {i + 1}: {os.path.basename(mat_file)}")

        data = loadmat(mat_file)

        # 显示基本信息
        frame_id = data.get('frame_id', ['Unknown'])[0]
        num_detections = data.get('num_detections', [0])[0, 0]

        print(f"帧ID: {frame_id}")
        print(f"检测对象数量: {num_detections}")

        # 显示检测对象详情
        detections = data.get('detections', np.zeros((0, 9)))

        if len(detections) > 0:
            print(f"检测对象格式 (shape): {detections.shape}")
            print("前几个检测对象:")

            for j in range(min(3, len(detections))):
                det = detections[j]
                print(f"  对象 {j}: [{det[0]:.3f}, {det[1]:.3f}, {det[2]:.3f}, "
                      f"{det[3]:.3f}, {det[4]:.3f}, {det[5]:.3f}, "
                      f"{det[6]:.0f}, {det[7]:.0f}, {det[8]:.3f}]")


# 主程序
if __name__ == "__main__":
    pkl_file_path = "result.pkl"  # 请替换为你的.pkl文件路径
    output_directory = "matdata"  # 输出目录

    try:
        # 1. 将每一帧保存为.mat文件
        save_frames_to_mat(
            pkl_file_path,
            output_dir=output_directory,
            score_threshold=0.6  # 置信度阈值，60% = 0.6
        )

        # 2. 检查生成的.mat文件
        # check_mat_files(output_dir=output_directory, max_files=3)

    except FileNotFoundError:
        print(f"错误: 找不到文件 {pkl_file_path}")
    except ImportError as e:
        print(f"错误: 缺少必要的库，请安装scipy: pip install scipy")
        print(f"详细错误: {e}")
    except Exception as e:
        print(f"处理文件时出错: {e}")
        import traceback

        traceback.print_exc()