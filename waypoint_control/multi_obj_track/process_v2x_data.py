import os
import re
import shutil
import scipy.io as sio


def get_min_send_time(file_path):
    """提取文件中的最早发送时间"""
    pattern = re.compile(r"发送时间:\s*([0-9\.]+)")
    times = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    times.append(float(match.group(1)))
        return min(times) if times else None
    except Exception as e:
        print(f"读取 {file_path} 出错: {e}")
        return None


def get_max_receive_time(file_path):
    """提取文件中的最晚接收时间"""
    pattern = re.compile(r"接收时间:\s*([0-9\.]+)")
    times = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    times.append(float(match.group(1)))
        return max(times) if times else None
    except Exception as e:
        print(f"读取 {file_path} 出错: {e}")
        return None


def calculate_simulation_time(folder_path):
    """
    计算指定文件夹内所有帧数据的总耗时，并将结果保存为 mat 和 txt
    """
    all_files = [f for f in os.listdir(folder_path) if f.startswith("frame_") and f.endswith(".txt")]

    if not all_files:
        print("错误：在指定文件夹中没有找到匹配的数据文件！")
        return False

    all_files.sort()
    first_file = os.path.join(folder_path, all_files[0])
    last_file = os.path.join(folder_path, all_files[-1])

    print(f"原始第一帧: {all_files[0]}")
    print(f"原始最后一帧: {all_files[-1]}")

    start_time = get_min_send_time(first_file)
    end_time = get_max_receive_time(last_file)

    if start_time and end_time:
        total_seconds = end_time - start_time
        print(f"数据传输总耗时: {total_seconds:.4f} 秒")

        # 保存结果到原始文件夹中
        mat_filename = os.path.join(folder_path, "simulation_total_time.mat")
        sio.savemat(mat_filename, {'total_time': total_seconds})
        print(f"已保存为 MATLAB 原生文件: {mat_filename}")

        txt_filename = os.path.join(folder_path, "simulation_total_time_pure.txt")
        with open(txt_filename, 'w') as f:
            f.write(f"{total_seconds:.4f}")
        print(f"已保存为 txt 备用文件: {txt_filename}")
        return True
    else:
        print("错误：未能成功提取时间。如果首尾帧数据完全损坏，可能会导致此错误。")
        return False


def clean_and_resort_frames(source_folder, target_folder, required_lines=35):
    """
    清洗并重排帧数据，剔除不完整的帧
    """

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f" 创建了新的目标文件夹: {target_folder}")
    else:
        print(f" 目标文件夹已存在: {target_folder}")

    all_files = [f for f in os.listdir(source_folder) if f.startswith("frame_") and f.endswith(".txt")]
    all_files.sort()

    valid_count = 0
    discarded_count = 0

    for original_filename in all_files:
        source_filepath = os.path.join(source_folder, original_filename)

        with open(source_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            valid_lines = [line.strip() for line in lines if line.strip()]

        line_count = len(valid_lines)

        if line_count == required_lines:
            new_filename = f"frame_{valid_count:06d}.txt"
            target_filepath = os.path.join(target_folder, new_filename)
            shutil.copy2(source_filepath, target_filepath)
            valid_count += 1
        else:
            print(f"  剔除: {original_filename} (仅包含 {line_count} 条数据)")
            discarded_count += 1

    print(f" 完成数据清洗！保留并重排: {valid_count} 个文件，剔除: {discarded_count} 个文件。")


def main():
    # 配置文件路径
    raw_data_folder = "./v2x_latency_logs"  # 含有全部原始数据的文件夹
    cleaned_data_folder = "./v2x_clean_logs"  # 清洗并重排后存放数据的目标文件夹

    # 假设每帧完整应该有的条数
    REQUIRED_LINES_PER_FRAME = 7

    # 计算总耗时
    calculate_simulation_time(folder_path=raw_data_folder)

    # 清洗数据，剔除坏帧并重排
    clean_and_resort_frames(
        source_folder=raw_data_folder,
        target_folder=cleaned_data_folder,
        required_lines=REQUIRED_LINES_PER_FRAME
    )

    print("\n 全部数据处理完毕！")


if __name__ == "__main__":
    main()