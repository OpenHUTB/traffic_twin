import os
import glob
import re


def split_data_by_junction(input_file_path, base_output_dir):
    """
    核心拆分逻辑：处理单个 txt 文件，按路口号拆分
    """
    original_filename = os.path.basename(input_file_path)
    file_handles = {}
    junc_pattern = re.compile(r'路口号:\s*([^,]+)')

    # 定义异常数据的关键字列表
    abnormal_keywords = ['NULL', 'null', 'NaN', 'None']

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in:
            for line_num, line in enumerate(f_in, 1):
                line_clean = line.strip()
                if not line_clean:
                    continue

                # 检查当前行是否包含任何异常关键字
                for keyword in abnormal_keywords:
                    if keyword in line_clean:
                        # 抛出 ValueError，中断当前文件的处理
                        raise ValueError(f"在第 {line_num} 行检测到异常数据 '{keyword}' -> 内容: {line_clean}")

                match = junc_pattern.search(line_clean)
                if match:
                    junction_id = match.group(1).strip()

                    if junction_id not in file_handles:
                        junc_folder = os.path.join(base_output_dir, junction_id)
                        os.makedirs(junc_folder, exist_ok=True)
                        target_file_path = os.path.join(junc_folder, original_filename)
                        # 开启写入通道
                        file_handles[junction_id] = open(target_file_path, 'w', encoding='utf-8')

                    # 写入当前行数据
                    file_handles[junction_id].write(line_clean + '\n')
    finally:
        # 安全关闭当前文件的所有写入通道
        for f_out in file_handles.values():
            f_out.close()


def batch_process_folder(input_folder, output_folder):
    """
    遍历整个文件夹，对所有 .txt 文件执行拆分
    """
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f" 找不到输入文件夹 '{input_folder}'，请检查路径是否正确。")
        return

    # 找出该文件夹下所有的 .txt 文件
    search_pattern = os.path.join(input_folder, '*.txt')
    txt_files = glob.glob(search_pattern)

    if not txt_files:
        print(f" 在 '{input_folder}' 中没有找到任何 .txt 文件。")
        return

    txt_files.sort()
    total_files = len(txt_files)

    # 循环处理每一个文件
    for idx, file_path in enumerate(txt_files, 1):
        filename = os.path.basename(file_path)
        try:
            # 处理单个文件
            split_data_by_junction(file_path, output_folder)

        except Exception as e:
            print(f"[{idx}/{total_files}] ❌ {filename} 处理失败，错误信息: {e}")

    print("\n 所有文件批量处理完成！")
    print(f" 整理好的数据已保存在: {os.path.abspath(output_folder)}")


if __name__ == '__main__':
    # 原始数据路径
    INPUT_DIR = './v2x_latency_logs'

    # 数据输出路径
    OUTPUT_DIR = './split_data'

    batch_process_folder(INPUT_DIR, OUTPUT_DIR)