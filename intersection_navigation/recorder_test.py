import carla
import time
import os

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")   # 将文件保存在桌面
filename = os.path.join(desktop_path, 'recording01.log')


def start_recording():
    try:
        # 连接到Carla服务器
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        # 开始录制仿真数据
        additional_data = False  # 决定是否记录附加数据（如包围盒位置、物理控制参数等
        client.start_recorder(filename, additional_data)
        print(f"Started recording to {filename} with additional data: {additional_data}")

        # 让模拟运行一段时间
        time.sleep(30)  # 录制 30 秒钟

        # 停止录制
        client.stop_recorder()
    finally:
        print("Stopped recording.")


if __name__ == "__main__":
    start_recording()
