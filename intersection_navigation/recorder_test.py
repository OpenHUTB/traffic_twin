import carla
import time


def start_recording():
    try:
        # 连接到Carla服务器
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        # 开始录制仿真数据
        filename = "C:\\Users\\ASUS\\Desktop\\recording01.log"  # 保存到默认路径
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
