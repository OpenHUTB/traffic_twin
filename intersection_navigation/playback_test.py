"""
    复现recorder没有最大速度限制，主要是视觉可观测的大小，设置100-200倍数都没有问题！！
"""

import carla


def replay_simulation():
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10)

    # 定义参数
    name = r'C:\Users\ASUS\Desktop\recording01.log'
    start = float(0.0)
    duration = float(30.0)
    replay_sensors = bool(False)
    follow_id = 0
    # 调用重放方法
    # replay_file(self, name, start, duration, follow_id, replay_sensors)
    client.replay_file(name, start, duration, follow_id, replay_sensors)
    client.set_replayer_time_factor(200.0)
    # follow_id 指定要跟随的演员的ID。重放期间，仿真环境中的摄像机将会跟随这个特定的演员。如果设置为0，则相机不会跟随任何演员
    # replay_sensors bool 选择是否重新生成传感器数据

    # 现有车辆再跑，再复现完了后，原来跑的车辆处于自动驾驶模式，而行人则停止

    # 若是单独的复现的话，需要记录车辆的id，而且复现完了后，车辆不会进入自动驾驶模式，而是停止


if __name__ == "__main__":
    replay_simulation()