"""
   显示全部交通灯的红绿灯时间
"""
import carla
show_traffic_time = {
    -1: carla.Location(x=-608.165161, y=-205.942261, z=70.1),
    -2: carla.Location(x=-576.757751, y=-225.365173, z=69.7),
    -3: carla.Location(x=-610.242310, y=-181.834778, z=69.8),
    -4: carla.Location(x=196.691071, y=-533.198120, z=53.2),
    -5: carla.Location(x=279.408966, y=-522.701843, z=53.2),
    -6: carla.Location(x=259.294281, y=-543.615967, z=53.5),
    -7: carla.Location(x=871.028625, y=101.347870, z=57.5),
    -8: carla.Location(x=860.652100, y=147.128693, z=56.5),
    -9: carla.Location(x=847.039978, y=41.110916, z=58.6),
    -10: carla.Location(x=901.015930, y=710.794250, z=41.8),
    -11: carla.Location(x=870.408508, y=646.815979, z=42.3),
    -12: carla.Location(x=852.884033, y=658.005676, z=42.2),
    -13: carla.Location(x=884.594299, y=722.484070, z=42),
    -14: carla.Location(x=-908.415894, y=330.243500, z=50.8),
    -15: carla.Location(x=-928.443420, y=331.184052, z=50.8),
    -16: carla.Location(x=-932.984070, y=376.556519, z=51.3),
    -17: carla.Location(x=102.694298, y=17.884031, z=51),
    -18: carla.Location(x=118.788002, y=66.2, z=50.5),
    -19: carla.Location(x=49.691074, y=36.101894, z=51.5),
    -20: carla.Location(x=70.605713, y=72.615997, z=51)
}


def main():
    # 连接到Carla服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    try:
        # 尝试连接
        world = client.get_world()
        print("Connected to Carla server!")
    except Exception as e:
        print(f"Error connecting to Carla: {e}")
        return
    # 确保世界状态同步
    world.tick()
    # 获取世界和所有交通信号灯
    traffic_lights = world.get_actors().filter('traffic.traffic_light')
    if not traffic_lights:
        print("No traffic lights found.")
        return
    # 存储所有交通信号灯的状态和倒计时
    traffic_light_info = {}
    # 遍历所有交通信号灯
    for light in traffic_lights:
        opendrive_id = light.get_opendrive_id()

        # 如果信号灯的ID在字典中，保存其初始状态
        if float(opendrive_id) in show_traffic_time:
            location = show_traffic_time[float(opendrive_id)]

            # 获取绿灯、红灯、黄灯的持续时间
            green_time = light.get_green_time()
            red_time = light.get_red_time()
            yellow_time = light.get_yellow_time()

            # 保存初始状态
            traffic_light_info[float(opendrive_id)] = {
                'light': light,
                'location': location,
                'green_time': green_time,
                'red_time': red_time,
                'yellow_time': yellow_time,
                'last_light_state': light.state,
                'time_left': int(green_time),
                'last_update_time': 0.0
            }

    while True:
        # 获取当前仿真时间（秒）
        simulation_time = world.get_snapshot().timestamp.elapsed_seconds

        # 更新每个交通信号灯的倒计时
        for opendrive_id, light_info in traffic_light_info.items():
            light = light_info['light']
            location = light_info['location']
            green_time = light_info['green_time']
            red_time = light_info['red_time']
            yellow_time = light_info['yellow_time']
            last_light_state = light_info['last_light_state']
            time_left = light_info['time_left']
            last_update_time = light_info['last_update_time']

            # 获取时间差
            delta_time = simulation_time - last_update_time

            # 只有当时间流逝超过 1 秒时才更新倒计时
            if delta_time >= 1.0:
                light_info['last_update_time'] = simulation_time
                color = None
                # 获取当前信号灯状态
                light_state = light.state
                # 根据信号灯状态设置倒计时和颜色
                if light_state == carla.TrafficLightState.Green:
                    color = carla.Color(0, 255, 0)  # 绿灯显示绿色
                elif light_state == carla.TrafficLightState.Red:
                    color = carla.Color(255, 0, 0)  # 红灯显示红色
                elif light_state == carla.TrafficLightState.Yellow:
                    color = carla.Color(255, 255, 0)  # 黄灯显示黄色

                # 只有信号灯状态变化时才更新倒计时
                if light_state != last_light_state:
                    # 根据信号灯状态设置倒计时和颜色
                    if light_state == carla.TrafficLightState.Green:
                        time_left = int(green_time)
                        color = carla.Color(0, 255, 0)  # 绿灯显示绿色
                    elif light_state == carla.TrafficLightState.Red:
                        time_left = int(red_time)
                        color = carla.Color(255, 0, 0)  # 红灯显示红色
                    elif light_state == carla.TrafficLightState.Yellow:
                        time_left = int(yellow_time)
                        color = carla.Color(255, 255, 0)  # 黄灯显示黄色
                    light_info['time_left'] = time_left  # 更新倒计时

                # 每次 tick 后，减少 1 秒的时间
                light_info['time_left'] -= 1

                # 如果倒计时小于零，重置倒计时
                if light_info['time_left'] <= 0:
                    light_info['time_left'] = 0

                # 显示当前倒计时
                world.debug.draw_string(location, str(time_left), life_time=0.1, color=color)
                # 更新信号灯状态
                light_info['last_light_state'] = light_state


if __name__ == "__main__":
    main()
