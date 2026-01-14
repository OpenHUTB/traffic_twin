import carla
import random
import time


def create_pedestrian(start_location, end_location, world):
    """创建并控制行人从起点走到终点"""

    # 选择行人蓝图
    walker_bp = random.choice(
        world.get_blueprint_library().filter('walker.pedestrian.*')
    )

    # 设置行人属性
    walker_bp.set_attribute('is_invincible', 'false')

    # 生成行人
    transform = carla.Transform(start_location, carla.Rotation())
    walker = world.try_spawn_actor(walker_bp, transform)

    if walker is None:
        print("无法生成行人")
        return None
    else:
        print("行人生成成功")

    # 创建AI控制器
    controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)

    # 开始移动
    controller.start()
    controller.go_to_location(end_location)
    controller.set_max_speed(5)  # 设置速度

    return walker, controller


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 设置起点和终点
    start_location = carla.Location(x=-74.2372235552250, y=34.0550152124566, z=1)
    end_location = carla.Location(x=-64.5043248693601, y=34.0973325433420, z=1)
    # end_location = carla.Location(x=-54.2372235552250, y=37.0550152124566, z=1)
    # start_location = carla.Location(x=-38.5043248693601, y=46.0973325433420, z=1)

    # 创建行人
    walker, controller = create_pedestrian(start_location, end_location, world)
    print("行人创建成功")

    if walker:
        try:
            # 等待行人到达目的地
            while True:
                world.tick()
                current_location = walker.get_location()
                distance = current_location.distance(end_location)

                print(f"当前距离目标: {distance:.2f} 米")

                if distance < 2.0:  # 当距离小于2米时认为到达
                    print("行人已到达目的地!")
                    controller.stop()
                    break

                time.sleep(0.5)

        except KeyboardInterrupt:
            pass
        finally:
            # 清理
            controller.stop()
            walker.destroy()
            controller.destroy()


if __name__ == "__main__":
    main()