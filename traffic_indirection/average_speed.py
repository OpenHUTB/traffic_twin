import carla 
import redis 
import math 
import time
client = carla.Client('localhost', 2000) 
world = client.get_world()
my_map = world.get_map()
# 连接到Redis服务器（默认是本地host和6379端口）  
redis_connection = redis.Redis(host='localhost', port=6379, db=0) 
while True:
    time.sleep(0.5)
    actors = world.get_actors() 
    #统计所有车辆数目
    vehicle_count = 0
    #统计所有车辆速度总和
    speed_sum = 0
    for actor in actors:
        type_id = actor.type_id
        if type_id.startswith("vehicle"):
            vehicle_count += 1 
            vector3D = actor.get_velocity()
            x = vector3D.x
            y = vector3D.y
            z = vector3D.z
            vector = math.sqrt(x**2 + y**2+z**2)
            speed_sum += vector
    average_speed = 0
    if vehicle_count>0:
        average_speed = speed_sum/vehicle_count
    #将数据实时存储到redis中
    redis_connection.set('average_speed',average_speed)
    #print(average_speed)