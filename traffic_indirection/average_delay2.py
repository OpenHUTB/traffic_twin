import carla 
import redis 
import math 
import time
from datetime import datetime
client = carla.Client('localhost', 2000) 
world = client.get_world()
my_map = world.get_map()
# 连接到Redis服务器（默认是本地host和6379端口）  
redis_connection = redis.Redis(host='localhost', port=6379, db=0) 
start_map = {}
while True:
    time.sleep(0.5)
    wait_map = {}
    actors = world.get_actors() 
    for actor in actors:
        type_id = actor.type_id
        if type_id.startswith("vehicle"):
            wait_map[str(actor.id)] = 0    
            vector3D = actor.get_velocity()
            x = vector3D.x
            y = vector3D.y
            z = vector3D.z
            vector = math.sqrt(x**2 + y**2+z**2)
            if vector >= 0 and vector < 1:
                if start_map.get(str(actor.id), 'None') == 'None':
                    #获取当前日期和时间，然后转换为时间戳  
                    now = datetime.now() 
                    start_map[str(actor.id)] = now.timestamp()
            elif start_map.get(str(actor.id), 'None') != 'None':
                #获取当前日期和时间，然后转换为时间戳  
                now = datetime.now()
                wait_map[str(actor.id)] = now.timestamp()-start_map[str(actor.id)]
                start_map.pop(str(actor.id))
    total_wait_time = 0
    for actor_id in wait_map:
        total_wait_time += wait_map[actor_id]
    redis_connection.set("average delay",total_wait_time/len(wait_map))
    print(total_wait_time/len(wait_map))
    