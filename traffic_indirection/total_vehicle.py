import carla 
import redis
import time
client = carla.Client('localhost', 2000) 
world = client.get_world()
# 连接到Redis服务器（默认是本地host和6379端口）  
redis_connection = redis.Redis(host='localhost', port=6379, db=0)
while True:
    time.sleep(0.5)
    count = 0    
    actors = world.get_actors() 
    for actor in actors:
        type_id = actor.type_id
        if type_id.startswith("vehicle"):
            count += 1
    redis_connection.set("total_vehicle", count)