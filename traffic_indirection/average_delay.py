import carla 
import redis 
import time
from datetime import datetime
client = carla.Client('localhost', 2000) 
world = client.get_world()
my_map = world.get_map()
# 连接到Redis服务器（默认是本地host和6379端口）  
redis_connection = redis.Redis(host='localhost', port=6379, db=0) 
road_map = {}
start_map = {}
while True:
    time.sleep(0.5)
    wait_map = {}
    actors = world.get_actors() 
    for actor in actors:
        type_id = actor.type_id
        if type_id.startswith("vehicle"):
            wait_map[str(actor.id)] = 0 
            #获取车辆所在航点对象
            way_point = my_map.get_waypoint(actor.get_location())
            #获取车辆所在道路
            road_id = way_point.road_id
            if road_map.get(str(actor.id), 'None') == 'None':
                traffic_light = actor.get_traffic_light()
                #判断红绿灯是否为红灯的状态
                if traffic_light and traffic_light.get_state() == carla.TrafficLightState.Red:
                    #记录车辆所在的道路id
                    road_map[str(actor.id)] = road_id
                    #获取当前日期和时间，然后转换为时间戳  
                    now = datetime.now() 
                    start_map[str(actor.id)] = now.timestamp()
            elif road_map.get(str(actor.id), 'None')!=road_id:
                #获取当前日期和时间，然后转换为时间戳  
                now = datetime.now()
                wait_map[str(actor.id)] = now.timestamp()-start_map[str(actor.id)]
                road_map.pop(str(actor.id))
                start_map.pop(str(actor.id))
    total_wait_time = 0
    for actor_id in wait_map:
        total_wait_time += wait_map[actor_id]
    redis_connection.set("average_delay",total_wait_time/len(wait_map))