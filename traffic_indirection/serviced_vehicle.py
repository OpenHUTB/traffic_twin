import carla 
import redis
import time
import json
client = carla.Client('localhost', 2000) 
world = client.get_world()
my_map = world.get_map()
#统计通过优化路口的车辆
count = 0
# 连接到Redis服务器（默认是本地host和6379端口）  
redis_connection = redis.Redis(host='localhost', port=6379, db=0) 
road_map = {}
while True:
    time.sleep(0.5)
    optimized_list = redis_connection.get("optimized_list")
    if optimized_list:
        optimized_list = json.loads(optimized_list)
        print(f"List elements: {optimized_list}")
        if optimized_list!=[]:
            actors = world.get_actors() 
            for actor in actors:
                type_id = actor.type_id
                if type_id.startswith("vehicle"):
                    #获取车辆所在航点对象
                    way_point = my_map.get_waypoint(actor.get_location())
                    #获取车辆所在道路
                    road_id = way_point.road_id
                    if road_map.get(str(actor.id),'None')== 'None':
                        traffic_light = actor.get_traffic_light()
                        if traffic_light and traffic_light.id in optimized_list:
                            #记录车辆所在的道路id
                            road_map[str(actor.id)] = road_id
                    elif road_map[str(actor.id)] != road_id:
                        count+=1;
                        road_map.pop(str(actor.id))
    redis_connection.set("serviced_vehicle", count)