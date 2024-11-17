import carla 
import redis
import math 
import time
client = carla.Client('localhost', 2000) 
world = client.get_world()
my_map = world.get_map()
# 连接到Redis服务器（默认是本地host和6379端口）  
redis_connection = redis.Redis(host='localhost', port=6379, db=0)
#计算两个carla.Location对象之间的欧几里得距离
def calculate_distance(location1, location2):  
    dx = location1.x - location2.x  
    dy = location1.y - location2.y  
    dz = location1.z - location2.z
    return math.sqrt(dx**2 + dy**2 + dz **2)  
while True:
    time.sleep(0.5) 
    road_map = {}
    actors = world.get_actors() 
    for actor in actors:
        type_id = actor.type_id
        if type_id.startswith("vehicle"):
            #获取车辆所在航点对象
            way_point = my_map.get_waypoint(actor.get_location())
            #获取车辆所在道路
            road_id = way_point.road_id
            traffic_light = actor.get_traffic_light()
            if traffic_light:
                traffic_light_state = traffic_light.get_state()
                #判断红绿灯是否为红灯的状态
                if traffic_light_state == carla.TrafficLightState.Red:
                    if road_map.get(str(road_id), 'None') == 'None':
                        road_map[str(road_id)] = 0
                    #获取某个交通灯下汽车目前已知的排队长度
                    length = road_map[str(road_id)]
                    #获取交通灯停车线位置
                    trigger_volume = traffic_light.trigger_volume
                    #获取某个车辆距离交通灯停止线的距离
                    distance = calculate_distance(trigger_volume.location,actor.get_location())
                    if distance >=length:
                        road_map[str(road_id)] = distance
    redis_connection.set("road_congestion_mileage", json.dumps(road_map))