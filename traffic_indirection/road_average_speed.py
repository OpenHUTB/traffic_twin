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
    #统计每条道路中的车辆数目
    vehicle_count = {}
    #统计每条道路中的车辆速度总和
    speed_sum = {}
    for actor in actors:
        type_id = actor.type_id
        if type_id.startswith("vehicle"):
            #获取车辆所在航点对象
            way_point = my_map.get_waypoint(actor.get_location())
            #获取车辆所在道路
            road_id = way_point.road_id
            if vehicle_count.get(str(road_id), 'None') == 'None':
                vehicle_count[str(road_id)] = 0
            vehicle_count[str(road_id)] += 1 
            vector3D = actor.get_velocity()
            x = vector3D.x
            y = vector3D.y
            z = vector3D.z
            vector = math.sqrt(x**2 + y**2+z**2)
            if speed_sum.get(str(road_id), 'None') == 'None':
                speed_sum[str(road_id)] = 0
            speed_sum[str(road_id)] += vector
    road_average_map = {}
    for road_id in vehicle_count:
        #获取每条道路上的车辆数目
        num = vehicle_count[road_id]
        vector = speed_sum[road_id]
        road_average_speed = 0
        if num>0:
            road_average_speed = vector/num
        road_average_map[road_id]=road_average_speed
    #将数据实时存储到redis中
    redis_connection.set('road_average_speed',json.dumps(road_average_map))