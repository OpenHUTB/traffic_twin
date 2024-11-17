from pyproj import Proj, transform
import json
import argparse
'''
前言：导入pyproj包，指令为：pip install pyproj
功能：将给定的经纬度坐标装换成指定平面坐标系中的坐标
输入参数：
    data: 1、接收到的数据
          2、其格式为："""[{"经度": "113.6534423", "维度": "27.6844241"}, {"x": "113.6534423", "y": "27.6844241"}}"""
          3、中间的逗号为英文格式
    lo: 指定为平面坐标系原点的点的经度坐标
    la: 指定为平面坐标系原点的点的维度坐标
返回值：
    ll_to_xy_1
        json对象数组的字符串形式
        """[{"经度": "113.6534423", "维度": "27.6844241"}, {"x": "113.6534423", "y": "27.6844241"}}"""
    ll_to_xy_2
        两个数组
        x: 指定平面坐标系中的x坐标，单位米/m
        y: 指定平面坐标系中的y坐标，单位米/m
其他说明：墨卡托投影不适合在高纬度地区使用
'''


# 坐标转换，放回值为字符串形式的json对象数组
def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--data_str',
        type=str,
        default="",
        help='original location of vehicle')
    argparser.add_argument(
        '--la',
        type=float,
        default=0.0,
        help='Longitude coordinates of the origin of the coordinate system')
    argparser.add_argument(
        '--lo',
        type=float,
        default=0.0,
        help='Latitude coordinates of the origin of the coordinate system')
    args = argparser.parse_args()
    point_xy = []
    wgs84 = Proj(init='epsg:4326')  # 定义WGS84坐标系统（地理坐标系统）
    merc = Proj(init='epsg:3857')   # 定义墨卡托投影坐标系统
    x_origin, y_origin = transform(wgs84, merc, args.lo, args.la)  # 从WGS84转换到墨卡托投影，并将该点指定为平面坐标的原点
    #print(args.lo)
    #print(args.data_str)
    data_json = json.loads(args.data_str)
    #print(data_json[0])

    for point_ll in data_json:
        point = {}
        longitude = float(point_ll['x'])  # 转换成浮点型数据
        latitude = float(point_ll['y'])  # 转换成浮点型数据
        x_temp, y_temp = transform(wgs84, merc, longitude, latitude)  # 从WGS84转换到墨卡托投影
        point["x"] = x_temp - x_origin
        point["y"] = y_temp - y_origin
        point_xy.append(point)
    print("transfer result:"+json.dumps(point_xy))


if __name__ == '__main__':
    # test_data = '[{"x":"123","y":"456"},{"x":"123","y":"456"}]'
    # aaa = json.loads(test_data)
    # bbb = type(aaa)
    # print(aaa)
    # print(bbb)
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')