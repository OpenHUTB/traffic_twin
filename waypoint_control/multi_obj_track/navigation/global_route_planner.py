import math
import numpy as np
import networkx as nx

import carla
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import vector


class GlobalRoutePlanner(object):
    """
    This class provides a very high level route plan.
    """

    def __init__(self, wmap, sampling_resolution):
        self._sampling_resolution = sampling_resolution
        self._wmap = wmap
        self._topology = None
        self._graph = None
        self._id_map = None
        self._road_id_to_edge = None
        self._intersection_end_node = -1
        self._previous_decision = RoadOption.VOID

        # Build the graph
        self._build_topology()
        self._build_graph()
        self._find_loose_ends()
        # self._lane_change_link()

    def _trace_route(self, origin, destination):
        """
        This method returns list of (carla.Waypoint, RoadOption)
        from origin to destination
        """
        # 初始化一个空列表 route_trace，用于存储最终的路径结果。
        route_trace = []
        # 调用 self._path_search() 方法，在图结构中搜索从 origin 到 destination 的最短路径，
        # 并返回一条由节点（路段）组成的 route 列表。
        route = self._path_search(origin, destination)
        current_waypoint = self._wmap.get_waypoint(origin)
        destination_waypoint = self._wmap.get_waypoint(destination)

        # 路径遍历及处理
        for i in range(len(route) - 1):
            # 调用 _turn_decision() 方法计算当前节点到下一个节点之间的道路选择（RoadOption）。
            road_option = self._turn_decision(i, route)
            # 获取当前节点 route[i] 与下一个节点 route[i+1] 之间的边（edge），表示当前道路的详细信息。
            edge = self._graph.edges[route[i], route[i + 1]]
            # 初始化一个空的 path 列表，用于存储当前边上的航点序列。
            path = []

            # 判断当前边的类型，若不是 LANEFOLLOW（即简单跟随车道）或 VOID（无效路径），则进行特殊处理。
            if edge['type'] != RoadOption.LANEFOLLOW and edge['type'] != RoadOption.VOID:
                # 将当前的 current_waypoint 和对应的 road_option 添加到 route_trace 路径中。
                route_trace.append(current_waypoint)
                # 获取当前边的出口航点 exit_wp。
                exit_wp = edge['exit_waypoint']
                # 获取当前出口航点 exit_wp 所对应的边（n1, n2），即路网图中的两个节点。
                n1, n2 = self._road_id_to_edge[exit_wp.road_id][exit_wp.section_id][exit_wp.lane_id]
                # 获取当前边的下一个边 next_edge，用于继续路径计算。
                next_edge = self._graph.edges[n1, n2]
                # 判断下一条边是否有可用的路径。
                if next_edge['path']:
                    # 找到当前航点 current_waypoint 在下一条边路径中的最接近的点的索引位置。
                    closest_index = self._find_closest_in_list(current_waypoint, next_edge['path'])
                    # 确保索引不超过路径长度，同时向前跳跃5个航点，以加速路径计算。
                    closest_index = min(len(next_edge['path']) - 1, closest_index + 5)
                    # 更新 current_waypoint 为下一条边的航点。
                    current_waypoint = next_edge['path'][closest_index]
                else:
                    current_waypoint = next_edge['exit_waypoint']
                # 再次将更新后的 current_waypoint 和对应的 road_option 添加到路径中。
                # route_trace.append((current_waypoint, road_option.LANEFOLLOW))
                route_trace.append(current_waypoint)

            else:
                # 将当前边的入口航点、路径中的航点以及出口航点一起添加到 path 列表中。
                path = path + [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
                # 找到当前航点在路径中的最近索引位置。
                closest_index = self._find_closest_in_list(current_waypoint, path)
                for waypoint in path[closest_index:]:
                    current_waypoint = waypoint

                    route_trace.append(current_waypoint)
                    if len(route) - i <= 2 and waypoint.transform.location.distance(
                            destination) < 2 * self._sampling_resolution:
                        break
                    elif len(
                            route) - i <= 2 and current_waypoint.road_id == destination_waypoint.road_id and current_waypoint.section_id == destination_waypoint.section_id and current_waypoint.lane_id == destination_waypoint.lane_id:
                        destination_index = self._find_closest_in_list(destination_waypoint, path)
                        if closest_index > destination_index:
                            break
        #  返回一个包含 (carla.Waypoint, RoadOption) 元组的列表，表示沿途的航点及其对应的道路选择。
        return route_trace

    def _build_topology(self):
        """
         CARLA 服务器中获取道路拓扑信息，并处理成包含详细道路段信息的结构。
         最终生成的拓扑结构 self._topology 是一个列表，其中每个元素代表
         一个路段信息，包含了起点、终点、路径及其位置。

        self._topology = [
                {
                    'entry': wp1,                    # 起点 Waypoint 对象
                    'exit': wp2,                     # 终点 Waypoint 对象
                    'entryxyz': (x1, y1, z1),        # 起点坐标
                    'exitxyz': (x2, y2, z2),         # 终点坐标
                    'path': [w1, w2, ..., wn]        # 从起点到终点的航点序列
                },
                ...
        ]
        """
        self._topology = []
        # 遍历每个路段，处理位置信息
        # _wmap.get_topology(), map.get_topology()
        """
            get_topology(self) 方法返回了一个表示道路拓扑结构的最小图（图中的节点和边），
            其中的节点表示道路的起点或终点。返回的结果是一个列表，其中的每个元素是一个包含两个
            carla.Waypoint 对象的元组 (w_start, w_end)。
            [(w0, w1), (w0, w2), (w1, w3), (w2, w3), (w0, w4)]
            道路段之间可能有间隔
        """
        i = 0
        for segment in self._wmap.get_topology():
            # 提取起点和终点坐标
            wp1, wp2 = segment[0], segment[1]
            l1, l2 = wp1.transform.location, wp2.transform.location
            # 位置取整
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1, l2
            # 初始化路段字典
            seg_dict = dict()
            seg_dict['entry'], seg_dict['exit'] = wp1, wp2
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
            seg_dict['path'] = []
            # 定义 self._sampling_resolution 表示采样分辨率，即每隔多少距离采集一个航点。
            endloc = wp2.transform.location
            # 判断 wp1 到 wp2 的距离是否超过分辨率 self._sampling_resolution

            if wp1.transform.location.distance(endloc) > self._sampling_resolution:
                """
                next(self, distance)
                Returns a list of waypoints at a certain approximate distance from the current one. 
                It takes into account the road and its possible deviations without performing any lane change and 
                returns one waypoint per option. The list may be empty if the lane is not connected to any other at the specified distance.
                Parameters:
                distance (float - meters) - The approximate distance where to get the next waypoints.
                Return: list(carla.Waypoint)
                """
                max_iterations = 100  # 限制最大采样次数
                iteration = 0

                # 采样取第一个点
                w = wp1.next(self._sampling_resolution)[0]
                while w.transform.location.distance(endloc) > self._sampling_resolution:
                    seg_dict['path'].append(w)
                    next_ws = w.next(self._sampling_resolution)
                    # if len(next_ws) == 0:
                    if len(next_ws) == 0 or iteration > max_iterations:
                        break
                    w = next_ws[0]
                    iteration += 1
            else:
                next_wps = wp1.next(self._sampling_resolution)
                if len(next_wps) == 0:
                    continue
                seg_dict['path'].append(next_wps[0])
            self._topology.append(seg_dict)


    def _build_graph(self):
        """
        This function builds a networkx graph representation of topology, creating several class attributes:
        - graph (networkx.DiGraph): networkx graph representing the world map, with:
            Node properties:
                vertex: (x,y,z) position in world map
            Edge properties:
                entry_vector: unit vector along tangent at entry point
                exit_vector: unit vector along tangent at exit point
                net_vector: unit vector of the chord from entry to exit
                intersection: boolean indicating if the edge belongs to an  intersection
        - id_map (dictionary): mapping from (x,y,z) to node id
        - road_id_to_edge (dictionary): map from road id to edge in the graph
        """
        # 创建一个空的 networkx.DiGraph 对象，用于存储地图拓扑的有向图。
        self._graph = nx.DiGraph()
        # 用于将每个 (x, y, z) 坐标与图中的节点 ID 进行映射，结构为 {(x, y, z): node_id, ...}
        self._id_map = dict()  # Map with structure {(x,y,z): id, ... }
        # 用于按 road_id、section_id 和 lane_id 组织和查找边信息
        self._road_id_to_edge = dict()  # Map with structure {road_id: {lane_id: edge, ... }, ... }
        # 遍历拓扑中的每个路段 segment
        for segment in self._topology:
            entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
            path = segment['path']
            entry_wp, exit_wp = segment['entry'], segment['exit']
            intersection = entry_wp.is_junction
            road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id
            # 为每个路段添加节点
            for vertex in entry_xyz, exit_xyz:
                # 如果该坐标不在 self._id_map 中，将其添加到 self._id_map，并分配一个唯一的 node_id
                if vertex not in self._id_map:
                    new_id = len(self._id_map)
                    self._id_map[vertex] = new_id
                    # 使用 self._graph.add_node() 将新节点加入到 self._graph 图中。
                    self._graph.add_node(new_id, vertex=vertex)
            # 标记起点和终点的节点 ID
            n1 = self._id_map[entry_xyz]
            n2 = self._id_map[exit_xyz]
            # 构建 road_id 到 edge 的映射
            # 检查当前的 road_id、section_id、lane_id 是否在 self._road_id_to_edge 中
            if road_id not in self._road_id_to_edge:
                # 如果不存在，则创建相应的字典层级，并将 lane_id 与其对应的节点 ID 对（(n1, n2)）关联起来
                self._road_id_to_edge[road_id] = dict()
            if section_id not in self._road_id_to_edge[road_id]:
                self._road_id_to_edge[road_id][section_id] = dict()
            self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
            # 计算路径方向向量
            # 是 entry 和 exit 航点的朝向向量，用于表示路径的切线方向，存储在 entry_vector 和 exit_vector 属性中。
            entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()
            # net_vector 表示从 entry 到 exit 的方向向量
            # 添加边到图
            self._graph.add_edge(
                n1, n2,
                length=len(path) + 1, path=path,
                entry_waypoint=entry_wp, exit_waypoint=exit_wp,
                entry_vector=np.array(
                    [entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]),
                exit_vector=np.array(
                    [exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z]),
                net_vector=vector(entry_wp.transform.location, exit_wp.transform.location),
                intersection=intersection, type=RoadOption.LANEFOLLOW)

    def _find_loose_ends(self):
        """
            目的是找到道路拓扑中那些没有连接的“末端”（loose ends）道路段，并将它们加入内部图结构 _graph 中。
            此方法会处理那些没有相邻路段直接连接的道路段（如死胡同或孤立的道路末端），并将其存储在图中，以便在路
            径规划时能识别这些末端。
            这个方法通过添加“虚拟”节点和边，确保图结构中包含了道路拓扑中未连接的末端，从而完整表示整个路网拓扑，
            便于路径规划算法识别所有道路段，包括死胡同和孤立末端。
        """
        count_loose_ends = 0
        hop_resolution = self._sampling_resolution
        for segment in self._topology:
            end_wp = segment['exit']
            exit_xyz = segment['exitxyz']
            road_id, section_id, lane_id = end_wp.road_id, end_wp.section_id, end_wp.lane_id
            if road_id in self._road_id_to_edge \
                    and section_id in self._road_id_to_edge[road_id] \
                    and lane_id in self._road_id_to_edge[road_id][section_id]:
                pass
            else:
                count_loose_ends += 1
                if road_id not in self._road_id_to_edge:
                    self._road_id_to_edge[road_id] = dict()
                if section_id not in self._road_id_to_edge[road_id]:
                    self._road_id_to_edge[road_id][section_id] = dict()
                n1 = self._id_map[exit_xyz]
                n2 = -1 * count_loose_ends
                self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
                next_wp = end_wp.next(hop_resolution)
                path = []
                while next_wp is not None and next_wp \
                        and next_wp[0].road_id == road_id \
                        and next_wp[0].section_id == section_id \
                        and next_wp[0].lane_id == lane_id:
                    path.append(next_wp[0])
                    next_wp = next_wp[0].next(hop_resolution)
                if path:
                    n2_xyz = (path[-1].transform.location.x,
                              path[-1].transform.location.y,
                              path[-1].transform.location.z)
                    self._graph.add_node(n2, vertex=n2_xyz)
                    self._graph.add_edge(
                        n1, n2,
                        length=len(path) + 1, path=path,
                        entry_waypoint=end_wp, exit_waypoint=path[-1],
                        entry_vector=None, exit_vector=None, net_vector=None,
                        intersection=end_wp.is_junction, type=RoadOption.LANEFOLLOW)

    def _lane_change_link(self):
        """
        目的是在拓扑图中添加“零成本”边，以表示在非交叉路口的道路上允许的变道。方法遍历每个道路段的路径，
        并在符合变道条件的位置建立变道链接，从而在导航和路径规划中支持车辆左右变道的选项。
        """
        for segment in self._topology:
            # 初始化左右变道标志，每次处理一个道路段（segment）时，left_found 和 right_found 被初始化为 False，
            # 表示当前段的路径还没有找到左右变道的连接
            left_found, right_found = False, False
            # 遍历 segment 的 path，其中 path 是该段道路的航点列表。每个航点 waypoint 代表路径中的一个位置。
            for waypoint in segment['path']:
                # 检查 segment['entry'] 是否属于交叉路口。如果当前段是交叉路口，则跳过变道逻辑，因为通常在交叉路口不执行变道操作。
                if not segment['entry'].is_junction:
                    next_waypoint, next_road_option, next_segment = None, None, None
                    # 代码检查当前 waypoint 的右侧车道是否允许变道
                    if waypoint.right_lane_marking and waypoint.right_lane_marking.lane_change & carla.LaneChange.Right and not right_found:
                        # 获取右侧航点
                        next_waypoint = waypoint.get_right_lane()
                        # 验证右侧车道有效性
                        if next_waypoint is not None \
                                and next_waypoint.lane_type == carla.LaneType.Driving \
                                and waypoint.road_id == next_waypoint.road_id:
                            # 在图结构中添加右侧变道的边
                            next_road_option = RoadOption.CHANGELANERIGHT
                            # 使用 _localize 方法查找右侧车道的 next_segment
                            next_segment = self._localize(next_waypoint.transform.location)
                            if next_segment is not None:
                                self._graph.add_edge(
                                    # next_segment[0]表示边的第一个点的id
                                    self._id_map[segment['entryxyz']], next_segment[0], entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint, intersection=False, exit_vector=None,
                                    path=[], length=0, type=next_road_option, change_waypoint=next_waypoint)
                                right_found = True
                    # 检查左侧车道是否允许变道，并与右侧变道逻辑相似
                    if waypoint.left_lane_marking and waypoint.left_lane_marking.lane_change & carla.LaneChange.Left and not left_found:
                        next_waypoint = waypoint.get_left_lane()
                        if next_waypoint is not None \
                                and next_waypoint.lane_type == carla.LaneType.Driving \
                                and waypoint.road_id == next_waypoint.road_id:
                            next_road_option = RoadOption.CHANGELANELEFT
                            next_segment = self._localize(next_waypoint.transform.location)
                            if next_segment is not None:
                                self._graph.add_edge(
                                    self._id_map[segment['entryxyz']], next_segment[0], entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint, intersection=False, exit_vector=None,
                                    path=[], length=0, type=next_road_option, change_waypoint=next_waypoint)
                                left_found = True
                if left_found and right_found:
                    break

    def _localize(self, location):
        """
        This function finds the road segment that a given location
        is part of, returning the edge it belongs to
        """
        waypoint = self._wmap.get_waypoint(location)
        edge = None
        try:
            edge = self._road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
        except KeyError:
            pass
        return edge

    def _distance_heuristic(self, n1, n2):
        """
        Distance heuristic calculator for path searching
        in self._graph
        """
        l1 = np.array(self._graph.nodes[n1]['vertex'])
        l2 = np.array(self._graph.nodes[n2]['vertex'])

        # 获取 l1 和 l2 对应的车道 ID
        lane_id1 = self.get_lane_id_from_coordinates(l1)
        lane_id2 = self.get_lane_id_from_coordinates(l2)

        # 欧几里得距离。
        distance = np.linalg.norm(l1 - l2)

        # 增加车道偏移惩罚项
        if lane_id1 is not None and lane_id2 is not None:
            lane_diff = abs(lane_id1 - lane_id2)

            # 设定惩罚系数
            lane_penalty = 20  # 可根据需求调整惩罚值
            distance += lane_penalty * lane_diff

        return distance

    def get_lane_id_from_coordinates(self, coordinates):
        """
        Convert coordinates to a CARLA Waypoint and retrieve the lane ID.

        :param world: CARLA world object
        :param coordinates: Tuple (x, y, z) representing the coordinates
        :return: The lane ID of the waypoint at the coordinates, or None if not found
        """
        # 将坐标转换为 CARLA 的 Location 对象
        location = carla.Location(x=coordinates[0], y=coordinates[1], z=coordinates[2])

        # 获取此位置的 Waypoint 对象
        waypoint = self._wmap.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

        # 返回车道 ID
        if waypoint is not None:
            return waypoint.lane_id
        else:
            return None

    def _path_search(self, origin, destination):
        """
        This function finds the shortest path connecting origin and destination
        using A* search with distance heuristic.
        origin      :   carla.Location object of start position
        destination :   carla.Location object of end position
        return      :   path as list of node ids (as int) of the graph self._graph
        connecting origin and destination
        这个函数 _path_search 用于在 CARLA 仿真环境的道路网络中，使用 A* 搜索算法找到从起点 origin 到终点 destination 的最短路径。
        """
        # 返回的 start 和 end 分别是起点和终点的节点id。
        start, end = self._localize(origin), self._localize(destination)
        # start[0] 和 end[0] 是起点和终点的节点 ID，它们用于路径搜索
        route = nx.astar_path(
            self._graph, source=start[0], target=end[0],
            heuristic=self._distance_heuristic, weight='length')
        # end[1] 是终点的第二个节点信息，通常是图中的实际终点位置。
        route.append(end[1])
        # 最后返回 route，这是包含起点、终点以及两点之间节点的 ID 列表，表示从起点到终点的最短路径。
        return route

    def _successive_last_intersection_edge(self, index, route):
        """
        This method returns the last successive intersection edge
        from a starting index on the route.
        This helps moving past tiny intersection edges to calculate
        proper turn decisions.
        """

        last_intersection_edge = None
        last_node = None
        for node1, node2 in [(route[i], route[i + 1]) for i in range(index, len(route) - 1)]:
            candidate_edge = self._graph.edges[node1, node2]
            if node1 == route[index]:
                last_intersection_edge = candidate_edge
            if candidate_edge['type'] == RoadOption.LANEFOLLOW and candidate_edge['intersection']:
                last_intersection_edge = candidate_edge
                last_node = node2
            else:
                break

        return last_node, last_intersection_edge

    def _turn_decision(self, index, route, threshold=math.radians(35)):
        """
        This method returns the turn decision (RoadOption) for pair of edges
        around current index of route list
        """
        decision = None
        previous_node = route[index - 1]
        current_node = route[index]
        next_node = route[index + 1]
        next_edge = self._graph.edges[current_node, next_node]
        if index > 0:

            if self._previous_decision != RoadOption.VOID \
                    and self._intersection_end_node > 0 \
                    and self._intersection_end_node != previous_node \
                    and next_edge['type'] == RoadOption.LANEFOLLOW \
                    and next_edge['intersection']:
                decision = self._previous_decision
            else:
                self._intersection_end_node = -1
                current_edge = self._graph.edges[previous_node, current_node]
                calculate_turn = current_edge['type'] == RoadOption.LANEFOLLOW and not current_edge[
                    'intersection'] and next_edge['type'] == RoadOption.LANEFOLLOW and next_edge['intersection']
                if calculate_turn:
                    last_node, tail_edge = self._successive_last_intersection_edge(index, route)
                    self._intersection_end_node = last_node
                    if tail_edge is not None:
                        next_edge = tail_edge
                    cv, nv = current_edge['exit_vector'], next_edge['exit_vector']
                    if cv is None or nv is None:
                        return next_edge['type']
                    cross_list = []
                    for neighbor in self._graph.successors(current_node):
                        select_edge = self._graph.edges[current_node, neighbor]
                        if select_edge['type'] == RoadOption.LANEFOLLOW:
                            if neighbor != route[index + 1]:
                                sv = select_edge['net_vector']
                                cross_list.append(np.cross(cv, sv)[2])
                    next_cross = np.cross(cv, nv)[2]
                    deviation = math.acos(np.clip(
                        np.dot(cv, nv) / (np.linalg.norm(cv) * np.linalg.norm(nv)), -1.0, 1.0))
                    if not cross_list:
                        cross_list.append(0)
                    if deviation < threshold:
                        decision = RoadOption.STRAIGHT
                    elif cross_list and next_cross < min(cross_list):
                        decision = RoadOption.LEFT
                    elif cross_list and next_cross > max(cross_list):
                        decision = RoadOption.RIGHT
                    elif next_cross < 0:
                        decision = RoadOption.LEFT
                    elif next_cross > 0:
                        decision = RoadOption.RIGHT
                else:
                    decision = next_edge['type']

        else:
            decision = next_edge['type']

        self._previous_decision = decision
        return decision

    def _find_closest_in_list(self, current_waypoint, waypoint_list):
        min_distance = float('inf')
        closest_index = -1
        for i, waypoint in enumerate(waypoint_list):
            distance = waypoint.transform.location.distance(
                current_waypoint.transform.location)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index
