#!/usr/bin/env python

import os
import sys

try:
    sys.path.append(os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))), 'official'))
    sys.path.append(os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))), 'utils'))
except IndexError:
    pass

import copy
import carla
import math
import numpy as np
import interpolate as itp
import carla_utils as ca_u
from enum import Enum
from collections import deque
from basic_agent import BasicAgent
from global_route_planner import GlobalRoutePlanner

import matplotlib.pyplot as plt
import time

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class Xagent(BasicAgent):
    def __init__(self, env, model, dt=0.1) -> None:
        '''
        vehicle: carla
        model: kinematic/dynamic model
        '''
        self.custom_route = None   # 用于存放自定义轨迹 Nx2

        self._env = env
        self._vehicle = env.ego_vehicle
        self._model = model
        
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        self._base_min_distance = 2.0
        self._waypoints_queue = deque(maxlen=100000)
        self._d_dist = 0.4
        self._sample_resolution = 2.0
        # state of lane_change
        self._a_opt = np.array([0.0]*self._model.horizon)
        self._delta_opt = np.array([0.0]*self._model.horizon)
        self._dt = dt

        self._next_states = None
        self._last_traffic_light = None
        self._last_traffic_waypoint = None

        self._model.solver_basis(Q=np.diag([10, 10, 10, 1.5, 0.1]), Rd=np.diag([1.0, 1000.0]))
        self.Q_origin = copy.deepcopy(self._model.Q)
        self._log_data = []
        self._simu_time = 0
        
        self._global_planner = GlobalRoutePlanner(self._map, self._sample_resolution)
        
        self.dist_move = 0.2
        self.dist_step = 1.5

    # def plan_route(self, start_location, end_location):  # 将所有的路径点的信息录入到_waypoints_queue中
    #     self._route = self.trace_route(start_location.location, end_location.location)
    #     for i in self._route:
    #         self._waypoints_queue.append(i)

    def plan_route(self, start_location=None, end_location=None, custom_route=None):
        """
        如果提供 custom_route，则使用自定义轨迹，否则使用全局规划器
        """
        if custom_route is not None:
            # 自定义轨迹：Nx2 数组
            self.custom_route = np.array(custom_route)
            self._waypoints_queue.clear()
            # 把自定义轨迹加入_waypoints_queue
            for pt in self.custom_route:
                # 这里需要生成一个“waypoint-like”对象或tuple，保持原代码逻辑兼容
                # 简单方式是用 tuple((x, y), 0) 代替原本 (waypoint, road_option)
                self._waypoints_queue.append((pt, 0))
        else:
            # 原来的全局规划器
            self._route = self.trace_route(start_location.location, end_location.location)
            self._waypoints_queue.clear()
            for i in self._route:
                self._waypoints_queue.append(i)

    # def set_start_end_transforms(self, start_idx, end_idx):
    #     """
    #     Set the start and end transform using given indices
    #     :param start_idx: start index of the spawn point
    #     :param end_idx: end index of the spawn point
    #     """
    #     spawn_points = self._map.get_spawn_points()  # 获取所有的spawn points
    #     if start_idx < len(spawn_points) and end_idx < len(spawn_points):
    #         self._start_transform = spawn_points[start_idx]  # 设置起始点的transform
    #         self._end_transform = spawn_points[end_idx]      # 设置终止点的transform
    #     else:
    #         raise IndexError("Start or end index out of bounds!")
    def set_start_end_transforms(self, custom_trajectory=None):
        """
        Set start and end transforms using custom trajectory (list of [x, y]).
        If custom_trajectory is None, fallback to spawn points (old behavior).
        """
        if custom_trajectory is not None:
            start_loc = custom_trajectory[0]
            end_loc = custom_trajectory[-1]

            # 使用 carla.Transform 生成 Transform
            self._start_transform = carla.Transform(carla.Location(x=start_loc[0], y=start_loc[1], z=0.5),
                                                    carla.Rotation())
            self._end_transform = carla.Transform(carla.Location(x=end_loc[0], y=end_loc[1], z=0.5),
                                                  carla.Rotation())
        else:
            # fallback to original spawn point method
            spawn_points = self._map.get_spawn_points()
            start_idx, end_idx = 0, len(spawn_points) - 1
            self._start_transform = spawn_points[start_idx]
            self._end_transform = spawn_points[end_idx]

    def calc_ref_trajectory_in_T_step(self, node, ref_path, sp):
        """
        calc referent trajectory in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param node: current information
        :param ref_path: reference path: [x, y, yaw]
        :param sp: speed profile (designed speed strategy)
        :return: reference trajectory
        """
        T = self._model.horizon
        z_ref = np.zeros((4, T + 1))
        length = ref_path.length
        ind, _ = ref_path.nearest_index(node)

        z_ref[0, 0] = ref_path.cx[ind]
        z_ref[1, 0] = ref_path.cy[ind]
        z_ref[2, 0] = sp[ind]
        z_ref[3, 0] = ref_path.cyaw[ind]

        dist_move = copy.copy(self.dist_move)

        for i in range(1, T + 1):
            dist_move += self.dist_step *abs(self._model.get_v()) * self._dt
            ind_move = int(round(dist_move / self._d_dist))
            index = min(ind + ind_move, length - 1)

            z_ref[0, i] = ref_path.cx[index]
            z_ref[1, i] = ref_path.cy[index]
            z_ref[2, i] = sp[index]
            z_ref[3, i] = ref_path.cyaw[index]

        return z_ref, ind

    def rotate(self, x, y, theta, ratio=1.75):
        return np.array([(x * np.cos(theta) - y * np.sin(theta)) * ratio, (x * np.sin(theta) + y * np.cos(theta)) * ratio])
    
    def lat_dis_wp_ev(self, wp, ev):
        '''
        calculate the lateral distance between the waypoint and the ego vehicle
        '''
        wp_loc = np.array([wp.transform.location.x, wp.transform.location.y])
        ev_loc = np.array([ev.get_location().x, ev.get_location().y])
        wp_yaw = wp.transform.rotation.yaw
        wp_loc = self.rotate(wp_loc[0], wp_loc[1], np.deg2rad(wp_yaw))
        ev_loc = self.rotate(ev_loc[0], ev_loc[1], np.deg2rad(wp_yaw))
        return np.abs(wp_loc[1] - ev_loc[1])
    
    def run_step(self,lv=None):
        self._simu_time += self._dt
        state, height = self._model.get_state_carla() # return the car's state and height
        current_state = np.array(ca_u.carla_vector_to_rh_vector(state[0:2], state[2], state[3:]))  # 把坐标轴转换为右手坐标系，因为右手的坐标系是的z轴是向上的
        # print("current_state:",current_state)
        # - Purge the queue of obsolete waypoints
        veh_location = self._vehicle.get_location()
        vehicle_speed = self._model.get_v()
        self._min_distance = self._base_min_distance + 0.5 * vehicle_speed  # 车辆和某个路径点的关系如果小于这个min_distance的话，那么这个路径点就会被删除

        # - Get waypoints
        if len(self._waypoints_queue) == 0:
            raise Exception("No waypoints to follow")
        else:
            # carla_wp, _ = np.array(self._waypoints_queue).T
            if self.custom_route is not None:
                # 自定义轨迹已经存入_waypoints_queue
                carla_wp = np.array([wp for wp, _ in self._waypoints_queue])
            else:
                # 原逻辑
                carla_wp, _ = np.array(self._waypoints_queue).T

            waypoints = []
            v = math.sqrt(current_state[3]**2+current_state[4]**2)
            waypoints.append(
                [current_state[0], current_state[1], v, current_state[2]])
            cnt = 0

            # delete the same waypoints to solve the problem of spline interpolation(NaN)
            '''
            对于一条路径来说30个点可以比较代表性的描绘出这样的路径的特点，因为这里面我们是吧路径上所有的点用右边坐标系的方法进行保存，
            所以如果要计算所有的路径点的话会比较耗时，并且在使用MPC进行路径跟踪的时候，我们是控制该车辆在未来的horizon的长度的预测路径上的位置，所以
            我们只需要保存未来horizon长度的路径点即可。
            '''
            last_state = None
            for wp in carla_wp:
                if cnt > 30:
                    break
                cnt += 1
                t = wp.transform
                ref_state = ca_u.carla_vector_to_rh_vector(
                    [t.location.x, t.location.y], t.rotation.yaw)
                if last_state is not None:
                    if np.sqrt(ref_state[0]**2+ref_state[1]**2) - last_state < 0.005: # 如果两个参考路径点之间的距离太小会使得在就算三次样条插值的时候出现NAN的情况
                        continue
                waypoints.append([ref_state[0], ref_state[1],
                                 self._model.target_v, ref_state[2]])
                last_state = np.sqrt(ref_state[0]**2+ref_state[1]**2)

            waypoints = np.array(waypoints).T

        num_waypoint_removed = 0

        for waypoint, _ in self._waypoints_queue:
            if len(self._waypoints_queue) - num_waypoint_removed == 1:   # 最后一个路径点的位置
                min_distance = 1  # Don't remove the last waypoint until very close by
            else:
                min_distance = self._min_distance

            if veh_location.distance(waypoint.transform.location) < min_distance:  #当前车辆点与这个waypoint的距离小于min_distance的话，那么这个waypoint就会被删除
                num_waypoint_removed += 1
            else:
                break

        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._waypoints_queue.popleft()  # 队首元素进行pop

        # - Interplote the waypoints
        # t_0 = time.time()
        cx, cy, cyaw, ck, s = itp.calc_spline_course_carla(
            waypoints[0], waypoints[1], waypoints[3][0], ds=self._d_dist)
        sp = itp.calc_speed_profile(cx, cy, cyaw, self._model.target_v)
        # print('Interpolation Time cost: ', time.time()-t_0)

        ref_path = itp.PATH(cx, cy, cyaw, ck)
        z_ref, target_ind = self.calc_ref_trajectory_in_T_step(
            [current_state[0], current_state[1], v, current_state[2]], ref_path, sp)
        ref_traj = np.array([z_ref[0], z_ref[1], z_ref[3], z_ref[2], [
                            0]*len(z_ref[0]), [0]*len(z_ref[0])])[:, :self._model.horizon]

        if self._next_states is None:
            self._next_states = np.zeros(
                (self._model.n_states, self._model.horizon+1)).T
        
        cur_v = self._model.get_v()
        self._next_states[:,3] = cur_v
        current_state[3:] = self._next_states[0][3:]
        u0 = np.array([self._a_opt, self._delta_opt]).reshape(-1, 2).T

        if self._next_states is None:
            self._next_states = np.zeros(
                (self._model.n_states, self._model.horizon+1)).T
        apf_obs = apf_nc_road = apf_c_road = 0
        self._model.solver_add_cost()
        self._model.solver_add_bounds()  #  u00 = u0[:,0])
        tick = time.time()
        state = self._model.solve_MPC(
            ref_traj.T, current_state, self._next_states, u0)
        time_2 = time.time()-tick
        
        ca_u.draw_planned_trj(self._world, ref_traj[:2, :].T, height+0.5)
        ca_u.draw_planned_trj(self._world, state[2][:, :2], height+0.5, color=(0, 233, 222))

        self._next_states = state[2]

        next_state = state[2][1]
        self._a_opt = state[0]
        self._delta_opt = state[1]
        
        next_state = self._model.predict(current_state, (self._a_opt[0], self._delta_opt[0]))
        self._model.set_state(next_state)

        dist_error = math.hypot(
            next_state[0] - ref_traj[0, 1], next_state[1] - ref_traj[1, 1] - self.dist_move)
        yaw_error = abs(next_state[2] - ref_traj[2, 1])
        vel_error = abs(state[2][0][3] - ref_traj[3, 0])
        acc = self._a_opt[0] # carla_acc # self._a_opt[0]
        steer = self._delta_opt[0] # carla_steer # self._delta_opt[0]
        
        cost_time = state[-1]
        return self._a_opt[0], self._delta_opt[0], (next_state, height+0.05)

    def trace_route(self, start_location, end_location):
        return self._global_planner.trace_route(start_location, end_location)