import time
import cvxpy
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from interpolate import calc_spline_course_carla  # 注意导入路径，或者直接粘到项目里
from new import adaptive_mpc_matrices
# State dimensions
NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 5   # horizon length

STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish threshold
TARGET_SPEED = 10.0 / 3.6  # target speed [m/s]
N_IND_SEARCH = 10  # Search index number
DT = 0.2  # time tick [s]
MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = 0  # minimum speed [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/s²]


class State:
    """Vehicle state representation"""
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None


def pi_2_pi(angle):
    """Normalize angle to the range [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def get_linear_model_matrix(v, phi, delta):
    """
    Get linearized model matrices
    Args:
        v: current velocity
        phi: current heading angle 
        delta: current steering angle
    Returns:
        A, B, C: State space model matrices
    """
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C


def update_state(state, a, delta):
    """
    Update vehicle state using kinematic bicycle model
    Args:
        state: current vehicle state
        a: acceleration input
        delta: steering angle input
    Returns:
        Updated state
    """
    # Input constraints
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.v = state.v + a * DT

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED

    return state


def get_nparray_from_matrix(x):
    """Convert matrix to numpy array"""
    return np.array(x).flatten()


def calc_nearest_index(state, cx, cy, cyaw, pind):
    """
    Calculate nearest point on the path to current vehicle position
    Args:
        state: current vehicle state
        cx, cy: path coordinates
        cyaw: path headings
        pind: previous path index
    Returns:
        ind: index of nearest point
        mind: distance to nearest point (signed)
    """
    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)
    ind = d.index(mind) + pind
    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def predict_motion(x0, oa, od, xref):
    """
    Predict vehicle motion using control inputs
    Args:
        x0: initial state
        oa: acceleration inputs
        od: steering angle inputs
        xref: reference trajectory
    Returns:
        xbar: predicted states
    """
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar


def iterative_linear_mpc_control(xref, x0, dref, oa, od):
    """
    MPC controller with iterative linearization
    Args:
        xref: reference trajectory
        x0: current state
        dref: reference steering angles
        oa, od: previous optimal control inputs
    Returns:
        oa, od: optimal control inputs
        ox, oy, oyaw, ov: predicted trajectory
    """
    ox, oy, oyaw, ov = None, None, None, None

    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        du = sum(abs(oa - poa)) + sum(abs(od - pod))  # control input change
        if du <= DU_TH:
            break
    else:
        print("Reached maximum iterations")

    return oa, od, ox, oy, oyaw, ov


def linear_mpc_control(xref, xbar, x0, dref):
    """
    Linear MPC control optimization
    Args:
        xref: reference trajectory
        xbar: linearization points
        x0: current state
        dref: reference steering angles
    Returns:
        oa, odelta: optimal control inputs
        ox, oy, oyaw, ov: predicted trajectory
    """
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.CLARABEL, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])
    else:
        print("Error: Cannot solve MPC optimization")
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov


def calc_ref_trajectory(state, cx, cy, cyaw, sp, dl, pind):
    """
    Calculate reference trajectory for MPC
    Args:
        state: current vehicle state
        cx, cy: path coordinates
        cyaw: path headings
        sp: speed profile
        dl: distance step
        pind: previous index
    Returns:
        xref: reference trajectory
        ind: target index
        dref: reference steering
    """
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0

    return xref, ind, dref


def calc_speed_profile(cx, cy, cyaw, target_speed):
    """
    Calculate speed profile considering path direction
    Args:
        cx, cy: path coordinates
        cyaw: path headings
        target_speed: desired speed
    Returns:
        speed_profile: speed at each point
    """
    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set speed profile based on path direction
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]
        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = -target_speed
        else:
            speed_profile[i] = target_speed

    # Set stop speed at the end
    speed_profile[-1] = 0.0
    return speed_profile


def smooth_yaw(yaw):
    """
    Smooth yaw angle array to avoid discontinuity
    Args:
        yaw: array of yaw angles
    Returns:
        smoothed yaw array
    """
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw


def draw_next_wp(next_wp, world):
    """Draw the next waypoint in red"""
    world.debug.draw_string(next_wp.transform.location, 'O', draw_shadow=False,
                            color=carla.Color(r=255, g=0, b=0), 
                            life_time=100.0, persistent_lines=True)

# Initialize CARLA client
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    map = world.get_map()
    blueprint_library = world.get_blueprint_library()
except Exception as e:
    print(f"Error in client setup: {e}")
    sys.exit(1)

# Set start and destination points
spawn_points = world.get_map().get_spawn_points()
start_point = 87
destination = 70
start_loc = spawn_points[start_point].location
dest_loc = spawn_points[destination].location

print(f"Start location: {start_loc}")
print(f"Destination location: {dest_loc}")

# Generate route using CARLA's global route planner
sample_res = 2  
grp = GlobalRoutePlanner(map, sample_res)
way_points = grp.trace_route(start_loc, dest_loc)
target_speed = 10  # target speed in m/s
cx, cy, cyaw = [], [], []

# Extract and visualize waypoints
# 先提取waypoints坐标点
waypoints_x = []
waypoints_y = []
for this_wp, _ in way_points:
    world.debug.draw_string(this_wp.transform.location, 'O', draw_shadow=False, 
                           color=carla.Color(r=0, g=255, b=0), life_time=100.0, persistent_lines=True)
    waypoints_x.append(this_wp.transform.location.x)
    waypoints_y.append(this_wp.transform.location.y)

# 使用三次样条拟合轨迹
from cubic_spline_planner import calc_spline_course_carla  # 注意导入路径，或者直接粘到项目里

# ds=0.5表示每隔0.5m采样一个轨迹点，可以根据需要调
cx, cy, cyaw, ck, s = calc_spline_course_carla(waypoints_x, waypoints_y, yaw=math.radians(way_points[0][0].transform.rotation.yaw), ds=0.5)

# Spawn ego vehicle
vehicle_bp = blueprint_library.filter('vehicle.*')[3]
first_wp = way_points[0][0]  
start_transform = carla.Transform(start_loc, first_wp.transform.rotation) 
ego = world.spawn_actor(vehicle_bp, start_transform)

# Extract vehicle dimensions
vehicle_L = ego.bounding_box.extent.x * 2
vehicle_W = ego.bounding_box.extent.y * 2

# Set vehicle parameters
LENGTH = vehicle_L  # [m]
WIDTH = vehicle_W  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = vehicle_W-WHEEL_WIDTH  # [m]


ego_transform = ego.get_transform()
print(f"Ego vehicle spawned at: {ego_transform.location} with rotation: {ego_transform.rotation}")

# Setup simulation elements
spectator = world.get_spectator()
sp = calc_speed_profile(cx, cy, cyaw, target_speed)
initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
i = 0
next_wp = way_points[i]
throttle_history, steer_history, brake_history = [], [], []

try:
    # Initialize MPC control variables
    target_ind, _ = calc_nearest_index(initial_state, cx, cy, cyaw, 0)
    odelta, oa = None, None
    
    # Set initial state with yaw aligned to reference path
    state = State(
        x=cx[0], 
        y=cy[0], 
        yaw=cyaw[0], 
        v=0.0
    )
    
    # Initial acceleration phase
    print("Starting vehicle with initial thrust...")
    
    # Apply initial acceleration
    for i in range(10):
        ego.apply_control(carla.VehicleControl(throttle=0.7, steer=0.0, brake=0.0))
        time.sleep(0.1)
        world.wait_for_tick()
    
    # Check if vehicle has started moving
    v = ego.get_velocity()
    speed = v.length()
    print(f"Vehicle started with initial speed: {speed:.2f} m/s")
    
    # Apply stronger acceleration if needed
    if speed < 0.1:
        print("WARNING: Vehicle didn't start moving. Applying stronger thrust...")
        for i in range(5):
            ego.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0))
            time.sleep(0.2)
            world.wait_for_tick()
        v = ego.get_velocity()
        speed = v.length()
        print(f"Second attempt speed: {speed:.2f} m/s")
    
    # Reset tracking variables
    throttle_history, steer_history, brake_history = [], [], []
    path_x, path_y = [], []
    
    # Main MPC control loop
    print("\nStarting MPC control loop...")
    control_iteration = 0
    
    while True:
        control_iteration += 1
        
        # Get current vehicle state
        ego_transform = ego.get_transform()
        ego_loc = ego.get_location()
        v = ego.get_velocity()
        speed = v.length()
        
        # Position spectator camera above vehicle
        spectator.set_transform(carla.Transform(
            ego_transform.location + carla.Location(z=60),
            carla.Rotation(pitch=-90))
        )
        
        # Update state object with current vehicle data
        state.x = ego_loc.x
        state.y = ego_loc.y
        state.v = speed
        state.yaw = math.radians(ego_transform.rotation.yaw)
        
        # Record actual path for plotting
        path_x.append(state.x)
        path_y.append(state.y)
        
        # print(f"\nIteration {control_iteration}")
        # print(f"Current state - x: {state.x:.2f}, y: {state.y:.2f}, v: {state.v:.2f}, yaw: {math.degrees(state.yaw):.2f}°")
        
        # Find nearest reference point
        nearest_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)
        # print(f"Nearest reference point: index {nearest_ind}, pos: ({cx[nearest_ind]:.2f}, {cy[nearest_ind]:.2f})")
        tracking_error = np.sqrt((state.x - cx[nearest_ind])**2 + (state.y - cy[nearest_ind])**2)
    
    # 使用自适应矩阵
        Q, Qf, R, Rd = adaptive_mpc_matrices(
            state, cx, cy, cyaw, 
            nearest_ind, target_ind, 
            tracking_error=tracking_error,
            enable_dynamic=True,  # 设置为False可以禁用动态调整
            debug_output=(control_iteration % 20 == 0)  # 每20次迭代输出一次调试信息
    )
        # Update target index for progress tracking
        if target_ind < nearest_ind:
            target_ind = nearest_ind
        
        # Calculate MPC control inputs
        x0 = [state.x, state.y, state.v, state.yaw]  # current state vector
        xref, target_ind, dref = calc_ref_trajectory(state, cx, cy, cyaw, sp, 2.0, target_ind)
        
        # Run MPC optimization
        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(xref, x0, dref, oa, odelta)
        
        # Extract control commands or use fallback values if optimization fails
        if oa is not None and odelta is not None:
            a, delta = oa[0], odelta[0]
        else:
            print("WARNING: MPC solver failed to find solution")
            a, delta = 0.5, 0.0  # fallback control values
        
        # print(f"MPC output - Acceleration: {a:.4f}, Steering angle: {math.degrees(delta):.2f}°")
        
        # Convert MPC outputs to CARLA control inputs
        if a >= 0.0:
            throttle = min(max(a / MAX_ACCEL, 0.2), 1.0)  # maintain minimum throttle
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(max(-a / MAX_ACCEL, 0.0), 1.0)
        
        steer = min(max(delta / MAX_STEER, -1.0), 1.0)
        
        # Record control history for plotting
        throttle_history.append(throttle)
        steer_history.append(steer)
        brake_history.append(brake)
        
        # print(f"Control outputs - Throttle: {throttle:.4f}, Steering: {steer:.4f}, Brake: {brake:.4f}")
        
        # Apply control to vehicle
        ego_control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake
        )
        ego.apply_control(ego_control)
        
        # Visualize predicted trajectory
        if ox is not None and oy is not None:
            for i in range(len(ox)):
                world.debug.draw_point(
                    carla.Location(x=ox[i], y=oy[i], z=ego_loc.z+0.5),
                    size=0.1,
                    color=carla.Color(r=0, g=0, b=255),
                    life_time=0.2
                )
        
        # Check if destination reached
        if target_ind >= len(cx) - 1:
            print('Destination reached!')
            break
        
        # Wait for next simulation step
        time.sleep(0.1)
        world.wait_for_tick()

finally:
    # Create performance analysis plots
    fig, axs = plt.subplots(4, 1, figsize=(8, 10))
    
    # Trajectory comparison plot
    axs[0].plot(cx, cy, 'r-', label='Reference Path')
    axs[0].plot(path_x, path_y, 'b-', label='Actual Path')
    axs[0].set_title('Path Comparison')
    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')
    axs[0].grid(True)
    axs[0].legend()
    
    # Throttle history plot
    axs[1].plot(throttle_history, label='Throttle', color='g')
    axs[1].set_title('Throttle History')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Throttle Value')
    axs[1].grid(True)
    
    # Steering history plot
    axs[2].plot(steer_history, label='Steering', color='b')
    axs[2].set_title('Steering History')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Steering Value')
    axs[2].grid(True)

    # Brake history plot
    axs[3].plot(brake_history, label='Brake', color='r')
    axs[3].set_title('Brake History')
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('Brake Value')
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()
    
    # Clean up
    ego.destroy()
    print('Simulation completed!')