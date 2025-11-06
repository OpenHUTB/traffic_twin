import math
import numpy as np
import cvxpy
from simulation import config

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
    A = np.zeros((config.NX, config.NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = config.DT * math.cos(phi)
    A[0, 3] = - config.DT * v * math.sin(phi)
    A[1, 2] = config.DT * math.sin(phi)
    A[1, 3] = config.DT * v * math.cos(phi)
    A[3, 2] = config.DT * math.tan(delta) / config.WB

    B = np.zeros((config.NX, config.NU))
    B[2, 0] = config.DT
    B[3, 1] = config.DT * v / (config.WB * math.cos(delta) ** 2)

    C = np.zeros(config.NX)
    C[0] = config.DT * v * math.sin(phi) * phi
    C[1] = - config.DT * v * math.cos(phi) * phi
    C[3] = - config.DT * v * delta / (config.WB * math.cos(delta) ** 2)

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
    if delta >= config.MAX_STEER:
        delta = config.MAX_STEER
    elif delta <= -config.MAX_STEER:
        delta = -config.MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * config.DT
    state.y = state.y + state.v * math.sin(state.yaw) * config.DT
    state.yaw = state.yaw + state.v / config.WB * math.tan(delta) * config.DT
    state.v = state.v + a * config.DT

    if state.v > config.MAX_SPEED:
        state.v = config.MAX_SPEED
    elif state.v < config.MIN_SPEED:
        state.v = config.MIN_SPEED

    return state

def get_nparray_from_matrix(x):
    """Convert matrix to numpy array"""
    return np.array(x).flatten()

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

    from simulation.vehicle_control import State
    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, config.T + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar

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
    x = cvxpy.Variable((config.NX, config.T + 1))
    u = cvxpy.Variable((config.NU, config.T))

    cost = 0.0
    constraints = []

    for t in range(config.T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (config.T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)

    cost += cvxpy.quad_form(xref[:, config.T] - x[:, config.T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= config.MAX_SPEED]
    constraints += [x[2, :] >= config.MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= config.MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= config.MAX_STEER]

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
        oa = [0.0] * config.T
        od = [0.0] * config.T

    for i in range(config.MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        du = sum(abs(oa - poa)) + sum(abs(od - pod))  # control input change
        if du <= config.DU_TH:
            break
    else:
        print("Reached maximum iterations")

    return oa, od, ox, oy, oyaw, ov

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
    from simulation.vehicle_control import calc_nearest_index
    
    xref = np.zeros((config.NX, config.T + 1))
    dref = np.zeros((1, config.T + 1))
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

    for i in range(config.T + 1):
        travel += abs(state.v) * config.DT
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

# 全局MPC参数
Q = np.diag([3.0, 3.0, 0.5, 1.0])  # 状态权重矩阵
R = np.diag([0.01, 0.1])  # 控制权重矩阵
Qf = Q * 5.0  # 终端状态权重矩阵
Rd = np.diag([0.01, 1.0])  # 控制变化率权重矩阵

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