"""Frozen V2 PostTurnReplan definitions extracted from notebook 01.

Only definitions and constants are included; importing does not run an episode.
The later initial-centering audit is kept in initial_centering_audit.py.
"""
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import heapq, json, time, uuid, math
from itertools import count
from pathlib import Path

wall_specs = [([10, 0.2, 0.8], [0, -4, 0.4]), ([10, 0.2, 0.8], [0, 4, 0.4]), ([0.2, 8, 0.8], [-5, 0, 0.4]), ([0.2, 8, 0.8], [5, 0, 0.4])]

ROBOT_RADIUS = 0.25

ROBOT_HEIGHT = 0.4

start_xy = [-3.5, 0]

goal_xy = [3.5, 0]

CAMERA = {'width': 320, 'height': 240, 'fov': 60, 'near': 0.05, 'far': 15.0}

DT = 1 / 60

motion = {'sim_time_s': 0.0, 'distance_m': 0.0, 'blocked_steps': 0}

motion_log = []

UNKNOWN = -1

FREE = 0

OCCUPIED = 1

GRID_RES = 0.1

X_MIN, X_MAX = (-5.0, 5.0)

Y_MIN, Y_MAX = (-4.0, 4.0)

NX = int(round((X_MAX - X_MIN) / GRID_RES))

NY = int(round((Y_MAX - Y_MIN) / GRID_RES))

Z_MIN = 0.02

Z_MAX = ROBOT_HEIGHT

NZ = 4

Z_RES = (Z_MAX - Z_MIN) / NZ

CLEARANCE = 0.05

UNKNOWN_COSTS = {'Optimistic': 2.0, 'Conservative': 8.0}

NAV_SPEED = 0.4

TURN_SPEED = np.pi / 2

SENSOR_PERIOD = 0.1

GOAL_TOLERANCE = 0.25

OBYM_VIEW_CONFIG = {'lookahead_m': 2.0, 'corridor_radius_m': 0.6, 'max_view_range_m': 4.0, 'gain_seconds_per_m2': 3.0, 'minimum_score_s': 0.15}

OBYM_CONTROL = {'cooldown_s': 2.0, 'max_views': 6, 'memory_radius_m': 0.3, 'memory_angle_deg': 22.5}

def create_box(size, position, color):
    half_size = [value / 2 for value in size]
    collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_size)
    visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_size, rgbaColor=color)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, basePosition=position)

def capture_rgbd(robot_body_id):
    position, orientation = p.getBasePositionAndOrientation(robot_body_id)
    rotation = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    offset = np.array([ROBOT_RADIUS / 2, 0, ROBOT_HEIGHT / 2 + 0.1])
    eye = np.array(position) + rotation @ offset
    forward = rotation @ np.array([1, 0, 0])
    up = rotation @ np.array([0, 0, 1])
    view = p.computeViewMatrix(cameraEyePosition=eye.tolist(), cameraTargetPosition=(eye + forward).tolist(), cameraUpVector=up.tolist())
    width, height = (CAMERA['width'], CAMERA['height'])
    near, far = (CAMERA['near'], CAMERA['far'])
    projection = p.computeProjectionMatrixFOV(fov=CAMERA['fov'], aspect=width / height, nearVal=near, farVal=far)
    image = p.getCameraImage(width=width, height=height, viewMatrix=view, projectionMatrix=projection, renderer=p.ER_TINY_RENDERER)
    rgb = np.asarray(image[2], dtype=np.uint8)
    rgb = rgb.reshape(height, width, 4)[:, :, :3].copy()
    depth_buffer = np.asarray(image[3], dtype=np.float64)
    depth_buffer = depth_buffer.reshape(height, width)
    depth_m = far * near / (far - (far - near) * depth_buffer)
    return (rgb, depth_m, view)

def command(v=0.0, omega=0.0, duration=1.0, animate=False):
    assert np.isfinite([v, omega, duration]).all()
    assert duration >= 0
    assert abs(v) <= 0.5, 'Maksimum sürət: 0.5 m/s'
    assert abs(omega) <= np.pi / 2, 'Maksimum dönmə: 90 dərəcə/s'
    blocked = False
    for i in range(int(np.ceil(duration / DT))):
        dt = min(DT, duration - i * DT)
        old_pos, old_q = p.getBasePositionAndOrientation(robot_id)
        yaw = p.getEulerFromQuaternion(old_q)[2]
        new_yaw = yaw + omega * dt
        if abs(omega) < 1e-09:
            dx = v * dt * np.cos(yaw)
            dy = v * dt * np.sin(yaw)
        else:
            dx = v / omega * (np.sin(new_yaw) - np.sin(yaw))
            dy = v / omega * (np.cos(yaw) - np.cos(new_yaw))
        new_pos = [old_pos[0] + dx, old_pos[1] + dy, old_pos[2]]
        new_q = p.getQuaternionFromEuler([0, 0, new_yaw])
        p.resetBasePositionAndOrientation(robot_id, new_pos, new_q)
        blocked = any((p.getClosestPoints(robot_id, body_id, distance=0) for body_id in solid_ids))
        if blocked:
            p.resetBasePositionAndOrientation(robot_id, old_pos, old_q)
            motion['blocked_steps'] += 1
        else:
            motion['distance_m'] += float(np.hypot(dx, dy))
        p.setTimeStep(dt)
        p.stepSimulation()
        motion['sim_time_s'] += dt
        pos, q = p.getBasePositionAndOrientation(robot_id)
        motion_log.append({'time_s': motion['sim_time_s'], 'x': pos[0], 'y': pos[1], 'yaw_rad': p.getEulerFromQuaternion(q)[2], 'distance_m': motion['distance_m'], 'blocked': blocked})
        if animate:
            time.sleep(dt)
        if blocked:
            print('Maneə ilə təmas: komanda dayandırıldı.')
            break
    return {'blocked': blocked, **motion}

def reset_scene():
    global floor_id, wall_ids, obstacle_id
    global robot_id, goal_id, solid_ids
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    floor_id = create_box([10, 8, 0.1], [0, 0, -0.05], [0.75, 0.75, 0.75, 1])
    wall_ids = [create_box(size, position, [0.4, 0.45, 0.5, 1]) for size, position in wall_specs]
    obstacle_id = create_box([1, 4.8, 2], [0, 0, 1], [0.8, 0.4, 0.15, 1])
    collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=ROBOT_RADIUS, height=ROBOT_HEIGHT)
    visual = p.createVisualShape(p.GEOM_CYLINDER, radius=ROBOT_RADIUS, length=ROBOT_HEIGHT, rgbaColor=[0.1, 0.35, 0.9, 1])
    robot_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, basePosition=[*start_xy, ROBOT_HEIGHT / 2])
    goal_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=[0.1, 0.8, 0.2, 1])
    goal_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=goal_visual, basePosition=[*goal_xy, 0.15])
    solid_ids = wall_ids + [obstacle_id]
    motion.update(sim_time_s=0.0, distance_m=0.0, blocked_steps=0)
    motion_log.clear()
    return [floor_id, *wall_ids, obstacle_id, robot_id, goal_id]

def reset_mapping_scene():
    global goal_id
    reset_scene()
    p.removeBody(goal_id)
    goal_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1, basePosition=[*goal_xy, 0.15])

def depth_to_world(depth, rgb_image, view, stride=4):
    height, width = depth.shape
    vv, uu = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[vv, uu]
    valid = np.isfinite(z) & (z > CAMERA['near']) & (z < CAMERA['far'] - 0.001)
    focal = height / (2 * np.tan(np.deg2rad(CAMERA['fov']) / 2))
    x = (uu + 0.5 - width / 2) * z / focal
    y = -(vv + 0.5 - height / 2) * z / focal
    camera_points = np.column_stack([x[valid], y[valid], -z[valid], np.ones(np.count_nonzero(valid))])
    view_matrix_4x4 = np.array(view).reshape(4, 4, order='F')
    camera_to_world = np.linalg.inv(view_matrix_4x4)
    world_points = (camera_points @ camera_to_world.T)[:, :3]
    colors = rgb_image[vv[valid], uu[valid]] / 255.0
    origin = camera_to_world[:3, 3]
    return (world_points, colors, origin)

def voxel_indices(points):
    cols = np.floor((points[:, 0] - X_MIN) / GRID_RES).astype(int)
    rows = np.floor((points[:, 1] - Y_MIN) / GRID_RES).astype(int)
    layers = np.floor((points[:, 2] - Z_MIN) / Z_RES).astype(int)
    valid = (cols >= 0) & (cols < NX) & (rows >= 0) & (rows < NY) & (layers >= 0) & (layers < NZ)
    return (rows, cols, layers, valid)

def update_occupancy(voxels, points, origin, robot_xy):
    free_evidence = np.zeros(voxels.shape, dtype=bool)
    occupied_evidence = np.zeros(voxels.shape, dtype=bool)
    for endpoint in points:
        delta = endpoint - origin
        count = max(2, int(np.ceil(np.linalg.norm(delta) / (GRID_RES / 2))) + 1)
        fractions = np.linspace(0, 1, count, endpoint=False)[:, None]
        ray_points = origin + fractions * delta
        rows, cols, layers, valid = voxel_indices(ray_points)
        free_evidence[rows[valid], cols[valid], layers[valid]] = True
    rows, cols, layers, valid = voxel_indices(points)
    occupied_evidence[rows[valid], cols[valid], layers[valid]] = True
    voxels[free_evidence & (voxels != OCCUPIED)] = FREE
    voxels[occupied_evidence] = OCCUPIED
    xs = X_MIN + (np.arange(NX) + 0.5) * GRID_RES
    ys = Y_MIN + (np.arange(NY) + 0.5) * GRID_RES
    xx, yy = np.meshgrid(xs, ys)
    inner_radius = max(0.0, ROBOT_RADIUS - GRID_RES / np.sqrt(2))
    footprint = (xx - robot_xy[0]) ** 2 + (yy - robot_xy[1]) ** 2 <= inner_radius ** 2
    voxels[footprint] = FREE
    grid = np.full((NY, NX), UNKNOWN, dtype=np.int8)
    grid[np.all(voxels == FREE, axis=2)] = FREE
    grid[np.any(voxels == OCCUPIED, axis=2)] = OCCUPIED
    return grid

def world_to_cell(xy):
    col = int(np.floor((xy[0] - X_MIN) / GRID_RES))
    row = int(np.floor((xy[1] - Y_MIN) / GRID_RES))
    return (row, col)

def cell_to_world(cell):
    row, col = cell
    return np.array([X_MIN + (col + 0.5) * GRID_RES, Y_MIN + (row + 0.5) * GRID_RES])

def inflate_mask(mask, radius_m):
    cells = int(np.ceil(radius_m / GRID_RES))
    height, width = mask.shape
    padded = np.pad(mask, cells, constant_values=False)
    inflated = np.zeros_like(mask, dtype=bool)
    for dr in range(-cells, cells + 1):
        for dc in range(-cells, cells + 1):
            if np.hypot(dr, dc) * GRID_RES <= radius_m + 1e-12:
                inflated |= padded[cells + dr:cells + dr + height, cells + dc:cells + dc + width]
    return inflated

def make_costmap(grid, unknown_cost=2.0, known_only=False):
    assert np.isfinite(unknown_cost) and unknown_cost >= 1
    padding = ROBOT_RADIUS + CLEARANCE + GRID_RES / np.sqrt(2)
    blocked = inflate_mask(grid == OCCUPIED, padding)
    uncertain = inflate_mask(grid == UNKNOWN, padding)
    height, width = grid.shape
    xs = (np.arange(width) + 0.5) * GRID_RES
    ys = (np.arange(height) + 0.5) * GRID_RES
    edge = ROBOT_RADIUS + CLEARANCE
    blocked |= (xs[None, :] < edge) | (xs[None, :] > width * GRID_RES - edge) | (ys[:, None] < edge) | (ys[:, None] > height * GRID_RES - edge)
    costs = np.ones(grid.shape, dtype=float)
    costs[uncertain] = np.inf if known_only else unknown_cost
    costs[blocked] = np.inf
    return (costs, blocked, uncertain)

def astar(costs, start, goal):
    height, width = costs.shape

    def inside(cell):
        return 0 <= cell[0] < height and 0 <= cell[1] < width
    if not inside(start) or not inside(goal):
        return None
    if not np.isfinite(costs[start]) or not np.isfinite(costs[goal]):
        return None

    def heuristic(cell):
        return GRID_RES * np.hypot(goal[0] - cell[0], goal[1] - cell[1])
    serial = count()
    frontier = [(heuristic(start), next(serial), 0.0, start)]
    scores = np.full(costs.shape, np.inf)
    scores[start] = 0.0
    parent = {}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while frontier:
        _, _, current_g, current = heapq.heappop(frontier)
        if current_g > scores[current] + 1e-12:
            continue
        if current == goal:
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            return path[::-1]
        row, col = current
        for dr, dc in directions:
            neighbor = (row + dr, col + dc)
            if not inside(neighbor):
                continue
            if not np.isfinite(costs[neighbor]):
                continue
            if dr and dc:
                if not np.isfinite(costs[row + dr, col]) or not np.isfinite(costs[row, col + dc]):
                    continue
            distance = GRID_RES * np.hypot(dr, dc)
            new_g = current_g + distance * max(costs[current], costs[neighbor])
            if new_g < scores[neighbor] - 1e-12:
                scores[neighbor] = new_g
                parent[neighbor] = current
                heapq.heappush(frontier, (new_g + heuristic(neighbor), next(serial), new_g, neighbor))
    return None

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def observe_map(voxels):
    image, depth, view = capture_rgbd(robot_id)
    points, _, origin = depth_to_world(depth, image, view)
    position, orientation = p.getBasePositionAndOrientation(robot_id)
    grid = update_occupancy(voxels, points, origin, position[:2])
    return (grid, position, orientation, image)

def known_frontier_path(grid, start, visited):
    costs, _, _ = make_costmap(grid, known_only=True)
    if not np.isfinite(costs[start]):
        return (None, None)
    radius = ROBOT_RADIUS + CLEARANCE + GRID_RES / np.sqrt(2) + 0.2
    frontier_zone = inflate_mask(grid == UNKNOWN, radius) & np.isfinite(costs)
    candidates = []
    for row, col in np.argwhere(frontier_zone):
        cell = (int(row), int(col))
        if cell in visited:
            continue
        distance = np.hypot(row - start[0], col - start[1])
        if distance * GRID_RES < 0.2:
            continue
        goal_distance = np.linalg.norm(cell_to_world(cell) - np.array(goal_xy))
        candidates.append((goal_distance + 0.2 * distance * GRID_RES, cell))
    for _, cell in sorted(candidates)[:16]:
        path = astar(costs, start, cell)
        if path is not None:
            return (path, cell)
    return (None, None)

def encode_view_sample(grid, robot_xy, robot_yaw, candidate_xy, candidate_yaw, target_region):
    baseline = estimate_visible_unknown(grid, robot_xy, robot_yaw, target_region, OBYM_VIEW_CONFIG)
    visible = estimate_visible_unknown(grid, candidate_xy, candidate_yaw, target_region, OBYM_VIEW_CONFIG)
    candidate_mask = np.zeros(grid.shape, dtype=bool)
    candidate_cell = world_to_cell(candidate_xy)
    if not (0 <= candidate_cell[0] < grid.shape[0] and 0 <= candidate_cell[1] < grid.shape[1]):
        raise ValueError('Baxış mövqeyi xəritədən kənardadır.')
    candidate_mask[candidate_cell] = True
    planes = np.stack([grid == FREE, grid == OCCUPIED, grid == UNKNOWN, target_region, candidate_mask, visible & ~baseline, np.ones(grid.shape, dtype=bool)]).astype(np.uint8)
    row, col = world_to_cell(robot_xy)
    half = ML_PATCH_SIZE // 2
    top, left = (row - half, col - half)
    bottom = top + ML_PATCH_SIZE
    right = left + ML_PATCH_SIZE
    src_top = max(0, top)
    src_left = max(0, left)
    src_bottom = min(grid.shape[0], bottom)
    src_right = min(grid.shape[1], right)
    image = np.zeros((len(ML_CHANNELS), ML_PATCH_SIZE, ML_PATCH_SIZE), dtype=np.uint8)
    image[2] = 1
    dst_top = src_top - top
    dst_left = src_left - left
    image[:, dst_top:dst_top + src_bottom - src_top, dst_left:dst_left + src_right - src_left] = planes[:, src_top:src_bottom, src_left:src_right]
    if not image[4].any():
        raise ValueError('Baxış mövqeyi lokal patch-dən kənardadır.')
    offset = (np.asarray(candidate_xy, dtype=float) - np.asarray(robot_xy, dtype=float)) / (half * GRID_RES)
    pose = np.array([offset[0], offset[1], np.cos(candidate_yaw), np.sin(candidate_yaw), np.cos(robot_yaw), np.sin(robot_yaw)], dtype=np.float32)
    return (image, pose)

def grid_line_cells(start_xy, end_xy):
    start_xy = np.asarray(start_xy, dtype=float)
    end_xy = np.asarray(end_xy, dtype=float)
    row, col = world_to_cell(start_xy)
    end_row, end_col = world_to_cell(end_xy)
    dx, dy = end_xy - start_xy
    sx, sy = (int(np.sign(dx)), int(np.sign(dy)))
    tx = (X_MIN + (col + (sx > 0)) * GRID_RES - start_xy[0]) / dx if sx else np.inf
    ty = (Y_MIN + (row + (sy > 0)) * GRID_RES - start_xy[1]) / dy if sy else np.inf
    dtx = GRID_RES / abs(dx) if sx else np.inf
    dty = GRID_RES / abs(dy) if sy else np.inf
    yield (row, col)
    while (row, col) != (end_row, end_col):
        if tx < ty - 1e-12:
            col += sx
            tx += dtx
        elif ty < tx - 1e-12:
            row += sy
            ty += dty
        else:
            yield (row, col + sx)
            yield (row + sy, col)
            row += sy
            col += sx
            tx += dtx
            ty += dty
        yield (row, col)

def route_unknown_region(grid, robot_xy, path, config):
    route_mask = np.zeros(grid.shape, dtype=bool)
    previous = np.asarray(robot_xy, dtype=float)
    walked = 0.0
    route_mask[world_to_cell(robot_xy)] = True
    for cell in path:
        point = cell_to_world(cell)
        walked += float(np.linalg.norm(point - previous))
        if walked > config['lookahead_m']:
            break
        route_mask[cell] = True
        previous = point
    return inflate_mask(route_mask, config['corridor_radius_m']) & (grid == UNKNOWN)

def estimate_visible_unknown(grid, robot_xy, yaw, target_mask, config):
    visible = np.zeros(grid.shape, dtype=bool)
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    eye_xy = np.asarray(robot_xy, dtype=float) + ROBOT_RADIUS / 2 * forward
    eye_cell = world_to_cell(eye_xy)
    if not (0 <= eye_cell[0] < grid.shape[0] and 0 <= eye_cell[1] < grid.shape[1]):
        return visible
    if grid[eye_cell] == OCCUPIED:
        return visible
    horizontal_fov = 2 * np.arctan(np.tan(np.deg2rad(CAMERA['fov']) / 2) * CAMERA['width'] / CAMERA['height'])
    max_range = min(config['max_view_range_m'], CAMERA['far'])
    for row, col in np.argwhere(target_mask & (grid == UNKNOWN)):
        target = cell_to_world((row, col))
        delta = target - eye_xy
        distance = float(np.linalg.norm(delta))
        forward_depth = float(delta @ forward)
        angle = wrap_angle(np.arctan2(delta[1], delta[0]) - yaw)
        if distance > max_range or forward_depth < CAMERA['near']:
            continue
        if abs(angle) > horizontal_fov / 2:
            continue
        clear = True
        for cell in grid_line_cells(eye_xy, target):
            if not (0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]):
                clear = False
                break
            if grid[cell] == OCCUPIED:
                clear = False
                break
        if clear:
            visible[row, col] = True
    return visible

def new_obym_state():
    return {'completed_views': [], 'last_view_s': None}

def remember_obym_view(state, xy, yaw, sim_time_s):
    state['completed_views'].append({'position': np.asarray(xy, dtype=float).tolist(), 'yaw': float(wrap_angle(yaw)), 'time_s': float(sim_time_s)})
    state['last_view_s'] = float(sim_time_s)

def estimate_obym_view_cost(candidate, xy, yaw, target_yaw):
    previous = np.asarray(xy, dtype=float)
    heading = float(yaw)
    turning = 0.0
    sensor_waits = 1
    for cell in candidate['path']:
        point = cell_to_world(cell)
        delta = point - previous
        if np.linalg.norm(delta) > 1e-09:
            next_heading = float(np.arctan2(delta[1], delta[0]))
            turn = abs(wrap_angle(next_heading - heading))
            turning += turn
            sensor_waits += 1 + int(turn > 1e-06)
            heading = next_heading
        previous = point
    turning += abs(wrap_angle(target_yaw - heading))
    travel_s = candidate['travel_distance_m'] / NAV_SPEED
    turn_s = turning / TURN_SPEED
    sensor_s = sensor_waits * SENSOR_PERIOD
    return (float(travel_s), float(turn_s), float(sensor_s))

def ml_scene_geometry(spec):
    rng = np.random.default_rng(spec['seed'])
    width = float(rng.uniform(0.7, 1.3))
    length = float(rng.uniform(3.5, 5.0))
    center_x = float(rng.uniform(-0.35, 0.35))
    center_y = float(rng.uniform(-0.35, 0.35))
    height = float(rng.choice([0.45, 0.8, 1.3, 2.0]))
    gate_width = float(rng.uniform(0.8, 1.4))
    if spec['kind'] not in {'barrier', 'gate_open', 'gate_closed'}:
        raise ValueError(f"Naməlum ML səhnəsi: {spec['kind']}")
    if spec['kind'] != 'barrier':
        height = 2.0
    boxes = [([width, length, height], [center_x, center_y, height / 2])]
    if spec['kind'] != 'barrier':
        main_right = center_x + width / 2
        main_top = center_y + length / 2
        gate_x = main_right + gate_width / 2
        boxes.append(([gate_width, 0.4, 2.0], [gate_x, main_top - 0.2, 1.0]))
        if spec['kind'] == 'gate_closed':
            gap_height = 3.9 - main_top
            boxes.append(([gate_width, gap_height, 2.0], [gate_x, main_top + gap_height / 2, 1.0]))
    return boxes

def reset_ml_scene(spec):
    global obstacle_id, solid_ids
    reset_mapping_scene()
    p.removeBody(obstacle_id)
    bodies = [create_box(size, position, [0.8, 0.4, 0.15, 1]) for size, position in ml_scene_geometry(spec)]
    obstacle_id = bodies[0]
    solid_ids = list(wall_ids) + bodies
    p.setRealTimeSimulation(0)

def run_postturn_audit(method='OBYM', max_sim_time=180.0, max_decisions=300, animate=False, reset_fn=None, scene_name='fixed_warehouse'):
    if method not in {'Optimistic', 'Conservative', 'OBYM'}:
        raise ValueError(f"Naməlum üsul: {method}")
    started = time.perf_counter()
    reset_function = reset_mapping_scene if reset_fn is None else reset_fn
    reset_function()
    p.setRealTimeSimulation(0)
    penalty = UNKNOWN_COSTS['Optimistic' if method == 'OBYM' else method]
    voxels = np.full((NY, NX, NZ), UNKNOWN, dtype=np.int8)
    grid = np.full((NY, NX), UNKNOWN, dtype=np.int8)
    state = new_obym_state()
    pending = None
    events, decisions, recent_positions = ([], [], [])
    visited_frontiers = set()
    active_frontier = None
    scans_without_motion = 0
    observations = planning_cycles = frontier_scans = view_attempts = 0
    last_rgb = None
    outcome = 'STEP_LIMIT'

    def act(v=0.0, omega=0.0, duration=0.0):
        remaining = max(0.0, max_sim_time - motion['sim_time_s'])
        if remaining <= 1e-09:
            return 'TIMEOUT'
        result = command(v=v, omega=omega, duration=min(duration, remaining), animate=animate)
        if result['blocked']:
            return 'COLLISION'
        return 'TIMEOUT' if duration > remaining + 1e-09 else None

    def observe():
        nonlocal grid, observations, last_rgb
        status = act(duration=SENSOR_PERIOD)
        if status is None:
            grid, _, _, last_rgb = observe_map(voxels)
            observations += 1
        return status

    def rotate_to(target_yaw):
        _, orientation = p.getBasePositionAndOrientation(robot_id)
        yaw = p.getEulerFromQuaternion(orientation)[2]
        turn = wrap_angle(target_yaw - yaw)
        if abs(turn) <= 1e-06:
            return None
        status = act(omega=float(np.sign(turn) * TURN_SPEED), duration=float(abs(turn) / TURN_SPEED))
        return status if status is not None else observe()

    def finish_view(status):
        nonlocal pending
        position, orientation = p.getBasePositionAndOrientation(robot_id)
        newly_known = (pending['before'] == UNKNOWN) & (grid != UNKNOWN)
        events.append({'status': status, 'candidate_id': pending['candidate_id'], 'target_position': pending['position'], 'target_yaw_rad': pending['yaw'], 'start_time_s': pending['start_s'], 'end_time_s': float(motion['sim_time_s']), 'predicted_area_m2': pending['predicted_area_m2'], 'actual_route_area_m2': float(np.count_nonzero(newly_known & pending['target_region']) * GRID_RES ** 2), 'actual_total_area_m2': float(np.count_nonzero(newly_known) * GRID_RES ** 2), 'estimated_time_s': pending['estimated_time_s'], 'actual_time_s': float(motion['sim_time_s'] - pending['start_s']), 'actual_distance_m': float(motion['distance_m'] - pending['start_m']), 'score_s': pending['score_s']})
        if status == 'COMPLETED':
            remember_obym_view(state, position[:2], p.getEulerFromQuaternion(orientation)[2], motion['sim_time_s'])
        else:
            state['last_view_s'] = float(motion['sim_time_s'])
        pending = None
    for iteration in range(max_decisions):
        position, _ = p.getBasePositionAndOrientation(robot_id)
        xy = np.array(position[:2])
        if np.linalg.norm(xy - np.array(goal_xy)) <= GOAL_TOLERANCE:
            outcome = 'GOAL'
            break
        status = observe()
        if status is not None:
            outcome = status
            break
        position, orientation = p.getBasePositionAndOrientation(robot_id)
        xy = np.array(position[:2])
        yaw = p.getEulerFromQuaternion(orientation)[2]
        start = world_to_cell(xy)
        goal = world_to_cell(goal_xy)
        costs, _, _ = make_costmap(grid, unknown_cost=penalty)
        goal_path = astar(costs, start, goal)
        planning_cycles += 1
        if method == 'OBYM' and pending is None and (view_attempts < OBYM_CONTROL['max_views']):
            choice = choose_obym_view(grid, xy, yaw, goal_path, state, motion['sim_time_s'])
            decisions.append({'iteration': iteration, 'time_s': float(motion['sim_time_s']), 'action': 'VIEW_DECISION', 'choice': choice['action'], 'reason': choice['reason']})
            if choice['action'] == 'OBSERVE':
                pending = {**choice['view'], 'before': grid.copy(), 'start_s': float(motion['sim_time_s']), 'start_m': float(motion['distance_m'])}
                view_attempts += 1
        if pending is not None:
            target_cell = world_to_cell(pending['position'])
            if np.linalg.norm(np.array(pending['position']) - xy) <= 1e-06:
                _, orientation = p.getBasePositionAndOrientation(robot_id)
                current_yaw = p.getEulerFromQuaternion(orientation)[2]
                needs_turn = abs(wrap_angle(pending['yaw'] - current_yaw)) > 1e-06
                status = rotate_to(pending['yaw'])
                if status is None and (not needs_turn):
                    status = observe()
                if status is not None:
                    outcome = status
                    break
                finish_view('COMPLETED')
                scans_without_motion = 0
                continue
            known_costs, _, _ = make_costmap(grid, known_only=True)
            path = astar(known_costs, start, target_cell)
            planning_cycles += 1
            if path is None:
                finish_view('ABORTED_NO_KNOWN_PATH')
                continue
            mode = 'VIEW_MOVE'
        else:
            recent_positions.append(xy.copy())
            if len(recent_positions) >= 30:
                window = np.array(recent_positions[-30:])
                if np.max(np.linalg.norm(window - xy, axis=1)) < 0.1:
                    outcome = 'STUCK'
                    break
            path, mode = (goal_path, 'NAVIGATE')
            if path is not None:
                active_frontier = None
            else:
                mode = 'FRONTIER'
                if active_frontier is not None:
                    reached = np.linalg.norm(cell_to_world(active_frontier) - xy) <= 0.12
                    if reached:
                        visited_frontiers.add(active_frontier)
                        active_frontier = None
                    else:
                        known_costs, _, _ = make_costmap(grid, known_only=True)
                        path = astar(known_costs, start, active_frontier)
                        planning_cycles += 1
                        if path is None:
                            visited_frontiers.add(active_frontier)
                            active_frontier = None
                if path is None and active_frontier is None:
                    path, active_frontier = known_frontier_path(grid, start, visited_frontiers)
        if path is None:
            if scans_without_motion >= 4:
                outcome = 'NO_PATH'
                break
            status = act(omega=TURN_SPEED, duration=1.0)
            frontier_scans += 1
            scans_without_motion += 1
            decisions.append({'iteration': iteration, 'time_s': float(motion['sim_time_s']), 'action': 'FRONTIER_SCAN'})
            if status is not None:
                outcome = status
                break
            continue
        if np.linalg.norm(cell_to_world(path[0]) - xy) > 0.025:
            target_cell = path[0]
        else:
            target_cell = path[1] if len(path) > 1 else path[0]
        target = cell_to_world(target_cell)
        delta = target - xy
        if np.linalg.norm(delta) <= 1e-09:
            continue
        decisions.append({'iteration': iteration, 'time_s': float(motion['sim_time_s']), 'action': mode, 'position': xy.tolist(), 'target': target.tolist()})
        status = rotate_to(float(np.arctan2(delta[1], delta[0])))
        if status is not None:
            outcome = status
            break
        if mode == 'NAVIGATE' and pending is None and (abs(wrap_angle(float(np.arctan2(delta[1], delta[0])) - yaw)) > 1e-06):
            AUDIT_TRACE['postturn_checks'] += 1
            fresh_costs, _, _ = make_costmap(grid, unknown_cost=penalty)
            fresh_path = astar(fresh_costs, start, goal)
            planning_cycles += 1
            if fresh_path is None:
                AUDIT_TRACE['changed_next_action'] += 1
                decisions[-1]['action'] = 'AUDIT_REPLAN_AFTER_TURN'
                continue
            if np.linalg.norm(cell_to_world(fresh_path[0]) - xy) > 0.025:
                fresh_target = fresh_path[0]
            else:
                fresh_target = fresh_path[1] if len(fresh_path) > 1 else fresh_path[0]
            if fresh_target != target_cell:
                AUDIT_TRACE['changed_next_action'] += 1
                decisions[-1]['action'] = 'AUDIT_REPLAN_AFTER_TURN'
                continue
        updated_costs, _, _ = make_costmap(grid, unknown_cost=penalty, known_only=mode in {'FRONTIER', 'VIEW_MOVE'})
        blocked = not np.isfinite(updated_costs[target_cell])
        if mode in {'FRONTIER', 'VIEW_MOVE'}:
            blocked |= not np.isfinite(updated_costs[start])
        dr = target_cell[0] - start[0]
        dc = target_cell[1] - start[1]
        if abs(dr) == 1 and abs(dc) == 1:
            blocked |= not np.isfinite(updated_costs[start[0] + dr, start[1]]) or not np.isfinite(updated_costs[start[0], start[1] + dc])
        if blocked:
            decisions[-1]['action'] = 'REPLAN'
            if pending is not None:
                finish_view('ABORTED_MAP_CHANGED')
            continue
        position, _ = p.getBasePositionAndOrientation(robot_id)
        distance = float(np.linalg.norm(target - np.array(position[:2])))
        status = act(v=NAV_SPEED, duration=distance / NAV_SPEED)
        if status is not None:
            outcome = status
            break
        scans_without_motion = 0
        if iteration % 25 == 0:
            print(f"{method} | addım={iteration} | t={motion['sim_time_s']:.1f}s")
    position, _ = p.getBasePositionAndOrientation(robot_id)
    goal_error = float(np.linalg.norm(np.array(position[:2]) - np.array(goal_xy)))
    if outcome != 'COLLISION' and goal_error <= GOAL_TOLERANCE:
        outcome = 'GOAL'
    if pending is not None:
        finish_view(f"INTERRUPTED_{outcome}")
    run_id = f"{method.lower()}_{uuid.uuid4().hex[:8]}"
    run_dir = Path('results/day3') / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {'run_id': run_id, 'runner_revision': 'day3_shared_v1', 'method': method, 'scene': scene_name, 'outcome': outcome, 'success': outcome == 'GOAL', 'sim_time_s': float(motion['sim_time_s']), 'distance_m': float(motion['distance_m']), 'collision_attempts': int(motion['blocked_steps']), 'goal_error_m': goal_error, 'final_position': list(position), 'observations': observations, 'planning_cycles': planning_cycles, 'frontier_scans': frontier_scans, 'view_attempts': view_attempts, 'completed_views': len(state['completed_views']), 'max_sim_time_s': max_sim_time, 'max_decisions': max_decisions, 'wall_time_s': float(time.perf_counter() - started), 'config': {'unknown_cost': penalty, 'speed_m_s': NAV_SPEED, 'turn_speed_rad_s': float(TURN_SPEED), 'sensor_period_s': SENSOR_PERIOD, 'goal_tolerance_m': GOAL_TOLERANCE, 'view_scoring': dict(OBYM_VIEW_CONFIG), 'view_control': dict(OBYM_CONTROL)}, 'directory': str(run_dir)}
    for name, data in [('summary.json', summary), ('motion.json', list(motion_log)), ('decisions.json', decisions), ('views.json', events)]:
        (run_dir / name).write_text(json.dumps(data, indent=2), encoding='utf-8')
    np.save(run_dir / 'map.npy', grid)
    np.save(run_dir / 'voxels.npy', voxels)
    if last_rgb is not None:
        plt.imsave(run_dir / 'last_rgb.png', last_rgb)
    print(f"{method}: {outcome} | {summary['distance_m']:.3f} m | {summary['sim_time_s']:.3f} s | baxış={summary['completed_views']} | təmas={summary['collision_attempts']}")
    return summary
