"""Frozen held-out eligibility, prefix verification, and one-look selector.

Extracted unchanged from notebook 01, cell 72. The evaluation entry point binds
ns, test_config and test_predict to the portable simulator and isolated worker.
"""
import time
import numpy as np

def test_region(grid, xy, path):
    if path is None or len(path) < 2:
        return None
    region = ns['route_unknown_region'](grid, xy, path, ns['OBYM_VIEW_CONFIG'])
    return region if region.any() else None

def test_selector(mode, snapshots, trace, capture):
    attempted, index = (False, 0)

    def selector(grid, xy, yaw, path, memory, sim_time_s):
        nonlocal attempted, index
        started = time.perf_counter()
        current = index
        index += 1
        normalized_path = None if path is None else [[int(r), int(c)] for r, c in path]
        snapshot = {'grid': grid.copy(), 'xy': np.asarray(xy, dtype=np.float64).copy(), 'yaw': float(yaw), 'time_s': float(sim_time_s), 'path': normalized_path}
        if capture:
            snapshots.append(snapshot)
        elif not attempted:
            if current >= len(snapshots):
                raise RuntimeError('Test baza tarixçəsindən kənara çıxdı.')
            ref = snapshots[current]
            assert np.array_equal(grid, ref['grid']) and np.allclose(xy, ref['xy'], atol=1e-08, rtol=0) and (abs(ns['wrap_angle'](yaw - ref['yaw'])) <= 1e-08) and (abs(sim_time_s - ref['time_s']) <= 1e-07) and (normalized_path == ref['path']), f'Test tarixçəsi fərqləndi: qərar {current}'
        row = {'decision_index': current, 'sim_time_s': float(sim_time_s), 'xy': np.asarray(xy).tolist(), 'yaw_rad': float(yaw), 'prefix_verified': not attempted}
        choice = {'action': 'NAVIGATE', 'reason': 'NO_OPTIONAL_VIEW', 'view': None}
        if not capture and (not attempted):
            region = test_region(grid, xy, path)
            if region is not None:
                if mode in ('minus', 'plus'):
                    offset = -45 if mode == 'minus' else 45
                    angle = float(ns['wrap_angle'](yaw + np.deg2rad(offset)))
                    score, area = (None, None)
                else:
                    angles = [float(ns['wrap_angle'](yaw + np.deg2rad(a))) for a in (-45, 45)]
                    samples = [ns['encode_view_sample'](grid, xy, yaw, xy, angle, region) for angle in angles]
                    images = np.stack([sample[0] for sample in samples])
                    poses = np.stack([sample[1] for sample in samples])
                    row['encoding_wall_s'] = time.perf_counter() - started
                    predicting = time.perf_counter()
                    values = test_predict(images, poses, mode)
                    row['prediction_wall_s'] = time.perf_counter() - predicting
                    best = int(np.argmax(values))
                    offset = [-45, 45][best]
                    angle = angles[best]
                    score = float(values[best])
                    area = float(images[best, 5].sum() * ns['GRID_RES'] ** 2)
                    row.update({'values_s': values.tolist(), 'best_offset_deg': offset})
                if mode in ('minus', 'plus') or score > test_config['minimum_value_s']:
                    attempted = True
                    row['selected_offset_deg'] = offset
                    candidate = {'id': 0, 'position': np.asarray(xy), 'path': [], 'travel_distance_m': 0.0}
                    choice = {'action': 'OBSERVE', 'reason': 'FIXED_FIRST_ELIGIBLE' if score is None else 'PREDICTED_NET_TIME_GAIN', 'view': {'candidate_id': 0, 'position': np.asarray(xy).tolist(), 'path': [], 'yaw': angle, 'predicted_area_m2': area, 'estimated_time_s': float(sum(ns['estimate_obym_view_cost'](candidate, xy, yaw, angle))), 'score_s': score, 'target_region': region.copy()}}
        row.update({'action': choice['action'], 'reason': choice['reason'], 'selector_wall_s': time.perf_counter() - started})
        trace.append(row)
        return choice
    return selector
