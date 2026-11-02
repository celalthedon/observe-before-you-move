"""Later first-movement audit; never used in the frozen V2 evaluation."""
import ast
import copy

def make_initial_centering_audit(patched):
    runner_center = copy.deepcopy(patched)
    runner_center.name = 'run_initial_centering_audit'
    changed_center = []
    for node in ast.walk(runner_center):
        if not isinstance(node, ast.If) or not node.body or (not isinstance(node.body[0], ast.Assign)):
            continue
        target_names = {target.id for target in node.body[0].targets if isinstance(target, ast.Name)}
        if target_names not in ({'target_cell'}, {'fresh_target'}):
            continue
        text = ast.unparse(node.test)
        if 'cell_to_world' not in text or '0.025' not in text:
            continue
        guard = ast.parse('not (mode == "NAVIGATE" and pending is None and motion["distance_m"] <= 1e-9)', mode='eval').body
        node.test = ast.BoolOp(op=ast.And(), values=[node.test, guard])
        changed_center.append(next(iter(target_names)))
    assert sorted(changed_center) == ['fresh_target', 'target_cell'], 'İki planlama yeri də uyğun dəyişməlidir.'
    return ast.fix_missing_locations(runner_center)
