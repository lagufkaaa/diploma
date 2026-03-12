import sys
import os
from pathlib import Path
import time

import numpy as np
import pytest
from shapely.geometry import Polygon, MultiPolygon

# Ensure 'src' is on sys.path so we can import the package modules
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from core.data import Data, Item
from core.encoding import Encoding
from solvers.model import Problem
from solvers.greedy_solver import GreedySolver
from utils.helpers import util_model

from shapely import affinity

# bring visualization helper for debugging
vis = util_model().visualize_solution


DATA_DIR = Path(__file__).resolve().parents[2] / 'data_car_mats'


def test_parse_items_and_data_items():
    file_path = DATA_DIR / 'test.txt'
    items = util_model.parse_items(str(file_path))
    assert isinstance(items, list)
    assert len(items) > 0

    # limit the number of shapes to avoid heavy computations during tests
    items = items[:5]

    data = Data(items, R=1)
    assert isinstance(data.items, list)
    assert all(isinstance(it, Item) for it in data.items)
    # each item should have a reference back to data and an nfp dict (possibly empty)
    for it in data.items:
        assert it.data is data
        assert it.nfp is None or isinstance(it.nfp, dict)


def test_shared_memory_cache_reused_between_data_instances():
    items = [
        np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]]),
        np.array([[0.0, 0.0], [15.0, 0.0], [15.0, 30.0], [0.0, 30.0]]),
    ]
    shared_cache = {}

    data_model = Data(
        items,
        R=1,
        parallel_nfp=False,
        use_cache=False,
        use_memory_cache=True,
        shared_memory_cache=shared_cache,
    )
    assert data_model.nfp_stats["computed_pairs"] > 0

    data_greedy = Data(
        items,
        R=1,
        parallel_nfp=False,
        use_cache=False,
        use_memory_cache=True,
        shared_memory_cache=shared_cache,
    )
    assert data_greedy.nfp_stats["computed_pairs"] == 0
    assert data_greedy.nfp_stats["memory_cache_hits"] == data_greedy.nfp_stats["pairs_total"]


def test_item_rotation_area_preserved():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    it = Item(square)
    orig_area = it.area
    it.change_rotation(90)
    assert pytest.approx(orig_area, rel=1e-9) == pytest.approx(it.area, rel=1e-9)
    assert it.rotation == 90


def test_outer_NFP_returns_geometry():
    # two simple squares, one shifted to the right
    square1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    square2 = np.array([[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]])
    a = Item(square1)
    b = Item(square2)
    poly = a.outer_NFP(b, anchor_point=(0, 0))
    assert isinstance(poly, (Polygon, MultiPolygon))


def test_encode_polygon_basic():
    poly = Polygon([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    # create Y levels for S=2 rows
    Y = [0.0, 0.5, 1.0]
    segs = Encoding.encode_polygon(poly, Y)
    assert isinstance(segs, list)
    assert len(segs) > 0
    # each segment should be [[x1,y],[x2,y]]
    for s in segs:
        assert isinstance(s, list) and len(s) == 2
        (x1, y1), (x2, y2) = s[0], s[1]
        assert pytest.approx(y1, rel=1e-9) == pytest.approx(y2, rel=1e-9)
        
def placed_geometry(item, x_shift, y_shift):
    pts = np.asarray(item.points, dtype=float)   # как в Item, без norm()
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return affinity.translate(poly, xoff=float(x_shift), yoff=float(y_shift))


def visualize_greedy_solution(items, solution, width, height):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(12, 8))

    if solution.get('status') != 'OPTIMAL':
        ax.text(
            0.5,
            0.5,
            f"status: {solution.get('status')}",
            ha='center',
            va='center',
            transform=ax.transAxes,
            bbox=dict(facecolor='red', alpha=0.3),
        )
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        plt.show()
        return

    p_vals = solution.get('p', [0.0 for _ in range(len(items))])
    x_vals = solution.get('x', [0.0 for _ in range(len(items))])
    y_vals = solution.get('y', [0.0 for _ in range(len(items))])

    for idx, it in enumerate(items):
        if idx >= len(p_vals) or p_vals[idx] <= 0.5:
            continue

        arr = np.asarray(it.points, dtype=float)
        x = float(x_vals[idx]) if idx < len(x_vals) else 0.0
        y = float(y_vals[idx]) if idx < len(y_vals) else 0.0
        coords = [(float(xx) + x, float(yy) + y) for xx, yy in arr]

        poly = patches.Polygon(coords, closed=True, alpha=0.6)
        ax.add_patch(poly)
        ax.text(coords[0][0], coords[0][1], str(idx), color='white')

    ax.axvline(x=0, color='black', linewidth=2)
    ax.axvline(x=width, color='black', linewidth=2)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=height, color='black', linewidth=2)

    ax.set_xlim(-width * 0.05, width * 1.05)
    ax.set_ylim(-height * 0.05, height * 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.show()


def test_model_basic():
    file_path = DATA_DIR / 'car_mats_2.txt' #'test.txt'
    items = util_model.parse_items(str(file_path))
    assert len(items) > 0, "no items parsed from test file"
    
    mdl_items = []
    for i in range( len(items)//5):
        mdl_items.append(items[5*i])

    items = mdl_items #[:4] 
    
    R = 4
    S = 10
    height = 3000.0
    width = 3000.0

    # file_path = DATA_DIR / 'test.txt'
    # items = util_model.parse_items(str(file_path))
    # assert len(items) > 0, "no items parsed from test file"
    # items = items 
    
    # R = 4
    # S = 10
    # height = 1250.0
    # width = 1250.0

    
    total_start = time.time()
    
    data_start = time.time()
    data = Data(items, R)
    data_time = time.time() - data_start
    
    problem_start = time.time()
    problem = Problem(data, S, R, height, width)
    problem_time = time.time() - problem_start
    
    solve_start = time.time()
    results = problem.solve()
    solve_time = time.time() - solve_start
    
    total_time = time.time() - total_start

    print("--- model solve results ---")
    print(results)
    print(f"Data creation time: {data_time:.4f} seconds")
    print(f"Problem creation time: {problem_time:.4f} seconds")
    print(f"Solve() call time: {solve_time:.4f} seconds")
    print(f"Total (Data + Problem + Solve): {total_time:.4f} seconds")

    status = results.get("status")
    has_primal = (
        status == "OPTIMAL"
        and isinstance(results.get("x"), list)
        and isinstance(results.get("deltas"), list)
        and len(results["x"]) == len(data.items)
        and len(results["deltas"]) == len(data.items)
    )
    assert has_primal, f"No primal solution to inspect. Solver status: {status}"

    try:
        vis(data.items, results, problem.width, problem.height, problem.S)
    except Exception as e:
        print("visualization failed:", e)

    print("\n--- item positions and bounds ---")
    packed_items = []
    for idx, item in enumerate(data.items):
        p_val = problem.p[item].solution_value()
        x_val = problem.x[item].solution_value()
        deltas_val = results["deltas"][idx]

        is_packed = any(d > 0.5 for d in deltas_val)
        if not is_packed:
            continue

        strip_idx = next((s for s, d in enumerate(deltas_val) if d > 0.5), None)
        packed_items.append({
            "orig_idx": idx,
            "item": item,
            "x": float(x_val),
            "s": int(strip_idx),
        })

    print("\n--- geometry intersection check (ALL strips) ---")
    h_strip = height / S
    eps_area = 1e-6

    placed = []
    for rec in packed_items:
        s_idx = rec["s"]
        x_val = rec["x"]
        y_shift = s_idx * h_strip

        geom_global = placed_geometry(rec["item"], x_shift=x_val, y_shift=y_shift)

        placed.append({
            "orig_idx": rec["orig_idx"],
            "strip": s_idx,
            "geom": geom_global,
        })

        miny, maxy = geom_global.bounds[1], geom_global.bounds[3]
        if miny < s_idx * h_strip - 1e-6 or maxy > (s_idx + 1) * h_strip + 1e-6:
            print(f"!! ITEM {rec['orig_idx']} leaves strip {s_idx}: y=[{miny:.3f},{maxy:.3f}] "
                f"strip=[{s_idx*h_strip:.3f},{(s_idx+1)*h_strip:.3f}]")

    # check overlaps
    overlaps = []
    for a_i in range(len(placed)):
        for b_i in range(a_i + 1, len(placed)):
            ga = placed[a_i]["geom"]
            gb = placed[b_i]["geom"]

            # bbox fast reject
            if (ga.bounds[2] <= gb.bounds[0] or gb.bounds[2] <= ga.bounds[0] or
                ga.bounds[3] <= gb.bounds[1] or gb.bounds[3] <= ga.bounds[1]):
                continue

            if ga.intersects(gb):
                inter = ga.intersection(gb)
                area = getattr(inter, "area", 0.0) or 0.0
                if area > eps_area:
                    overlaps.append((
                        placed[a_i]["orig_idx"],
                        placed[b_i]["orig_idx"],
                        area,
                        placed[a_i]["strip"],
                        placed[b_i]["strip"],
                    ))

    if overlaps:
        print("!! OVERLAPS DETECTED:")
        for i, j, area, si, sj in overlaps:
            print(f"  items {i} and {j} overlap, area={area:.6f}, strips=({si},{sj})")

    assert not overlaps, "Packed items have geometric intersections!"


def test_greedy_basic():
    file_path = DATA_DIR / 'car_mats_1.txt'
    items = util_model.parse_items(str(file_path))
    assert len(items) > 0, "no items parsed from test file"

    mdl_items = []
    for i in range( len(items)//5):
        mdl_items.append(items[5*i])

    items = mdl_items[:4] 

    R = 4
    S = 10
    height = 125000.0
    width = 125000.0

    total_start = time.time()

    print("starting data")
    
    data_start = time.time()
    data = Data(items, R, parallel_nfp=False)
    data_time = time.time() - data_start
    
    print("starting greedy")

    unique_ids = len({it.id for it in data.items})
    print(
        f"orig_items={len(items)}, expanded_items={len(data.items)}, unique_item_ids={unique_ids}, R={R}"
    )
    assert unique_ids == len(items), (
        "Each original piece should map to one item.id group. "
        "If unique_item_ids is 1 while expanded_items is R, "
        "you passed one piece with rotations, not multiple pieces."
    )

    solver_start = time.time()
    solver = GreedySolver(data, height=height, width=width, S=S, delta_x=10.0)
    solver_time = time.time() - solver_start

    solve_start = time.time()
    results = solver.solve()
    solve_time = time.time() - solve_start

    total_time = time.time() - total_start

    print("--- greedy solve results ---")
    print(results)
    print(f"Data creation time: {data_time:.4f} seconds")
    print(f"Solver creation time: {solver_time:.4f} seconds")
    print(f"Solve() call time: {solve_time:.4f} seconds")
    print(f"Total (Data + Solver + Solve): {total_time:.4f} seconds")

    status = results.get("status")
    has_primal = (
        status == "OPTIMAL"
        and isinstance(results.get("x"), list)
        and isinstance(results.get("y"), list)
        and isinstance(results.get("deltas"), list)
        and len(results["x"]) == len(data.items)
        and len(results["y"]) == len(data.items)
        and len(results["deltas"]) == len(data.items)
    )
    assert has_primal, f"No primal solution to inspect. Solver status: {status}"

    try:
        visualize_greedy_solution(data.items, results, width, height)
    except Exception as e:
        print("visualization failed:", e)

    print("\n--- item positions and bounds (greedy, continuous y) ---")
    packed_items = []
    for idx, item in enumerate(data.items):
        p_val = float(results["p"][idx])
        is_packed = p_val > 0.5
        if not is_packed:
            continue

        x_val = float(results["x"][idx])
        y_val = float(results["y"][idx])
        packed_items.append({
            "orig_idx": idx,
            "item": item,
            "x": x_val,
            "y": y_val,
        })

    print("\n--- geometry intersection check (global, greedy) ---")
    eps_area = 1e-6

    placed = []
    for rec in packed_items:
        x_val = rec["x"]
        y_shift = rec["y"]

        geom_global = placed_geometry(rec["item"], x_shift=x_val, y_shift=y_shift)

        placed.append({
            "orig_idx": rec["orig_idx"],
            "x": x_val,
            "y": y_shift,
            "geom": geom_global,
        })

    overlaps = []
    for a_i in range(len(placed)):
        for b_i in range(a_i + 1, len(placed)):
            ga = placed[a_i]["geom"]
            gb = placed[b_i]["geom"]

            if (
                ga.bounds[2] <= gb.bounds[0]
                or gb.bounds[2] <= ga.bounds[0]
                or ga.bounds[3] <= gb.bounds[1]
                or gb.bounds[3] <= ga.bounds[1]
            ):
                continue

            if ga.intersects(gb):
                inter = ga.intersection(gb)
                area = getattr(inter, "area", 0.0) or 0.0
                if area > eps_area:
                    overlaps.append((
                        placed[a_i]["orig_idx"],
                        placed[b_i]["orig_idx"],
                        area,
                        placed[a_i]["y"],
                        placed[b_i]["y"],
                    ))

    if overlaps:
        print("!! OVERLAPS DETECTED (greedy):")
        for i, j, area, yi, yj in overlaps:
            print(f"  items {i} and {j} overlap, area={area:.6f}, y=({yi:.3f},{yj:.3f})")

    assert not overlaps, "Greedy packed items have geometric intersections!"


def test_single_square_fits():
    square = np.array([[0.0,0.0],[80.0,0.0],[80.0,80.0],[0.0,80.0]])
    data = Data([square], R=1)
    assert len(data.items) == 1

    problem = Problem(data, S=1, R=1, height=100.0, width=100.0)
    res = problem.solve()
    assert res['status'] == 'OPTIMAL'
    assert res['objective_value'] == pytest.approx(80.0 * 80.0)
    print(res)
