import math as math
import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.affinity import translate
import pyclipper
from shapely.ops import unary_union

class util_polygon:
    def __init__(self):
        pass        
    
    def polygon_to_path(polygon):
        """РџСЂРµРѕР±СЂР°Р·СѓРµС‚ shapely Polygon РІ numpy РјР°СЃСЃРёРІ С‚РѕС‡РµРє (РІРЅРµС€РЅРёР№ РєРѕРЅС‚СѓСЂ + РѕС‚РІРµСЂСЃС‚РёСЏ РїСЂРё РЅРµРѕР±С…РѕРґРёРјРѕСЃС‚Рё)."""
        if not isinstance(polygon, Polygon):
            raise TypeError(f"[polygon_to_path] РћР¶РёРґР°Р»СЃСЏ Polygon, РїРѕР»СѓС‡РµРЅ: {type(polygon).__name__}")

        # Р’РЅРµС€РЅРёР№ РєРѕРЅС‚СѓСЂ (Р±РµР· РїРѕРІС‚РѕСЂРЅРѕР№ РєРѕРЅРµС‡РЅРѕР№ С‚РѕС‡РєРё)
        exterior = np.array(polygon.exterior.coords[:-1], dtype=np.float64)

        # Р•СЃР»Рё РЅСѓР¶РЅРѕ С‚Р°РєР¶Рµ РІРєР»СЋС‡Р°С‚СЊ РѕС‚РІРµСЂСЃС‚РёСЏ:
        interiors = [np.array(interior.coords[:-1], dtype=np.float64) for interior in polygon.interiors]

        return exterior, interiors
    
class util_encoding:
    def __init__(self):
        pass        

    def is_inside(point, poly):
        return poly.contains(Point(point)) or poly.touches(Point(point))
    
    def intersect_rows(y0, point1, point2): 
        y_min, y_max = min(point1[1], point2[1]), max(point1[1], point2[1])
        
        if y0 < y_min or y0 > y_max:
            return None
        
        if point1[1] == point2[1]:
            return None
        
        x = point1[0] + (y0 - point1[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1])

        if y0 == point1[1]:
            return (point1[0], point1[1])
        if y0 == point2[1]:
            return (point2[0], point2[1])

        return (x, y0)
    
    def get_intersections_rows(Y, poly):
        exterior_path, interior_path = util_polygon.polygon_to_path(poly)
        # print(exterior_path)
        temp = []
        for y0 in Y:
            for i in range(len(exterior_path)):
                p1, p2 = exterior_path[i], exterior_path[(i + 1) % len(exterior_path)]
                point = util_encoding.intersect_rows(y0, p1, p2)
                if point and point not in temp:
                    temp.append(point)
                
            for i in range(len(interior_path)):
                p1, p2 = interior_path[i], interior_path[(i + 1) % len(interior_path)]
                point = util_encoding.intersect_rows(y0, p1, p2)
                if point and point not in temp:
                    temp.append(point)
                
        return temp
    
class util_NFP:
    def __init__(self):
        pass
    
    SCALE = 1e6
    
    def polygon_to_path_pycl(polygon):
        """РџСЂРµРѕР±СЂР°Р·СѓРµС‚ shapely Polygon РІ С„РѕСЂРјР°С‚, РїРѕРґС…РѕРґСЏС‰РёР№ РґР»СЏ pyclipper (С†РµР»РѕС‡РёСЃР»РµРЅРЅС‹Рµ РєРѕРѕСЂРґРёРЅР°С‚С‹)."""
        if not isinstance(polygon, Polygon):
            raise TypeError(f"[polygon_to_path] РћР¶РёРґР°Р»СЃСЏ Polygon, РїРѕР»СѓС‡РµРЅ: {type(polygon).__name__}")

        def to_int_coords(ring):
            return [(int(x * util_NFP.SCALE), int(y * util_NFP.SCALE)) for x, y in ring]

        outer = to_int_coords(polygon.exterior.coords[:-1])
        holes = [to_int_coords(interior.coords[:-1]) for interior in polygon.interiors]
        return outer, holes
    
    def minkowski_difference(polygon1, polygon2, anchor_point=(0, 0)):
        SCALE = 1e6
        if not isinstance(polygon1, Polygon):
            raise TypeError(f"[minkowski_difference] polygon1 РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ Polygon, РїРѕР»СѓС‡РµРЅ {type(polygon1).__name__}")
        if not isinstance(polygon2, Polygon):
            raise TypeError(f"[minkowski_difference] polygon2 РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ Polygon, РїРѕР»СѓС‡РµРЅ {type(polygon2).__name__}")

        outer1, _ = util_NFP.polygon_to_path_pycl(polygon1)
        outer2, _ = util_NFP.polygon_to_path_pycl(polygon2)

        ax, ay = int(anchor_point[0] * SCALE), int(anchor_point[1] * SCALE)
        inv_outer1 = [(-x + ax, -y + ay) for x, y in outer1]

        result = pyclipper.MinkowskiSum(outer2, inv_outer1, True)

        polygons = [Polygon([(x / SCALE, y / SCALE) for x, y in path]) for path in result]
        polygons = [p.buffer(0) for p in polygons if p.is_valid and not p.is_empty]

        if not polygons:
            return Polygon()

        return unary_union(polygons)

class util_model:
    def __init__(self):
        pass

    def find_bounding_box_numpy(points):
        points_array = np.array(points, dtype=float)
        if points_array.size == 0:
            raise ValueError("points must be non-empty")
        if points_array.ndim != 2 or points_array.shape[1] != 2:
            raise ValueError("points must have shape (n, 2)")

        xs = points_array[:, 0]
        ys = points_array[:, 1]
        
        return {
            'min_x': np.min(xs),
            'max_x': np.max(xs),
            'min_y': np.min(ys),
            'max_y': np.max(ys),
        }
        
    def parse_items(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        i = 0
        items = []

        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('PIECE'):
                i += 1

                quantity_line = lines[i].strip()
                quantity = int(quantity_line.split()[1])
                i += 1

                vertex_line = lines[i].strip()
                vertex_count = int(vertex_line.split()[3])
                i += 1

                if not lines[i].strip().startswith('VERTICES'):
                    raise ValueError(f"РћР¶РёРґР°Р»Р°СЃСЊ СЃС‚СЂРѕРєР° 'VERTICES', РїРѕР»СѓС‡РµРЅРѕ: {lines[i]}")
                i += 1

                vertices = []
                for _ in range(vertex_count):
                    x_str, y_str = lines[i].strip().split()
                    x, y = float(x_str), float(y_str)
                    vertices.append([x, y])
                    i += 1

                # # РќРѕСЂРјР°Р»РёР·Р°С†РёСЏ РїРѕ РїРµСЂРІРѕР№ С‚РѕС‡РєРµ
                # if len(vertices) > 0:
                #     first_point = vertices[0].copy()  # РЎРѕС…СЂР°РЅСЏРµРј РїРµСЂРІСѓСЋ С‚РѕС‡РєСѓ
                #     normalized_vertices = []
                #     for vertex in vertices:
                #         # Р’С‹С‡РёС‚Р°РµРј РєРѕРѕСЂРґРёРЅР°С‚С‹ РїРµСЂРІРѕР№ С‚РѕС‡РєРё РёР· РІСЃРµС… РІРµСЂС€РёРЅ
                #         normalized_vertex = [vertex[0] - first_point[0], vertex[1] - first_point[1]]
                #         normalized_vertices.append(normalized_vertex)
                    
                shape = np.array(vertices)
                for _ in range(quantity):
                    items.append(shape)

            else:
                i += 1

        return items
    
    def normalize_polygon(poly):
        """
        РЎРґРІРёРіР°РµС‚ РїРѕР»РёРіРѕРЅ С‚Р°Рє, С‡С‚РѕР±С‹ РµРіРѕ bounding box РЅР°С‡РёРЅР°Р»СЃСЏ РІ (0, 0).
        poly: shapely.geometry.Polygon РёР»Рё MultiPolygon
        """
        if poly.is_empty:
            return poly

        minx, miny, maxx, maxy = poly.bounds
        normalized_polygon = translate(poly, xoff=-minx, yoff=-miny)
        return normalized_polygon    
    
    def visualize_solution(self, items, solution, width, height, S, all_items=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        fig, ax = plt.subplots(figsize=(12, 8))

        if solution.get('status') != 'OPTIMAL':
            ax.text(0.5, 0.5, f"РЎС‚Р°С‚СѓСЃ: {solution.get('status')}", ha='center', va='center',
                    transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.3))
            ax.set_xlim(0, width); ax.set_ylim(0, height)
            plt.show()
            return

        h = height / S

        def pts(it):
            if hasattr(it, 'points'): return np.asarray(it.points, dtype=float)
            elif isinstance(it, np.ndarray): return it.astype(float)
            elif isinstance(it, list): return np.asarray(it, dtype=float)
            else: raise TypeError(f"unsupported item type {type(it)}")

        packed_patches = []
        unpacked = []

        for idx, it in enumerate(items):
            deltas = solution.get('deltas', [[0]*S for _ in range(len(items))])[idx]
            strip_idx = next((s for s in range(len(deltas)) if deltas[s] > 0.5), None)
            is_packed = strip_idx is not None

            arr = pts(it)  # <-- Р‘Р•Р— norm()

            if is_packed:
                x = float(solution['x'][idx])
                y = strip_idx * h

                coords = [(float(xx) + x, float(yy) + y) for xx, yy in arr]
                poly = patches.Polygon(coords, closed=True, alpha=0.6)
                packed_patches.append((poly, idx))
            else:
                unpacked.append((arr, idx))

        for poly, idx in packed_patches:
            ax.add_patch(poly)
            ax.text(poly.xy[0][0], poly.xy[0][1], str(idx), color='white')

        # РїРѕР»РѕСЃС‹
        for s in range(S + 1):
            ax.axhline(y=s * h, color='red', linestyle='-', alpha=1, linewidth=1)

        # РіСЂР°РЅРёС†С‹ РєРѕРЅС‚РµР№РЅРµСЂР°
        ax.axvline(x=0, color='black', linewidth=2)
        ax.axvline(x=width, color='black', linewidth=2)
        ax.axhline(y=0, color='black', linewidth=2)
        ax.axhline(y=height, color='black', linewidth=2)

        ax.set_xlim(-width * 0.05, width * 1.05)
        ax.set_ylim(-h * 0.1, height + h * 0.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        plt.show()

