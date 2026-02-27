from numpy import array
from shapely.geometry import Polygon
from shapely.ops import unary_union
from utils.helpers import util_NFP, util_model  
import math
import uuid
import numpy as np

class Data:
    def __init__(self, items: list, R: int):
        self.R = R
        self.angle = 360 / R
        self.N = len(items)
        
        if items and isinstance(items[0], Item):
            items_with_rotation, dict_rot = self._get_items_with_rotation(items)
        else:
            items_with_rotation, dict_rot = self._get_items_with_rotation([Item(points) for points in items])

        self.items = items_with_rotation
        self.dict_rot = dict_rot

        for it in self.items:
            it.data = self
        for it in self.items:
            it.compute_nfp()

    def _get_items_with_rotation(self, items):
        temp_dict = {}
        temp_all_items = []

        for it in items:
            temp_dict[it] = []
            # keep the original orientation
            temp_all_items.append(it)
            temp_dict[it].append(it)

            for r in range(1, self.R):
                new_it = Item(it.points.copy())
                new_it.rotation = it.rotation
                new_it.change_rotation(self.angle)
                temp_all_items.append(new_it)
                temp_dict[it].append(new_it)
        return temp_all_items, temp_dict
        
class Item:
    def __init__(self, points: array, data: 'Data' = None):
        self.id = uuid.uuid4() # одинаковый у разных поворотов одного предмета!!!!
        self.nfp = None
        self.rotation = 0

        pts = np.asarray(points, dtype=float)
        anchor = pts[0].copy()
        pts = pts - anchor
        self.points = pts
        
        self.data = data
        self.polygon = Polygon(self.points)
        
        self.area = self.polygon.area

        bbox = util_model.find_bounding_box_numpy(points)
        
        self.xmin = bbox['min_x']
        self.xmax = bbox['max_x'] 
        self.ymin = bbox['min_y']
        self.ymax = bbox['max_y']


    def area(self):
        return self.polygon.area
    
    def change_rotation(self, angle: int):
        rotated_points = []
        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        for x, y in self.points:
            x_rotated = x * cos_angle - y * sin_angle
            y_rotated = x * sin_angle + y * cos_angle
            rotated_points.append((x_rotated, y_rotated))

        self.points = array(rotated_points)
        self.polygon = Polygon(self.points)
        self.area = self.area
        self.rotation = (self.rotation + angle) % 360

    def compute_nfp(self):
        """Вычислить и сохранить NFP — вызывается после присоединения `data` к Item."""
        if self.data is None:
            return
        self.nfp = self.NFP(self.data)

    def outer_NFP(self, other_item, anchor_point=(0, 0)): # TODO Почему anchor_point=(0, 0) а не первая точка в points?
        if not isinstance(self.polygon, Polygon):
            raise TypeError(f"[outer_no_fit_polygon] polygon1 должен быть Polygon, получен {type(self.polygon).__name__}")
        if not isinstance(other_item.polygon, Polygon):
            raise TypeError(f"[outer_no_fit_polygon] polygon2 должен быть Polygon, получен {type(other_item.polygon).__name__}")

        minkowski = util_NFP.minkowski_difference(self.polygon, other_item.polygon, anchor_point)
        base_polygon = other_item.polygon.buffer(0)

        return unary_union([base_polygon, minkowski])
    
    def NFP(self, data):
        anchor_point = self.points[0]
        NFP_dict = {}
        for item in data.items:
            if item.id == self.id:
                continue

            NFP_dict[item] = self.outer_NFP(item, anchor_point)

        return NFP_dict