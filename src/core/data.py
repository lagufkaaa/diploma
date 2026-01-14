from numpy import array
from shapely.geometry import Polygon
from shapely.ops import unary_union
from src.utils.helpers import util_NFP
import math
import uuid

class Data:
    def __init__(self, items: list, R: int):
        self.R = R
        self.angle = 360 / R
        self.N = len(items)

        # `items` может быть списком готовых `Item` или списком "сырых" точек.
        if items and isinstance(items[0], Item):
            items_with_rotation = self._get_items_with_rotation(items)
        else:
            # создаём объекты Item из сырых точек (без вычисления NFP)
            items_with_rotation = self._get_items_with_rotation([Item(points) for points in items])

        self.items = items_with_rotation

        # привязываем ссылку на Data и вычисляем NFP после создания всех Item
        for it in self.items:
            it.data = self
        for it in self.items:
            it.compute_nfp()

    def _get_items_with_rotation(self, items):
        temp = []
        for it in items:
            temp.append(it)
            for r in range(self.R):
                it.change_rotation(self.angle)
                temp.append(it)
        return temp
    
class Item:
    def __init__(self, points: array, data: 'Data' = None):
        self.id = uuid.uuid4() # одинаковый у разных поворотов одного предмета!!!!
        self.points = points
        self.polygon = Polygon(points)
        self.area = self.polygon.area
        self.nfp = None
        self.rotation = 0
        self.data = data

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