import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from ..core.encoding import Encoding

class Test_Encoding:
    def __init__(self):
        pass
    
    def vis_encoding(poly: Polygon, n: int = None, h: float = None):
        """
        Тестовая визуализация горизонтального кодирования полигона.

        poly : shapely.geometry.Polygon
        n    : количество строк (S) — если указано, высота строки вычисляется как H/n
        h    : высота строки — если указана, количество строк вычисляется как H/h
        """

        # --- 1️⃣ Отрисовка исходного полигона
        x, y = poly.exterior.xy
        plt.plot(x, y, color='gray', marker='o', linestyle='-', alpha=1, label='Полигон')

        # --- 2️⃣ Определяем сетку по Y
        min_y, max_y = poly.bounds[1], poly.bounds[3]
        H = max_y - min_y

        if n is not None:
            S = n
            step = H / S
        elif h is not None:
            step = h
            S = int(np.ceil(H / step))
        else:
            raise ValueError("нужно указать либо n (число строк), либо h (высоту строки)")

        Y = [min_y + step * i for i in range(S + 1)]

        # --- 3️⃣ Кодирование полигона
        segments = Encoding.encode_polygon(poly, Y)

        # --- 4️⃣ Отрисовка горизонталей
        for y_line in Y:
            plt.axhline(y=y_line, color='red', linestyle='--', alpha=0.3)

        # --- 5️⃣ Отрисовка сегментов (результат кодирования)
        for seg in segments:
            (x1, y1), (x2, y2) = seg
            if abs(x1 - x2) < 1e-9:
                # точка
                plt.scatter(x1, y1, color='blue', s=60, alpha=0.7)
            else:
                # отрезок
                plt.plot([x1, x2], [y1, y2], color='blue', linewidth=2, alpha=0.7)

        # --- 6️⃣ Визуальные настройки
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Горизонтальное кодирование полигона (S={S}, step={step:.3f})")
        plt.legend()
        plt.grid(True)
        plt.show()
