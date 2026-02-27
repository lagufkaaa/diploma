import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from utils.helpers import util_encoding as ue
from utils.helpers import util_polygon as up

class Encoding:
    def __init__(self, data, S, height):
        self.data = data
        self.S = S
        self.height = height
        self.h = height / S

        # k = s_j - s_i
        self.K = list(range(-(S - 1), (S - 1) + 1))
        self.Y_rel = [k * self.h for k in self.K]  # уровни в координатах NFP

        # enc[(i, j, k)] = list of segments [[x1,y],[x2,y]] on y = k*h
        self.enc = self._get_encoding_by_k()

        # k_bounds[(i,j)] = (n_min, n_max)
        self.k_bounds = self._get_k_bounds()

    def _get_k_bounds(self):
        """
        Минимальный вариант: разрешаем все k из self.K для всех пар.
        Если у тебя есть реальные n_min/n_max из геометрии/статей — сюда подставишь их.
        """
        bounds = {}
        for i in self.data.items:
            for j in self.data.items:
                if i.id == j.id:
                    continue
                bounds[(i, j)] = (min(self.K), max(self.K))
        return bounds

    def _encode_one_nfp(self, nfp_geom):
        """
        Возвращает общий список сегментов по всем y из self.Y_rel.
        Если MultiPolygon — объединяем сегменты по частям и мерджим.
        """
        if isinstance(nfp_geom, Polygon):
            segs = Encoding.encode_polygon(nfp_geom, self.Y_rel)
            return segs

        if isinstance(nfp_geom, MultiPolygon):
            all_segs = []
            for poly in nfp_geom.geoms:
                all_segs.extend(Encoding.encode_polygon(poly, self.Y_rel))
            # обязательно слить, чтобы не было дублей/перекрытий
            return Encoding.merge_segments_y(all_segs)

        raise TypeError(f"NFP geometry must be Polygon/MultiPolygon, got {type(nfp_geom)}")

    def _bucket_by_k(self, segs, eps=1e-9):
        """
        segs: список [[x1,y],[x2,y]] для разных y.
        Возвращает dict k -> list of segments for y≈k*h.
        """
        by_k = {k: [] for k in self.K}
        for s in segs:
            y = float(s[0][1])
            k_float = y / self.h
            k_round = int(round(k_float))
            if k_round in by_k and abs(y - k_round * self.h) <= eps:
                # нормализуем порядок x
                x1 = float(min(s[0][0], s[1][0]))
                x2 = float(max(s[0][0], s[1][0]))
                by_k[k_round].append([[x1, k_round * self.h], [x2, k_round * self.h]])

        # слить по каждому k для стабильности
        for k in list(by_k.keys()):
            if by_k[k]:
                by_k[k] = Encoding.merge_segments_y(by_k[k])
        return by_k

    def _get_encoding_by_k(self):
        enc = {}
        for p in self.data.items:
            # ВАЖНО: не .values(), а пары (other -> nfp)
            if not p.nfp:
                continue
            for other, nfp_geom in p.nfp.items():
                if other is None or other.id == p.id:
                    continue
                if nfp_geom is None:
                    continue

                segs_all = self._encode_one_nfp(nfp_geom)
                by_k = self._bucket_by_k(segs_all)

                for k, segs_k in by_k.items():
                    if segs_k:
                        enc[(p, other, k)] = segs_k

        return enc

    def seg_y(T, poly):
        exterior_path, interior_path = up.polygon_to_path(poly)
        seg = []

        for t1 in T:
            for t2 in T:
                # работаем ТОЛЬКО с парами на одной горизонтали
                if t1[1] == t2[1]:
                    # есть ли точки T между t1 и t2 на той же горизонтали?
                    between_same_row = [
                        ((t1[0] < p[0] < t2[0]) or (t2[0] < p[0] < t1[0])) and (p[1] == t1[1])
                        for p in T
                    ]

                    # если промежуточных точек нет — это кандидат на сегмент
                    if between_same_row.count(True) < 1:
                        y = t1[1]
                        px_min, px_max = min(t1[0], t2[0]), max(t1[0], t2[0])

                        # точка-середина для проверки принадлежности полигону
                        point = np.zeros(2)
                        point[0] = (px_min + px_max) / 2.0
                        point[1] = y

                        # если середина внутри полигона — добавляем горизонтальный сегмент
                        if ue.is_inside(point, poly):
                            seg.append([[px_min, y], [px_max, y]])

                        # если середина совпадает с вершиной контура — добавить вырожденный сегмент
                        paths = [exterior_path] + (interior_path or [])  # interior_path: список дыр или []
                        if any(np.all(point == p) for path in paths for p in path):
                            seg.append([point, point])

        return seg

    def trace_rows(poly, Y, seg, mode, eps=1e-9):
        """
        Горизонтальный аналог elp(), адаптированный под твой формат seg.
        Добавляет в seg горизонтальные отрезки (или вырожденные точки как отрезок нулевой длины).

        P    : список вершин полигона [(x, y), ...] по порядку (замкнутого)
        Y    : отсортированный список уровней по y (горизонтальные линии сетки)
        seg  : список, куда мы ДОписываем найденные отрезки:
            каждый элемент формата [[x_start, y0], [x_end, y0]]
            точка хранится как [[x0, y0], [x0, y0]]
        mode : "for" или "inv" — направление обхода рёбер
        eps  : допуск для проверки "точка внутри"
        """
        
        P, inter = up.polygon_to_path(poly)

        for i in range(len(P)):
            # Выбираем ориентацию обхода так же, как в оригинальном elp
            if mode == "inv":
                p1, p2 = P[(i + 1) % len(P)], P[i]
            else:
                p1, p2 = P[i], P[(i + 1) % len(P)]
            
            # print(p1)    
            # print(p2)

            # Найдём интервал по Y, в который попадает p2 по оси y
            band_idx = None
            for j in range(len(Y) - 1):
                if Y[j] <= p2[1] <= Y[j + 1]:
                    band_idx = j
                    break

            if band_idx is None:
                # точка p2 вообще не попадает ни в какую полосу между Y[j] и Y[j+1]
                continue

            # Если p2 не лежит ровно на уровне сетки Y, пробуем построить связи
            # с нижней и верхней границей полосы
            if p2[1] not in Y:
                y_low  = Y[band_idx]
                y_high = Y[band_idx + 1]

                # пересечения ребра с y = y_low и y = y_high
                low_hit  = ue.intersect_rows(y_low,  p1, p2)
                high_hit = ue.intersect_rows(y_high, p1, p2)

                # ОБРАБОТКА НИЖНЕЙ ГРАНИЦЫ ПОЛОСЫ (y = y_low)
                if low_hit:
                    # проверка "снаружи ли точка чуть выше", как в исходном коде,
                    # только теперь мы двигаемся по y, а не по x.
                    test_point_above = [p2[0], p2[1] + eps]
                    if not ue.is_inside(test_point_above, poly):
                        # избегаем "дублировать" вершины полигона,
                        # та же проверка была в твоём elp: `if left not in P:`
                        if low_hit not in P:
                            x_start = min(p2[0], low_hit[0])
                            x_end   = max(p2[0], low_hit[0])
                            y_val   = low_hit[1]  # это y_low
                            # записываем как отрезок по горизонтали
                            seg.append([[x_start, y_val], [x_end, y_val]])

                            # важный момент:
                            # в оригинале ты ещё пихала K_points.append([...])
                            # но ты сказала, что теперь seg хранит и точки тоже,
                            # поэтому сделаем "вырожденный отрезок", чтобы сохранить p2->другая_граница.
                            # на другой границе полосы (верхняя граница y_high),
                            # x остаётся p2[0], y = y_high.
                            seg.append([[p2[0], y_high], [p2[0], y_high]])

                # ОБРАБОТКА ВЕРХНЕЙ ГРАНИЦЫ ПОЛОСЫ (y = y_high)
                if high_hit:
                    test_point_below = [p2[0], p2[1] - eps]
                    if not ue.is_inside(test_point_below, poly):
                        if high_hit not in P:
                            x_start = min(p2[0], high_hit[0])
                            x_end   = max(p2[0], high_hit[0])
                            y_val   = high_hit[1]  # это y_high
                            seg.append([[x_start, y_val], [x_end, y_val]])

                            # симметрично кладём "вырожденный" маркер на нижнюю границу
                            seg.append([[p2[0], y_low], [p2[0], y_low]])

                # СЛУЧАЙ, КОГДА РЕБРО ЦЕЛИКОМ ЛЕЖИТ ВНУТРИ ЭТОЙ ПОЛОСЫ и
                # не пересекло ни нижнюю, ни верхнюю границу
                if not low_hit and not high_hit:
                    # тогда в исходном коде ты добавляла два вертикальных отрезка,
                    # по X[k] и X[k+1]; здесь будет два горизонтальных
                    # по y_low и y_high.
                    x_left  = min(p1[0], p2[0])
                    x_right = max(p1[0], p2[0])

                    seg.append([[x_left,  y_low],  [x_right, y_low]])
                    seg.append([[x_left,  y_high], [x_right, y_high]])

            # Если p2[1] ЛЕЖИТ РОВНО на уровне из Y:
            # В оригинальном elp в этой ветке ничего явно не делается (ветка if t2[0] not in X:),
            # так что здесь мы тоже ничего не добавляем.
            # Если тебе нужно фиксировать такие вершины как точки-сегменты, можно раскомментировать:
            #
            # else:
            #     y_val = p2[1]
            #     seg.append([[p2[0], y_val], [p2[0], y_val]])
            #
            # но я оставляю поведение эквивалентным оригиналу.

    def merge_segments_y(seg, eps=1e-12):
        """
        Принимает список `seg`, где каждый элемент — это или горизонтальный отрезок,
        или точка, представленная тем же форматом:
            [[x_start, y0], [x_end, y0]]
        (у точки x_start == x_end и одинаковый y0)

        Делает:
        1. Разделяет интервалы и точки.
        2. Сливает пересекающиеся/соприкасающиеся интервалы с одинаковым y.
        3. Удаляет точки, которые и так покрыты каким-то итоговым интервалом.
        4. Возвращает новый список в том же формате:
            [[x1, y], [x2, y]]
        где точки тоже представлены как вырожденные отрезки.
        """

        # 1. Разбиваем seg на интервальные куски и одиночные точки
        intervals_by_y = {}  # y -> list of (x1, x2)
        points_by_y = {}     # y -> list of x

        for s in seg:
            (x1, y1), (x2, y2) = s[0], s[1]

            # безопасно на случай плавающей точки: считаем, что y1==y2
            y_val = y1  # предполагаем горизонтальный отрезок

            left_x = min(x1, x2)
            right_x = max(x1, x2)

            if abs(left_x - right_x) <= eps:
                # это точка
                points_by_y.setdefault(y_val, []).append(left_x)
            else:
                # это интервал
                intervals_by_y.setdefault(y_val, []).append((left_x, right_x))

        # 2. Для каждого y сольём все интервалы (как ты делала для вертикалей)
        merged_by_y = {}  # y -> list of (L, R) несмещающихся, слитых
        for y_val, spans in intervals_by_y.items():
            # отсортируем по левому концу
            spans.sort(key=lambda ab: ab[0])

            merged = []
            for (cur_l, cur_r) in spans:
                if not merged:
                    merged.append([cur_l, cur_r])
                else:
                    prev_l, prev_r = merged[-1]

                    # если пересекается или соприкасается (prev_r >= cur_l)
                    if prev_r + eps >= cur_l:
                        # расширяем последний интервал вправо
                        merged[-1][1] = max(prev_r, cur_r)
                    else:
                        merged.append([cur_l, cur_r])

            merged_by_y[y_val] = merged

        # 3. Фильтруем точки:
        #    выкинуть точку, если она уже лежит внутри одного из merged интервалов на том же y
        kept_points_by_y = {}
        for y_val, xs in points_by_y.items():
            merged_spans = merged_by_y.get(y_val, [])
            kept = []
            for x0 in xs:
                covered = any((span_l - eps <= x0 <= span_r + eps) for (span_l, span_r) in merged_spans)
                if not covered:
                    # не даём дубликатов той же точки
                    if not any(abs(x0 - x_prev) <= eps for x_prev in kept):
                        kept.append(x0)
            if kept:
                kept_points_by_y[y_val] = kept

        # 4. Собираем обратно в формат [[x1,y],[x2,y]]
        result = []

        # сначала добавляем слитые интервалы
        for y_val, spans in merged_by_y.items():
            for (l, r) in spans:
                result.append([[l, y_val], [r, y_val]])

        # потом добавляем оставшиеся точки как вырожденные отрезки
        for y_val, xs in kept_points_by_y.items():
            for x0 in xs:
                result.append([[x0, y_val], [x0, y_val]])

        # Можно отсортировать результат для стабильности:
        # сначала по y, потом по x_start, потом по x_end
        result.sort(key=lambda seg_item: (seg_item[0][1], seg_item[0][0], seg_item[1][0]))

        return result

    
    def encode_polygon(poly, Y):
        """
        Горизонтальное кодирование полигона:
        разбивает фигуру poly (shapely Polygon) на S горизонтальных интервалов по высоте H.

        poly : shapely.geometry.Polygon
        S    : int — количество строк (горизонталей)
        H    : float — общая высота (например, высота области/детали)

        Возвращает список сегментов:
            [[(x1, y), (x2, y)]] — горизонтальные отрезки внутри полигона.
            Точки (касаний) хранятся как [[(x0, y0), (x0, y0)]].
        """

        if len(Y) <= 0:
            raise ValueError("количество строк не может быть меньше 1")
        if not isinstance(poly, Polygon):
            raise TypeError("poly должен быть объектом shapely.geometry.Polygon")

        # 1️⃣ формируем горизонтальные уровни
        # уже передаются в функцию

        # 2️⃣ разбираем полигон на внешнюю и внутренние границы
        exterior = Polygon(poly.exterior.coords)

        # 3️⃣ получаем все пересечения горизонталей с полигоном (отрезки внутри полигона)
        inter_rows = ue.get_intersections_rows(Y, exterior)
        # print(inter_rows)

        # 4️⃣ строим базовые сегменты по этим пересечениям
        seg = []

        if isinstance(inter_rows, dict):
            # inter_rows = { y: [(x1, x2), ...], ... }
            for y, spans in inter_rows.items():
                for (x1, x2) in spans:
                    seg.append([[x1, y], [x2, y]])

        else:
            # inter_rows = [(x, y), (x, y), ...]
            # сгруппируем точки по y, отсортируем x и превратим в интервалы
            by_y = {}
            for x, y in inter_rows:
                by_y.setdefault(y, []).append(x)

            for y, xs in by_y.items():
                xs.sort()
                i = 0
                while i < len(xs):
                    if i + 1 < len(xs):
                        x1, x2 = xs[i], xs[i + 1]
                        seg.append([[x1, y], [x2, y]])
                        i += 2
                    else:
                        # нечётное число пересечений на строке -> касание
                        x0 = xs[i]
                        seg.append([[x0, y], [x0, y]])
                        i += 1

        # 5️⃣ достраиваем сегменты вдоль рёбер (аналог elp, но по y)
        Encoding.trace_rows(exterior, Y, seg, "for")
        Encoding.trace_rows(exterior, Y, seg, "inv")

        # 6️⃣ сливаем и очищаем результат
        merged = Encoding.merge_segments_y(seg)
        
        # print(merged)

        return merged
