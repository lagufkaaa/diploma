import sys
import os
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, MultiPolygon

# Ensure 'src' is on sys.path so we can import the package modules
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from core.data import Data, Item
from core.encoding import Encoding
from utils.helpers import util_model


DATA_DIR = Path(__file__).resolve().parents[2] / 'data_car_mats'


def test_parse_items_and_data_items():
    file_path = DATA_DIR / 'test.txt'
    items = util_model.parse_items(str(file_path))
    assert isinstance(items, list)
    assert len(items) > 0

    # limit the number of shapes to avoid heavy computations during tests
    items = items[:5]

    data = Data(items, R=0)
    assert isinstance(data.items, list)
    assert all(isinstance(it, Item) for it in data.items)
    # each item should have a reference back to data and an nfp dict (possibly empty)
    for it in data.items:
        assert it.data is data
        assert it.nfp is None or isinstance(it.nfp, dict)


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
