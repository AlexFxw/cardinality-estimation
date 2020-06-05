import numpy as np
from numba import jit, jitclass, int32
import numba as nb
from dataclasses import dataclass
from enum import Enum
import collections


@dataclass
class HistogramInfo:
    range_start: int
    range_end: int
    scaled: bool
    unit: float
    item_num: int


class AttributeType(Enum):
    CHAR = 0
    INT = 1
    PRIMARY = 2


histogram_config = {
    'cast_info': {
        'id': HistogramInfo(0, 40000000, True, 0.01, 400000),
    }
}


class HistogramUtils:
    def __init__(self):
        self.threshold = 100000
        self.info = dict()  # Used to record the histogram's information.

    def compute_info(self, key, raw_data: list):
        max_val = max(raw_data)
        min_val = min(raw_data)
        interval = max_val - min_val
        range_start = max(min_val - interval, 0)
        range_end = max_val + interval
        if interval > self.threshold:
            unit = (3 * self.threshold) / (range_end - range_start)
            info = HistogramInfo(range_start, range_end,
                                 True, unit, 3 * self.threshold + 2)
        else:
            info = HistogramInfo(range_start, range_end, False,
                                 1, (range_end - range_start) + 2)  # single unit
        self.info[key] = info

    def info_cached(self, key):
        if key in self.info.keys():
            return True
        return False

    def get_info(self, key):
        return self.info[key]


@jit(nopython=True, signature_or_function="void(int32,int32,int32,float32,boolean,int32,int32[:])")
def add_data(value, start, end, unit, scaled, interval, data):
    if start <= value <= end:
        if not scaled:
            index = value - start + 1
        else:
            index = (value - start) * unit + 1
            index = int(index)
    else:
        if value < start:
            index = 0
        elif value > end:
            index = interval - 1
    data[index] = data[index] + 1


class TestHistogram:
    def __init__(self, attr_type: AttributeType):
        self.type = attr_type
        self.min_val = None
        self.max_val = None
        self.data = dict()
        self.contained_value = []
        self.refracted_data = []
        self.item_num = 0

    def add_data(self, value):
        if self.type == AttributeType.INT:
            self.max_val = max(
                self.max_val, value) if self.max_val is not None else value
            self.min_val = min(
                self.min_val, value) if self.min_val is not None else value
            if value not in self.data.keys():
                self.data[value] = 1
                self.contained_value.append(value)
            else:
                self.data[value] = self.data[value] + 1

    def refract_data(self):
        if self.type == AttributeType.INT:
            self.item_num = len(self.contained_value)
            self.contained_value = sorted(self.contained_value)
            for item in self.contained_value:
                self.refracted_data.append(self.data[item])
            self.contained_value = np.array(self.contained_value)
            self.refracted_data = np.array(self.refracted_data)
            del self.data

    def query(self, value, relation):
        res = 0
        if relation == '=':
            for i in range(0, self.item_num):
                v = self.contained_value[i]
                if v == value:
                    return self.refracted_data[i]
                if v > value:
                    return 0
        elif relation == '<':
            for i in range(0, self.item_num):
                if self.contained_value[i] < value:
                    res = res + self.refracted_data[i]
                else:
                    break
        elif relation == '>':
            for i in range(self.item_num, -1, -1):
                if self.contained_value[i] > value:
                    res = res + self.refracted_data[i]
                else:
                    break
        return res

    @staticmethod
    def estimate_overlap(hist_a, hist_b):
        a_min, a_max = hist_a.min_val, hist_a.max_val
        b_min, b_max = hist_b.min_val, hist_b.max_val
        min_v = max(a_min, b_min)
        max_v = min(a_max, b_max)
        res = 0
        for v in range(min_v, max_v + 1):
            av = hist_a.query(v, '=')
            bv = hist_b.query(v, '=')
            res = res + min(av, bv)
        return res


# class Histogram(object):
#     def __init__(self, info: HistogramInfo):
#         super().__init__()
#         self.interval = info.item_num + 2
#         self.range_end = info.range_end
#         self.range_start = info.range_start
#         self.scaled = info.scaled
#         self.unit = info.unit
#         self.data = np.zeros(self.interval, dtype=np.int32)
#
#     def add_data(self, value):
#         index = add_data(value, self.range_start, self.range_end, self.unit, self.scaled, self.interval, self.data)
#         # self.data[index] = self.data[index] + 1
#
#     def get_index(self, data):
#         if self.range_start <= data <= self.range_end:
#             if not self.scaled:
#                 index = data - self.range_start + 1
#             else:
#                 index = (data - self.range_start) * self.unit + 1
#                 index = int(index)
#         else:
#             if data < self.range_start:
#                 index = 0
#             elif data > self.range_end:
#                 index = self.interval - 1
#         return index
#
#     def query(self, value, relation: str):
#         index = self.get_index(value)
#         res = 0
#         if relation == '<':
#             res = sum(self.data[:index])
#         elif relation == '>':
#             index = min(index + 1, self.interval - 1)
#             res = sum(self.data[index:])
#         return res
#
#     def query_certain_val(self, value):
#         index = self.get_index(value)
#         if self.scaled:
#             return np.int32(self.data[index] * self.unit)
#         return self.data[index]
#
#     def calc_histogram(self, raw_data):
#         np_data = np.array(raw_data)
#
#
#     @staticmethod
#     def estimate_overlap(hist_a, hist_b):
#         a_start, a_end = hist_a.range_start, hist_a.range_end
#         b_start, b_end = hist_b.range_start, hist_b.range_end
#         start, end = max(a_start, b_start), min(a_end, b_end)
#         print(f'  start, end: ({start}, {end})')
#         if start > end:
#             return 0
#         estimation = 0
#         for i in range(start, end + 1):
#             estimation = estimation + min(hist_a.query_certain_val(i), hist_b.query_certain_val(i))
#         return estimation

@jit(signature_or_function="void(int32,int32,int32,int32[:])", nopython=True)
def add_id(value, interval, item_num, num_counts):
    index = int(value / interval)
    index = min(index, item_num - 1)
    num_counts[index] = num_counts[index] + 1


class HistogramBase:
    def __init__(self, data_num):
        super().__init__()
        self.data_num = data_num

    def calc_data(self, raw_data: list):
        pass

    def add_data(self, value):
        pass

    def get_range(self):
        return None, None


class HistogramId(HistogramBase):
    def __init__(self, data_num):
        super().__init__(data_num)
        self.id_threshold = 50000000
        self.interval = 64
        self.interval_half = 32
        self.item_num = int(self.id_threshold / self.interval)
        self.items = np.zeros(self.item_num, dtype=np.int32)

    def add_data(self, value):
        add_id(value, self.interval, self.item_num, self.items)

    def query(self, value: int, relation: str):
        index = int(value / self.interval)
        index = min(self.item_num - 1, index)
        if relation == '=':
            res = self.items[index] / self.interval
        elif relation == '<':
            res = sum(self.items[:index + 1])
        else:
            res = sum(self.items[index:])
        return res / self.data_num

    @staticmethod
    def estimate_overlap(hist_a, hist_b):
        assert hist_a.item_num == hist_b.item_num
        num = hist_a.item_num
        res = 0
        a_num_inv = 1 / hist_a.data_num
        b_num_inv = 1 / hist_b.data_num
        for i in range(0, num):
            res = res + (hist_a.items[i] * a_num_inv) * (hist_b.items[i] * b_num_inv)
        return res


@jit(nopython=True)
def add_data(value, data_dict):
    try:
        data_dict[value] = data_dict[value] + 1
    except KeyError:
        data_dict[value] = 1


class HistogramInt(HistogramBase):
    def __init__(self, data_num):
        super().__init__(data_num)
        # self.data = nb.typed.Dict.empty(
        #     key_type=nb.types.int32,
        #     value_type=nb.types.int32
        # )
        self.data = dict()
        self.item_num = 0
        self.sorted_items = None

    def calc_data(self):
        # Used for non-primary key
        # dc = collections.Counter(raw_data)
        # sorted_items = sorted(dc.keys())
        # self.item_num = len(sorted_items)
        # self.sorted_items = np.array(sorted_items, dtype=np.int32)
        # self.num_counts = np.zeros(self.item_num, dtype=np.int32)
        # for i, key in enumerate(sorted_items):
        #     self.indices[key] = i
        #     self.num_counts[i] = dc[key]
        sorted_items = sorted(self.data.keys())
        self.item_num = len(sorted_items)
        print(f'  datanum: {self.item_num}')
        self.sorted_items = np.array(sorted_items, dtype=np.int32)

    def add_data(self, value):
        try:
            self.data[value] = self.data[value] + 1
        except KeyError:
            self.data[value] = 1

    def query(self, value: int, relation: str):
        res = 0
        if relation == '=':
            if value not in self.data.keys():
                return 0
            res = self.data[value]
        elif relation == '<':
            for item in self.sorted_items:
                if item >= value:
                    break
                res = res + self.data[item]
        elif relation == '>':
            for item in self.sorted_items:
                if item <= value:
                    break
                res = res + self.data[item]
        return res / self.data_num

    def get_range(self):
        return self.sorted_items[0], self.sorted_items[-1]

    @staticmethod
    def estimate_overlap(hist_a, hist_b):
        a_start, a_end = hist_a.get_range()
        b_start, b_end = hist_b.get_range()
        start, end = max(a_start, b_start), min(a_end, b_end)
        if start > end:
            return 0
        estimation = 0
        for i in range(start, end + 1):
            estimation = estimation + \
                         min(hist_a.query(i, '='), hist_b.query(i, '='))
        return estimation
