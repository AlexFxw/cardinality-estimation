import numpy as np
from numba import jit, jitclass, int32
import numba as nb
from dataclasses import dataclass
from enum import Enum
import collections


class AttributeType(Enum):
    CHAR = 0
    INT = 1
    PRIMARY = 2


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


# class HistogramId(HistogramBase):
#     def __init__(self, data_num):
#         super().__init__(data_num)
#         self.id_threshold = 50000000
#         self.interval = 64
#         self.interval_half = 32
#         self.item_num = int(self.id_threshold / self.interval)
#         self.items = np.zeros(self.item_num, dtype=np.int32)
#
#     def add_data(self, value):
#         add_id(value, self.interval, self.item_num, self.items)
#
#     def query(self, value: int, relation: str):
#         index = int(value / self.interval)
#         index = min(self.item_num - 1, index)
#         if relation == '=':
#             res = self.items[index] / self.interval_half
#         elif relation == '<':
#             res = sum(self.items[:index + 1])
#         else:
#             res = sum(self.items[index:])
#         return res / self.data_num
#
#     @staticmethod
#     def estimate_overlap(hist_a, hist_b):
#         return calc_overlap(hist_a.item_num, hist_a.items, hist_a.data_num, hist_b.items, hist_b.data_num)

@jit(nopython=True,
     signature_or_function="float32(int32[:],int32[:],int32,int32,int32[:],int32[:],int32,int32)")
def calc_overlap(a_values, a_counts, a_data_num, a_item_num,
                 b_values, b_counts, b_data_num, b_item_num):
    res = 0
    a_num_inv = 1 / float(a_data_num)
    b_num_inv = 1 / float(b_data_num)
    start = max(a_values[0], b_values[0])
    end = min(a_values[-1], b_values[-1])
    if start >= end:
        return 0
    iter = 0
    for i in range(0, a_item_num):
        a = a_values[i]
        if a < start:
            continue
        if a > end:
            break
        for k in range(iter, b_item_num):
            if b_values[k] == a:
                res = res + (a_counts[i] * a_num_inv) * (b_counts[k] * b_num_inv)
                iter = k
                break
            elif b_values[k] > a:
                iter = max(k-1, 0)
                break
    return res


class Histogram:
    def __init__(self, raw_data: list, data_num: int):
        dh = collections.Counter(raw_data)
        sort_dh = sorted(dh.keys())
        values, counts = list(), list()
        for v in sort_dh:
            values.append(v)
            counts.append(dh[v])
        self.values = np.array(values, dtype=np.int32)
        self.counts = np.array(counts, dtype=np.int32)
        self.data_num = data_num
        self.item_num = len(values)

    def query(self, value: int, relation: str):
        if relation == '=':
            index = np.where(self.values == value)
            res = self.counts[index] if len(index) > 0 else 0
        else:
            if relation == '<':
                items = np.argwhere(self.values < value)
            else:
                items = np.argwhere(self.values > value)
            cc = [self.counts[item[0]] for item in items.tolist()]
            res = sum(cc)
        return res / self.data_num

    @staticmethod
    def estimate_overlap(hist_a, hist_b):
        res = calc_overlap(hist_a.values, hist_a.counts, hist_a.data_num, hist_a.item_num,
                            hist_b.values, hist_b.counts, hist_b.data_num, hist_b.item_num)
        return res
