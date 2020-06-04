import numpy as np
from numba import jit
from dataclasses import dataclass


@dataclass
class HistogramInfo:
    range_start: int
    range_end: int
    scaled: bool
    unit: float
    item_num: int


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
            info = HistogramInfo(range_start, range_end, True, unit, 3 * self.threshold + 2)
        else:
            info = HistogramInfo(range_start, range_end, False, 1, (range_end - range_start) + 2)  # single unit
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


class Histogram(object):
    def __init__(self, info: HistogramInfo):
        super().__init__()
        self.interval = info.item_num
        self.range_end = info.range_end
        self.range_start = info.range_start
        self.scaled = info.scaled
        self.unit = info.unit
        self.data = np.zeros(self.interval, dtype=np.int32)

    def add_data(self, value):
        index = add_data(value, self.range_start, self.range_end, self.unit, self.scaled, self.interval, self.data)
        # self.data[index] = self.data[index] + 1

    def get_index(self, data):
        if self.range_start <= data <= self.range_end:
            if not self.scaled:
                index = data - self.range_start + 1
            else:
                index = (data - self.range_start) * self.unit + 1
                index = int(index)
        else:
            if data < self.range_start:
                index = 0
            elif data > self.range_end:
                index = self.interval - 1
        return index

    def query(self, value, relation: str):
        index = self.get_index(value)
        res = 0
        if relation == '<':
            res = sum(self.data[:index])
        elif relation == '>':
            res = sum(self.data[index:])
        return res

    def query_certain_val(self, value):
        index = self.get_index(value)
        if self.scaled:
            return np.int32(self.data[index] / self.unit)
        return self.data[index]

    @staticmethod
    def estimate_overlap(hist_a, hist_b):
        a_start, a_end = hist_a.range_start, hist_a.range_end
        b_start, b_end = hist_b.range_start, hist_b.range_end
        start, end = max(a_start, b_start), min(a_end, b_end)
        if start > end:
            return 0
        estimation = 0
        for i in range(start, end):
            estimation = estimation + min(hist_a.query_certain_val(i), hist_b.query_certain_val(i))
        return estimation
