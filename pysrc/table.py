from utils import CONFIG, GetLogger, USE_CHECKPOINTS
import utils
import sqlparse
import parser
import numpy as np
from enum import Enum
from dataclasses import dataclass
import re
from tqdm import tqdm
import os
import threading
import pickle

logger = GetLogger('Table')


class Op(Enum):
    GREATER = 0
    LESS = 1
    EQUAL = 2


class AttribueType(Enum):
    CHAR = 0
    INT = 1
    FLOAT = 2
    INVALID = 3


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


@dataclass
class KeyAttribute:
    key_label: str
    attribute_type: AttribueType

    def IsInt(self):
        return self.attribute_type == AttribueType.INT


class Histogram(object):
    def __init__(self, info: HistogramInfo):
        super().__init__()
        self.interval = info.item_num
        self.range_end = info.range_end
        self.range_start = info.range_start
        self.scaled = info.scaled
        self.unit = info.unit
        self.data = np.zeros(self.interval, dtype=np.int32)

    def calc(self, cache_data):
        for data in cache_data:
            index = self.get_index(data)
            self.data[index] = self.data[index] + 1

    def add_data(self, value):
        index = self.get_index(value)
        self.data[index] = self.data[index] + 1

    def get_index(self, data):
        if data < self.range_start:
            index = 0
        elif data > self.range_end:
            index = self.interval - 1
        else:
            if self.scaled:
                index = (data - self.range_start) * self.unit + 1
                index = int(index)
            else:
                index = data - self.range_start + 1
        return index

    def query(self, value, relation: str):
        index = self.get_index(value)
        if relation == '<':
            return sum(self.data[:index])
        elif relation == '>':
            return sum(self.data[index:])

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


class Table(object):
    def __init__(self, key_list: list, type_list: list, csv_dir: str, chart_name: str):
        super().__init__()
        self.data_dir = '../data/imdb/'
        self.col_num = len(key_list)
        self.key_attributes = self.bind_attribute(key_list, type_list)
        self.histograms = dict()
        self.csv_dir = csv_dir
        self.chart_name = chart_name
        self.interval = dict()

    def bind_attribute(self, key_list: list, type_list: list) -> list:
        res = list()
        assert len(key_list) == len(type_list)
        for i in range(0, len(key_list)):
            attribute_type = AttribueType.INVALID
            if type_list[i] == 'character':
                attribute_type = AttribueType.CHAR
            elif type_list[i] == 'integer':
                attribute_type = AttribueType.INT
            res.append(KeyAttribute(key_list[i], attribute_type))
        return res

    def prepare_histogram(self, histogram_util: HistogramUtils):
        # Cache the histogram information
        with open(f'{self.csv_dir}/{self.chart_name}.csv', 'r') as f:
            cache_data = dict()
            cache_lines = f.readlines(CONFIG['CACHE_ITEMS'])
            for index, line in enumerate(cache_lines):
                line = line.strip('\n')
                items = re.split(',', line)
                if len(items) != self.col_num:
                    continue
                for i, item in enumerate(items):
                    if item == '':
                        continue
                    if self.key_attributes[i].IsInt():
                        key = self.key_attributes[i].key_label
                        if key not in cache_data.keys():
                            cache_data[key] = list()
                        val = np.int32(item)
                        cache_data[key].append(val)
            for i, key_attribute in enumerate(self.key_attributes):
                if not key_attribute.IsInt():
                    continue
                key = key_attribute.key_label
                if not histogram_util.info_cached(key) and key in cache_data.keys():
                    histogram_util.compute_info(key, cache_data[key])

    def init_histograms(self, histogram_utils: HistogramUtils):
        for key_attribute in self.key_attributes:
            if not key_attribute.IsInt():  # only deal with int data yet
                continue
            key = key_attribute.key_label
            if histogram_utils.info_cached(key):
                self.histograms[key] = Histogram(histogram_utils.get_info(key))
            else:
                print(f'No cached info for {key} in table {self.chart_name}')

    def calc_histograms(self):
        checkpoint_path = f'{utils.checkpoint_dir}/{self.chart_name}.pkl'
        if utils.USE_CHECKPOINTS:
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'rb') as f:
                    self.histograms = pickle.load(f)
                return

        with open(f'{self.csv_dir}/{self.chart_name}.csv', 'r') as f:
            line = True
            lines = f.readlines()
            line_num = len(lines)
            with tqdm(total=line_num) as pbar:
                pbar.set_description(f'Calculating the histograms of {self.chart_name}')
                for index, line in enumerate(lines):
                    pbar.update()
                    line = line.strip('\n')
                    items = re.split(',', line)
                    if len(items) != self.col_num:
                        continue

                    for i, item in enumerate(items):
                        if item == '':
                            continue
                        if self.key_attributes[i].IsInt():
                            key = self.key_attributes[i].key_label
                            self.histograms[key].add_data(np.int32(item))

            with open(checkpoint_path, 'wb') as pkl:
                pickle.dump(self.histograms, pkl)

    def estimate(self, key_name, comp, value):
        return self.histograms[key_name].query(value, comp)

    def estimate_certain_value(self, key_name, value):
        return self.histograms[key_name].query_certain_val(value)


class TableManager(object):
    def __init__(self, csv_dir: str):
        super().__init__()
        self.tables = dict()
        self.csv_dir = csv_dir
        self.histogram_utils = HistogramUtils()

    def parse(self, raw_statements):
        for raw_statement in raw_statements:
            parse_res = sqlparse.parse(raw_statement)
            if len(parse_res) == 0:
                continue
            statement = parse_res[0]
            cur_table: str = ''
            key_list = list()
            type_list = list()
            for token in statement.tokens:
                if type(token) == sqlparse.sql.Identifier:
                    cur_table = str(token)
                elif type(token) == sqlparse.sql.Parenthesis:
                    key_list = list()
                    type_list = list()
                    for key in token.tokens:
                        # if str(key.ttype) == 'Token.Keyword':
                        #     print(str(key))
                        if type(key) == sqlparse.sql.Identifier:
                            key_list.append(str(key))
                        elif type(key) == sqlparse.sql.IdentifierList:
                            for t in key.tokens:
                                # FIXME: Weird bud, maybe from sqlparse?
                                if type(t) == sqlparse.sql.Identifier or str(t) == 'role' or str(t) == 'link':
                                    key_list.append(str(t))
                        elif str(key) == 'character' or str(key) == 'integer':
                            type_list.append(str(key))
                    logger.debug(f'{cur_table}: {key_list}, {type_list}; {len(key_list)} vs {len(type_list)}')
            self.tables[cur_table] = Table(key_list, type_list, self.csv_dir, cur_table)
            # self.tables[cur_table].calc_histograms()

    def estimate(self, table_name: str, key: str, comp: str, rvalue):
        # for < and >
        if comp == '<' or comp == '>':
            return self.tables[table_name].estimate(key, comp, rvalue)
        else:
            if type(rvalue) == np.int32:
                return self.tables[table_name].estimate_certain_value(key, rvalue)
            else:
                rtable_name, r_key = rvalue
                hist_a = self.tables[table_name].histograms[key]
                hist_b = self.tables[rtable_name].histograms[r_key]
                return Histogram.estimate_overlap(hist_a, hist_b)

    def calc_histograms(self):
        table_list = [key for key in self.tables.keys()]

        with tqdm(total=len(table_list)) as pbar:
            pbar.set_description('Preparing the histogram utils')
            for key in self.tables.keys():
                self.tables[key].prepare_histogram(self.histogram_utils)
                pbar.update()

        for table_name in table_list:
            self.tables[table_name].init_histograms(self.histogram_utils)
            self.tables[table_name].calc_histograms()

        # with tqdm(total=len(table_list)) as pbar:
        #     pbar.set_description('Calculating the histograms of tables')
        #     cnt = 0
        #     threads = []
        #     single_num = int(len(table_list) / 8)
        #     for i in range(0, 7):
        #         t = threading.Thread(target=TableManager.calc_histogram,
        #                              args=(self.tables, table_list[cnt:cnt + single_num], pbar,))
        #         cnt = cnt + single_num
        #         threads.append(t)
        #     t_last = threading.Thread(target=TableManager.calc_histogram, args=(self.tables, table_list[cnt:], pbar,))
        #     threads.append(t_last)
        #     for t in threads:
        #         t.start()
        #     for t in threads:
        #         t.join()

    @staticmethod
    def calc_histogram(tables, key_list, pbar):
        for key in key_list:
            tables[key].calc_histograms()
            pbar.update()


class QueryManager(object):
    def __init__(self, file_dir, table_manager: TableManager):
        self.file_dir = file_dir
        self.table_manager = table_manager

    def process(self, file_name):
        file_path = f'{self.file_dir}/{file_name}'
        with open(file_path, 'r') as f:
            raw_data = f.read()
            statements = sqlparse.split(raw_data)
            query_num = len(statements)
            with tqdm(total=query_num) as pbar:
                pbar.set_description(f'Processing {file_name}')
                for i, statement in enumerate(statements):
                    pbar.update()
                    parse_res = sqlparse.parse(statement)
                    if len(parse_res) == 0:
                        continue
                    query = parse_res[0]
                    var_list = []
                    conditions = []
                    for token in query.get_sublists():
                        if type(token) == sqlparse.sql.Where:
                            # The condition
                            for tk in token.tokens:
                                if type(tk) == sqlparse.sql.Comparison:
                                    conditions.append(tk)
                        elif type(token) == sqlparse.sql.Identifier:
                            # single to select
                            var_list.append(str(token))
                        elif type(token) == sqlparse.sql.IdentifierList:
                            # Multipule to select
                            for tk in token.tokens:
                                if type(tk) == sqlparse.sql.Identifier:
                                    var_list.append(str(tk))
                    variables = dict()
                    for v in var_list:
                        items = re.split(' ', v)
                        variables[items[1]] = items[0]
                    self.estimate(variables, conditions)

    def estimate(self, variables: dict, conditions: list):
        estimation = None
        for cond in conditions:
            items = cond.tokens
            lv = re.split('\\.', str(items[0]))
            lvalue = (variables[lv[0]], lv[1])
            comp = str(items[1])
            if comp == '=':
                # TODO
                rv = items[2]
                if type(rv) == sqlparse.sql.Identifier:
                    # like a.id = b.id
                    tmp = re.split('\\.', str(rv))
                    rvalue = (variables[tmp[0]], tmp[1])
                else:
                    # like a.id = 233
                    rvalue = np.int32(str(rv))
            elif comp == '>' or comp == '<':
                rvalue = np.int32(str(items[2]))
                table_name, key = lvalue
            res = self.table_manager.estimate(table_name, key, comp, rvalue)
            estimation = min(res, estimation) if estimation is not None else res
        return estimation


def test_table(csv_dir):
    key_list = ['id', 'movie_id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'episode_of_id',
                'season_nr', 'episode_nr', 'note', 'md5sum']
    type_list = ['integer', 'integer', 'character', 'character', 'integer', 'integer', 'character', 'integer',
                 'integer', 'integer', 'character', 'character']
    table = Table(key_list, type_list, csv_dir, 'aka_title')
    table.calc_histograms()


def test_query_manager(table_manager: TableManager):
    query_dir = '../data/sample_input_homework'
    file_name = 'test.sql'
    query_manager = QueryManager(query_dir, table_manager)
    query_manager.process(file_name)


if __name__ == '__main__':
    dataDir = '../data'
    csvDir = '../data/clean-imdb'
    # test_table(csvDir)
    # ParseSQL(f'../data/sample_input_homework/easy.sql')
    statements = parser.ParseSQL(f'{csvDir}/schematext.sql')
    table_manager = TableManager(csvDir)
    table_manager.parse(statements)
    table_manager.calc_histograms()
    test_query_manager(table_manager)
