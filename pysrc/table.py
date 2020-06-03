import parser
import utils
from numba import jit
import sqlparse
import parser
import numpy as np
from enum import Enum
from dataclasses import dataclass
import re
from tqdm import tqdm
import pandas as pd
import sys

logger = utils.GetLogger('Table')


class AttribueType(Enum):
    CHAR = 0
    INT = 1
    FLOAT = 2
    INVALID = 3


@dataclass
class KeyAttribute:
    key_label: str
    attribute_type: AttribueType

    def IsInt(self):
        return self.attribute_type == AttribueType.INT


class Histogram(object):
    def __init__(self, max_value, min_value):
        super().__init__()
        # FIXME: If range is too wide, do not do so.
        assert max_value >= min_value
        self.threshold = 100000
        self.histogram_size = min(max_value - min_value, self.threshold)  # TODO: Parameterize the threshold
        self.need_scale = False if self.histogram_size == (max_value - min_value) else True
        self.max_value = max_value
        self.min_value = min_value
        self.interval = self.histogram_size / (max_value - min_value)
        self.data = np.zeros(self.histogram_size + 2, dtype=np.int32)
        self.histogram = dict()

    def calc(self, cache_data):
        for data in cache_data:
            if data < self.min_value:
                index = 0
            elif data > self.max_value:
                index = self.histogram_size - 1
            else:
                if self.need_scale:
                    index = (data - self.min_value) * self.interval
                    index = int(index)
                else:
                    index = data - self.min_value + 1
            self.data[index] = self.data[index] + 1


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

    def calc_histograms(self):
        # raw_data = pd.read_csv(f'{self.csv_dir}/{self.chart_name}.csv', header=None)
        max_values = [None for i in range(0, self.col_num)]
        min_values = [None for i in range(0, self.col_num)]
        cache_data = dict()

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
                            if key not in cache_data.keys():
                                cache_data[key] = list()
                            val = np.int32(item)
                            cache_data[key].append(val)
                            max_values[i] = max(max_values[i], val) if max_values[i] is not None else val
                            min_values[i] = min(min_values[i], val) if min_values[i] is not None else val
        for i in range(0, self.col_num):
            if self.key_attributes[i].IsInt():
                key = self.key_attributes[i].key_label
                self.histograms[key] = Histogram(max_values[i], min_values[i])
                self.histograms[key].calc(cache_data[key])


class TableManager(object):
    def __init__(self, csv_dir: str):
        super().__init__()
        self.tables = dict()
        self.csv_dir = csv_dir

    def parse(self, raw_statements):
        for raw_statement in raw_statements:
            statement = sqlparse.parse(raw_statement)[0]
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


def test_table(csv_dir):
    key_list = ['id', 'movie_id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'episode_of_id',
                'season_nr', 'episode_nr', 'note', 'md5sum']
    type_list = ['integer', 'integer', 'character', 'character', 'integer', 'integer', 'character', 'integer',
                 'integer', 'integer', 'character', 'character']
    table = Table(key_list, type_list, csv_dir, 'aka_title')
    table.calc_histograms()


if __name__ == '__main__':
    dataDir = '../data'
    csvDir = '../data/clean-imdb'
    test_table(csvDir)
    # ParseSQL(f'../data/sample_input_homework/easy.sql')
    # statements = parser.ParseSQL(f'{dataDir}/imdb/schematext.sql')
    # table_manager = TableManager(csvDir)
    # table_manager.parse(statements)
