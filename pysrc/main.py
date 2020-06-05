from utils import CONFIG, GetLogger, USE_CHECKPOINTS
import utils
import sqlparse
import parser
import numpy as np
from enum import Enum
from dataclasses import dataclass
import re
import gc
from tqdm import tqdm
import os
from numba import jit
import pickle
from histogram import HistogramInfo, HistogramUtils, HistogramId, HistogramInt

logger = GetLogger('Main')


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
class KeyAttribute:
    key_label: str
    attribute_type: AttribueType

    def IsInt(self):
        return self.attribute_type == AttribueType.INT


@jit(nopython=True)
def separate_csv(line):
    line = line.strip('\n')
    return re.split(',', line)


def process_line(lines, ):
    pass


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
        self.data_num = 0

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
            if self.chart_name == 'cast_info' and key == 'id':
                info = HistogramInfo(0, 40000000, True, 0.01, 400000)
            else:
                info = HistogramInfo(0, 4000000, True, 0.1, 400000)
            self.histograms[key] = Histogram(info)

    def calc_histograms(self):
        checkpoint_path = f'{utils.checkpoint_dir}/{self.chart_name}.pkl'
        if utils.USE_CHECKPOINTS:
            if os.path.exists(checkpoint_path):
                print(f'Load cached histogram of table {self.chart_name}')
                with open(checkpoint_path, 'rb') as f:
                    self.histograms = pickle.load(f)
                    self.data_num = self.histograms['id'].data_num  # FIXME: No need after recomputing
                return

        with open(f'{self.csv_dir}/{self.chart_name}.csv', 'r') as f:
            lines = f.readlines()
            data_num = len(lines)
            self.data_num = data_num
            for i, key_attribute in enumerate(self.key_attributes):
                key = key_attribute.key_label
                self.histograms[key] = HistogramId(data_num)

            with tqdm(total=data_num) as pbar:
                pbar.set_description(f'Calculating the histograms of {self.chart_name}')
                for index, line in enumerate(lines):
                    pbar.update()
                    line = line.strip('\n')
                    items = re.split(',', line)
                    if len(items) != self.col_num:
                        continue

                    for i, key_attribute in enumerate(self.key_attributes):
                        key = key_attribute.key_label
                        item = items[i]
                        if item == '':
                            continue
                        if key_attribute.IsInt():
                            self.histograms[key].add_data(np.int32(item))

            if utils.write_checkpoints:
                with open(checkpoint_path, 'wb') as pkl:
                    pickle.dump(self.histograms, pkl)

    def estimate(self, key_name, comp, value):
        return self.histograms[key_name].query(value, comp)


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
                        if type(key) == sqlparse.sql.Identifier:
                            key_list.append(str(key))
                        elif type(key) == sqlparse.sql.IdentifierList:
                            for t in key.tokens:
                                # FIXME: Weird bud, maybe from sqlparse?
                                if type(t) == sqlparse.sql.Identifier or str(t) == 'role' or str(t) == 'link':
                                    key_list.append(str(t))
                        elif str(key) == 'character' or str(key) == 'integer':
                            type_list.append(str(key))
                    logger.debug(
                        f'{cur_table}: {key_list}, {type_list}; {len(key_list)} vs {len(type_list)}')
            self.tables[cur_table] = Table(
                key_list, type_list, self.csv_dir, cur_table)

    def estimate(self, table_name: str, key: str, comp: str, rvalue):
        # for < and >
        if comp == '<' or comp == '>' or (comp == '=' and type(rvalue) == np.int32):
            res = self.tables[table_name].estimate(key, comp, rvalue)
            return res
        else:
            rtable_name, r_key = rvalue
            hist_a = self.tables[table_name].histograms[key]
            hist_b = self.tables[rtable_name].histograms[r_key]
            res = HistogramId.estimate_overlap(hist_a, hist_b)
            return res

    def calc_histograms(self):
        table_list = [key for key in self.tables.keys()]

        for table_name in table_list:
            self.tables[table_name].calc_histograms()
            gc.collect()

    @staticmethod
    def calc_histogram(tables, key_list):
        for key in key_list:
            tables[key].calc_histograms()


class QueryManager(object):
    def __init__(self, file_dir, table_manager: TableManager):
        self.file_dir = file_dir
        self.table_manager = table_manager

    def process(self, file_name, ground_truth: list):
        file_path = f'{self.file_dir}/{file_name}'
        with open(file_path, 'r') as f:
            raw_data = f.read()
            statements = sqlparse.split(raw_data)
            for i, statement in enumerate(statements):
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
                res = self.estimate(variables, conditions)
                print(
                    f'Estimation / Ground Truth of query {i + 1} is: {res} vs {ground_truth[i]}, {res / ground_truth[i]}')

    def estimate(self, variables: dict, conditions: list):
        estimation = 1
        tables = dict()
        for cond in conditions:
            items = cond.tokens
            lv = re.split('\\.', str(items[0]))
            table_name, key = variables[lv[0]], lv[1]
            tables[table_name] = True
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
            res = self.table_manager.estimate(table_name, key, comp, rvalue)
            estimation = estimation * res

        for table in tables.keys():
            estimation = estimation * self.table_manager.tables[table].data_num
        return int(estimation)


def test_table(csv_dir):
    key_list = ['id', 'movie_id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'episode_of_id',
                'season_nr', 'episode_nr', 'note', 'md5sum']
    type_list = ['integer', 'integer', 'character', 'character', 'integer', 'integer', 'character', 'integer',
                 'integer', 'integer', 'character', 'character']
    table = Table(key_list, type_list, csv_dir, 'aka_title')
    table.calc_histograms()


def load_true_rows(data_dir, level_name):
    answers = []
    with open(f'{data_dir}/{level_name}.normal', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            answers.append(np.int32(line))
    return answers


def test_query_manager(table_manager: TableManager):
    query_dir = '../data/sample_input_homework'
    true_dir = '../data/true_rows'
    ground_truth = load_true_rows(true_dir, 'easy')
    file_name = 'test.sql'
    query_manager = QueryManager(query_dir, table_manager)
    query_manager.process(file_name, ground_truth)


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
