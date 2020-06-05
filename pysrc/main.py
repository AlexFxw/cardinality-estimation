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
from histogram import Histogram
from matplotlib import pyplot as plt

logger = utils.GetLogger('Main')


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

    def calc_histograms(self, used_keys: list):
        checkpoint_path = f'{utils.checkpoint_dir}/{self.chart_name}.pkl'
        if utils.use_checkpoints:
            if os.path.exists(checkpoint_path):
                print(f'Load cached histogram of table {self.chart_name}')
                with open(checkpoint_path, 'rb') as f:
                    self.histograms = pickle.load(f)
                    for key in self.histograms.keys():
                        self.data_num = self.histograms[key].data_num
                return

        key_index = [i for i, key_attr in enumerate(self.key_attributes) if key_attr.key_label in used_keys]
        logger.info(f'Used keys in calculating the histograms of {self.chart_name}: {used_keys}')

        with open(f'{self.csv_dir}/{self.chart_name}.csv', 'r') as f:
            lines = f.readlines()
            self.data_num = len(lines)

            cache_data = dict()

            with tqdm(total=self.data_num) as pbar:
                pbar.set_description(f'Calculating the histograms of {self.chart_name}')
                for index, line in enumerate(lines):
                    pbar.update()
                    line = line.strip('\n')
                    items = re.split(',', line)
                    if len(items) != self.col_num:
                        continue

                    for i in key_index:
                        item = items[i]
                        if item == '':
                            continue
                        key = self.key_attributes[i].key_label
                        try:
                            cache_data[key].append(np.int32(item))
                        except KeyError:
                            cache_data[key] = list()

            for key in used_keys:
                self.histograms[key] = Histogram(cache_data[key], self.data_num)

            if utils.write_checkpoints:
                print(f'Save the cache histograms of {self.chart_name}')
                with open(checkpoint_path, 'wb') as pkl:
                    pickle.dump(self.histograms, pkl)

    def estimate(self, key_name, comp, value):
        return self.histograms[key_name].query(value, comp)


class TableManager(object):
    def __init__(self, csv_dir: str):
        super().__init__()
        self.tables = dict()
        self.csv_dir = csv_dir

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
            res = Histogram.estimate_overlap(hist_a, hist_b)
            return res

    def calc_histograms(self, used_table_keys: dict):
        for table in used_table_keys.keys():
            used_keys = used_table_keys[table]
            self.tables[table].calc_histograms(used_keys)


class QueryManager(object):
    def __init__(self, file_dir, table_manager: TableManager):
        self.file_dir = file_dir
        self.table_manager = table_manager
        self.used_keys = dict()

    def get_var_cond(self, query):
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
        return variables, conditions

    def precompute_used_keys(self, statements):
        num = len(statements)
        tk_list = []
        with tqdm(total=num) as pbar:
            pbar.set_description(f'Cache all used keys')
            for i, statement in enumerate(statements):
                pbar.update()
                parse_res = sqlparse.parse(statement)
                if len(parse_res) == 0:
                    continue
                query = parse_res[0]
                variables, conditions = self.get_var_cond(query)
                for cond in conditions:
                    items = cond.tokens
                    lv = re.split('\\.', str(items[0]))
                    table_name, key = variables[lv[0]], lv[1]
                    tk_list.append((table_name, key))
                    comp = str(items[1])
                    if comp == '=':
                        rv = items[2]
                        if type(rv) == sqlparse.sql.Identifier:
                            tmp = re.split('\\.', str(rv))
                            rtable_name, r_key = variables[tmp[0]], tmp[1]
                            tk_list.append((rtable_name, r_key))

        for table_name, key in tk_list:
            try:
                self.used_keys[table_name][key] = True
            except KeyError:
                self.used_keys[table_name] = dict()
                self.used_keys[table_name][key] = True

        for table in self.used_keys.keys():
            self.used_keys[table] = list(self.used_keys[table].keys())

    def process(self, file_name, ground_truth: list, limit=None):
        file_path = f'{self.file_dir}/{file_name}'
        ratio_list = []
        with open(file_path, 'r') as f:
            raw_data = f.read()
            statements = sqlparse.split(raw_data)
            self.precompute_used_keys(statements)
            table_manager.calc_histograms(self.used_keys)
            for i, statement in enumerate(statements):
                if limit is not None and i > limit:
                    break
                parse_res = sqlparse.parse(statement)
                if len(parse_res) == 0:
                    continue
                query = parse_res[0]
                variables, conditions = self.get_var_cond(query)
                res = self.estimate(variables, conditions)
                ratio = max(res, ground_truth[i]) / (min(res, ground_truth[i]) + 1e-1)
                print(f'Ratio of query {i + 1} is: {ratio}')
                ratio_list.append(ratio)
        return ratio_list

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
    file_name = 'easy.sql'
    query_manager = QueryManager(query_dir, table_manager)
    res_list = query_manager.process(file_name, ground_truth, 50)
    # print(f'Ratio info, mean: {np.mean(res_list)}, median: {np.median(res_list)}')
    # plt.plot(res_list)
    # plt.show()


if __name__ == '__main__':
    dataDir = '../data'
    csvDir = '../data/clean-imdb'
    # test_table(csvDir)
    # ParseSQL(f'../data/sample_input_homework/easy.sql')
    statements = parser.ParseSQL(f'{csvDir}/schematext.sql')
    table_manager = TableManager(csvDir)
    table_manager.parse(statements)
    test_query_manager(table_manager)
