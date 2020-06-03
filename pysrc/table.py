import parser
import utils
from numba import jit
import sqlparse
import parser
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass

logger = utils.GetLogger('Table')


class AttribueType(Enum):
    STRING = 0
    INT = 1
    FLOAT = 2


@dataclass
class KeyAttribute:
    key_label: str
    attribute_type: AttribueType


class Histogram(object):
    def __init__(self, col_num):
        super().__init__()
        self.histogram_size = 256
        self.col_num = col_num
        self.data = np.zeros((col_num, self.histogram_size), dtype=np.int32)

    def calc_csv_histogram(self, file_path: str):
        df = pd.read_csv(file_path)
        print(df.head())


class Table(object):
    def __init__(self, key_list: list, type_list: list):
        super().__init__()
        self.data_dir = '../data/imdb/'
        self.col_num = len(key_list)
        self.key_list = key_list
        self.type_list = type_list


class TableManager(object):
    def __init__(self):
        super().__init__()
        self.tables = dict()

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
                        if type(key) == sqlparse.sql.Identifier:
                            key_list.append(str(key))
                        elif type(key) == sqlparse.sql.IdentifierList:
                            for t in key.tokens:
                                if type(t) == sqlparse.sql.Identifier:
                                    key_list.append(str(t))
                        elif str(key) == 'character' or str(key) == 'integer':
                            type_list.append(str(key))
                    logger.debug(f'{cur_table}: {key_list}, {type_list}; {len(key_list)} vs {len(type_list)}')
            self.tables[cur_table] = Table(key_list, type_list)


if __name__ == '__main__':
    dataDir = '../data'
    # ParseSQL(f'../data/sample_input_homework/easy.sql')
    statements = parser.ParseSQL(f'{dataDir}/imdb/schematext.sql')
    table_manager = TableManager()
    table_manager.parse(statements)
