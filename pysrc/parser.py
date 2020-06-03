'''
@Author: Hsuan-Wei Fan
@Date: 2020-06-03 15:04:44
@LastEditors: Hsuan-Wei Fan
@LastEditTime: 2020-06-03 15:34:15
@Description: 
'''

import sqlparse
from utils import GetLogger

logger = GetLogger('parser')

def ParseSQL(fileName: str):
    statements = None
    with open(fileName, 'r') as f:
        raw = f.read()
        statements = sqlparse.split(raw)
    return statements

