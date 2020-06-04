import sqlparse
from utils import GetLogger
from enum import Enum

logger = GetLogger('parser')


def ParseSQL(fileName: str):
    statements = None
    with open(fileName, 'r') as f:
        raw = f.read()
        statements = sqlparse.split(raw)
    return statements
