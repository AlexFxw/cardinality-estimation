//
// Created by 范軒瑋 on 2020/6/3.
//

#ifndef GUESS_SQL_TABLE_H
#define GUESS_SQL_TABLE_H

#include <vector>
#include <map>
#include "sql/CreateStatement.h"
#include "utils.h"

namespace CardEst
{

class Table {
public:
    Table() = default;
    static Table CreateTable(const hsql::CreateStatement *statement);

private:
    typedef hsql::ColumnDefinition ColDef;
    uint16_t colNum;
    std::string tableName;
    std::vector<std::unique_ptr<ColDef>> colAttribute;
};

class TableManager {
public:
    int ParseCreateSql(const std::string &fileName);
private:
    std::vector<Table> tables;
};

}
#endif //GUESS_SQL_TABLE_H
