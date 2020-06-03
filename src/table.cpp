//
// Created by 范軒瑋 on 2020/6/3.
//

#include <fstream>
#include <iostream>
#include "table.h"

namespace CardEst {


Table Table::CreateTable(const hsql::CreateStatement *statement) {
    Table table;
    table.tableName = std::string(statement->tableName);
    std::cout << "Table: " << table.tableName;
    for (const hsql::ColumnDefinition *colDef: *statement->columns) {
        table.colAttribute.push_back(std::unique_ptr<ColDef>(new ColDef(*colDef)));
        printf("  col name: %s\n", table.colAttribute.back()->name);
    }
    return table;
}

int TableManager::ParseCreateSql(const std::string &fileName) {
    ParseResult result;
    int parseRes = ParseSQLFile(fileName, result);
    if (parseRes) {
        return 1;
    }
    for (const hsql::SQLStatement *statement: result.getStatements()) {
        if (!statement->isType(hsql::StatementType::kStmtCreate)) {
            std::cout << "Invalid statement type in creating query." << std::endl;
            return 1;
        }
        const auto *createStatement = (const hsql::CreateStatement *) statement;
        tables.push_back(Table::CreateTable(createStatement));
    }
    return 0;
}

}