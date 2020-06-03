/*
 * @Author: Hsuan-Wei Fan
 * @Date: 2020-06-02 15:46:52
 * @LastEditors: Hsuan-Wei Fan
 * @LastEditTime: 2020-06-02 22:03:44
 * @Description: main function
 */



#include <iostream>
#include <string>
#include "utils.h"
#include "table.h"

using namespace CardEst;

int ParseQuery(const std::string &fileName)
{
    std::ifstream file;
    ParseResult result;
    int parseRet = ParseSQLFile(fileName, result);
    if(parseRet) {
        return 1;
    }
    std::cout << "Successfully parsed: " << result.size() << " queries" << std::endl;
    return 0;
}

int main(int argc, char **argv) {
    // ParseQuery("../data/sample_input_homework/easy.sql");

    TableManager tableManager;
    tableManager.ParseCreateSql("../data/imdb/test.sql");
    return 0;
}