/*
 * @Author: Hsuan-Wei Fan
 * @Date: 2020-06-02 15:46:52
 * @LastEditors: Hsuan-Wei Fan
 * @LastEditTime: 2020-06-02 22:03:44
 * @Description: main function
 */



#include "SQLParser.h"
#include <fstream>
#include <iostream>
#include <string>
#include "utils.h"

using namespace CardEst;

int ParseQuery(const std::string &fileName)
{
    std::ifstream file;
    file.open(fileName);
    if(!file.is_open()) {
        std::cout << "Fail to open the query file: " << fileName << std::endl;
        return 1;
    }

    std::string rawFile((std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>());

    // std::cout << rawFile << std::endl;
    hsql::SQLParserResult result;
    hsql::SQLParser::parse(rawFile, &result);
    if(!result.isValid() || result.size() == 0) {
        std::cout << "Fail to parse query file: " << fileName << std::endl;
        return 1;
    }

    std::cout << "Successfully parsed: " << result.size() << " queries" << std::endl;

    return 0;
}

int main(int argc, char **argv) {
    ParseQuery("../data/sample_input_homework/easy.sql");
    return 0;
}