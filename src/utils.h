/*
 * @Author: Hsuan-Wei Fan
 * @Date: 2020-06-02 18:10:35
 * @LastEditors: Hsuan-Wei Fan
 * @LastEditTime: 2020-06-02 21:02:25
 * @Description: 
 */

#ifndef CARDINALITY_ESTIMATION_UTILS_H
#define CARDINALITY_ESTIMATION_UTILS_H

#include <cstdlib>
#include <cstdint>
#include <string>
#include <memory>
#include <fstream>
#include "SQLParserResult.h"
#include "SQLParser.h"

namespace CardEst {
typedef std::string Key;
typedef int Value;
typedef hsql::SQLParserResult ParseResult;

constexpr static uint32_t HISTO_SIZE = 256;

inline int ParseSQLFile(const std::string &fileName, ParseResult &result) {
    std::ifstream file;
    file.open(fileName);
    if (!file.is_open()) {
        std::cout << "Fail to open file to parse" << std::endl;
        return 1;
    }
    std::string rawFile((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    hsql::SQLParser::parse(rawFile, &result);
    if (!result.isValid() || result.size() == 0) {
        std::cout << "result is invalid" << std::endl;
        return 1;
    }
    file.close();
    return 0;
}

} // namespace CardEst

#endif // CARDINALITY_ESTIMATION_UTILS_H
