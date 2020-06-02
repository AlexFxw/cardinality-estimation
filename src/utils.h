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

namespace CardEst
{
  typedef std::string Key; 
  typedef int Value;

  constexpr static uint32_t HISTO_SIZE = 256;

} // namespace CardEst

#endif // CARDINALITY_ESTIMATION_UTILS_H
