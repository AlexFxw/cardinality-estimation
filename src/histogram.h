/*
 * @Author: Hsuan-Wei Fan
 * @Date: 2020-06-02 18:07:58
 * @LastEditors: Hsuan-Wei Fan
 * @LastEditTime: 2020-06-02 21:23:11
 * @Description: 
 */

#ifndef CARDINALITY_ESTIMATION_HISTOGRAM_H
#define CARDINALITY_ESTIMATION_HISTOGRAM_H

#include "utils.h"
#include <fstream>
#include <map>

namespace CardEst
{
  struct Histogram
  {
    Histogram() = default;
    void AddData(Value value);
    uint32_t data[HISTO_SIZE];
  };

  class HistogramManager
  {
  public:
    HistogramManager() = default;

  private:
    std::string chartName;
    std::map<Key, std::unique_ptr<Histogram>> histograms;
  };

} // namespace CardEst
#endif // CARDINALITY_ESTIMATION_HISTOGRAM_H