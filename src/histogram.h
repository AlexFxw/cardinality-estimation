#ifndef CARDINALITY_ESTIMATION_HISTOGRAM_H
#define CARDINALITY_ESTIMATION_HISTOGRAM_H

#include "utils.h"
#include <fstream>
#include <map>

namespace CardEst {

struct Histogram {
    Histogram() = default;
    void AddData(Value value);
    uint32_t data[HISTO_SIZE];
};

class HistogramManager {
public:
    HistogramManager() = default;
    int CalcHistogram(const std::string &chartName);

private:
    std::string dataDirectory;
    typedef std::map<Key, std::unique_ptr<Histogram>> ChartHistogram;
    std::map<std::string, std::unique_ptr<ChartHistogram>> histograms; // ChartName <-> Histogram
};

} // namespace CardEst
#endif // CARDINALITY_ESTIMATION_HISTOGRAM_H