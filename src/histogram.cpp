#include <iostream>
#include <sstream>
#include "histogram.h"

namespace CardEst {

void Histogram::AddData(Value value) {
    // TODO: Add data to histogram
}

int HistogramManager::CalcHistogram(const std::string &chartName) {
    auto findIter = histograms.find(chartName);
    if(findIter != histograms.end()) {
        std::cout << chartName << "'s histogram is already calculated." << std::endl;
        return 1;
    }
    const std::string fileName = dataDirectory + "/" + chartName + ".csv";
    ChartHistogram chartHistogram;
    std::ifstream file;
    file.open(fileName);
    if(!file.is_open()) {
        std::cout << "Fail to open chart: " << chartName << std::endl;
        return 1;
    }
    // How to get the key?
    while(!file.eof()) {
        std::string line;
        getline(file, line);
        std::istringstream iss(line);
        std::string item;
    }

    histograms[chartName] = std::unique_ptr<ChartHistogram>(new ChartHistogram(std::move(chartHistogram)));
    file.close();
    return 0;
}

} // namespace CardEst