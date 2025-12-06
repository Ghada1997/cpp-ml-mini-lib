#pragma once

#include <string>
#include <vector>

namespace ml {

/**
 * Brief: Simple container for supervised or unsupervised datasets.
 *
 * X : features matrix (samples × features)
 * y : labels (only for supervised)
 */

struct Dataset {
    std::vector<std::vector<double>> X; 
    std::vector<int> y;                 
};

/**
 * Brief: A CSV loader for supervised and unsupervised datasets.
 */

class CSVLoader {
public:

    /**
     * Brief: Load a supervised CSV dataset (last column = label).
     */

    static Dataset load_supervised(const std::string& path,
                                   char delimiter = ',');

    /**
     * Brief: Load an unsupervised dataset where all columns are features.
     */
    static Dataset load_unsupervised(const std::string& path,
                                     char delimiter = ',');
};

} // namespace ml
