#include "metrics.hpp"

#include <vector>
#include <iostream>

float accuracy(std::vector<int> const& y_pred, std::vector<int> const& y_true) {
    int correct(0);
    int total(0);

    for (size_t i = 0; i < y_pred.size(); i++) {
        if (y_pred.at(i) == y_true.at(i)) {
            correct++;
        }
        total++;
    }
    return static_cast<float>(correct) / total;
}