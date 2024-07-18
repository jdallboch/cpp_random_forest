#include "data_helpers.hpp"

#include <fstream>
#include <sstream>

#include <vector>
#include <tuple>
#include <map>
#include <random>

using std::vector;
using std::tuple;
using std::ifstream;
using std::string;
using std::getline;
using std::istringstream;
using std::map;


void load_iris(std::vector<std::vector<float> >& data, std::vector<int>& labels) {
    std::ifstream file("iris.data");
    if (!file.is_open()) {
        return;
    }

    std::string line;
    std::map<std::string, int> labelMap = {
        {"Iris-setosa", 0},
        {"Iris-versicolor", 1},
        {"Iris-virginica", 2}
    };

    for (int i = 0; i < 150; i++) {
        std::getline(file, line);
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;

        // Read the first four decimal values
        for (int i = 0; i < 4; ++i) {
            if (std::getline(ss, value, ',')) {
                row.push_back(std::stof(value));
            }
        }

        // Read the label
        if (std::getline(ss, value, ',')) {
            labels.push_back(labelMap[value]);
        }

        data.push_back(row);
    }

    file.close();
}

void train_test_split(vector<vector<float> > const& data, vector<int> const& labels, vector<vector<float> >& X_train, vector<vector<float> >& X_test, vector<int>& y_train, vector<int>& y_test, float test_split, unsigned int random_state) {

    int test_size = int(data.size()) * test_split;

    std::mt19937 gen(random_state);

    vector<int> indices;
    for (int i = 0; i < int(data.size()); i++) {
        indices.push_back(i);
    }

    std::shuffle(indices.begin(), indices.end(), gen);

    vector<int> test(indices.begin(), indices.begin() + test_size);
    vector<int> train(indices.begin() + test_size, indices.end());

    for (vector<int>::iterator idx = test.begin(); idx != test.end(); idx++) {
        X_test.push_back(data.at(*idx));
        y_test.push_back(labels.at(*idx));
    }

    for (vector<int>::iterator idx = train.begin(); idx != train.end(); idx++) {
        X_train.push_back(data.at(*idx));
        y_train.push_back(labels.at(*idx));
    }

}
