#include "node.hpp"
#include "tree.hpp"
#include "random_forest.hpp"
#include "data_helpers.hpp"
#include "metrics.hpp"

#include <vector>
#include <tuple>

#include <iostream>
#include <iomanip>

using std::vector;
using std::tuple;
using std::get;
using std::cout;
using std::string;
using std::endl;

int main() {

    vector< vector<float> > data;
    vector<int> labels;
    load_iris(data, labels);
   
    vector< vector<float> > X_train, X_test;
    vector<int> y_train, y_test;
    train_test_split(data, labels, X_train, X_test, y_train, y_test, 0.25, 0);

    int n_estimators = 10;
    int max_depth=4;
    int min_samples_leaf = 1; 
    float min_information_gain = 0.0001f; 
    int max_features = 2;
    int bootstrap_sample_size = 30; 
    string criterion = "gini";
    int num_classes = 3;


    RandomForestClassifier rf = RandomForestClassifier(n_estimators, max_depth, min_samples_leaf, min_information_gain, max_features, bootstrap_sample_size, criterion, num_classes);
    
    rf.train(X_train, y_train);
    vector<int> y_pred = rf.predict(X_test);

    float acc = accuracy(y_pred, y_test);
    cout << "Accuracy on Iris Test Set: " << acc << std::endl;

    return 0;
}