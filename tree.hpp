//tree.h
#ifndef TREE_HPP
#define TREE_HPP

#include <vector>
#include <tuple>
#include <random>
#include <string>
#include "node.hpp"

class DecisionTreeClassifier {

public:
    DecisionTreeClassifier(int max_depth=4, int min_samples_leaf=1, float min_information_gain=0.0001f, int _max_features=1, std::string criterion="info", int num_classes=2);
    ~DecisionTreeClassifier();

    void train(std::vector< std::vector<float> > const& data, std::vector<int> const& labels);
    std::vector< std::vector<float> > predict_proba(std::vector< std::vector<float> > const& X);
    std::vector<int> predict(std::vector< std::vector<float> > const& X);
    std::vector<float> predict_single_sample(std::vector<float> const& sample);

    void printTree();

    float gini(std::vector<float> const& class_probs);

    float entropy(std::vector<float> const& class_probs);

private:
    int max_depth;
    int min_samples_leaf;
    float min_information_gain;
    int max_features;
    std::string criterion;
    int num_classes;
    Node* head;

    void destroy_tree(Node* node);

    
    std::vector<float> class_probs(std::vector<int> const& labels);
    float partition_impurity(std::vector< std::vector<int> > const& subsets_labels, float (DecisionTreeClassifier::*imp_measure)(const std::vector<float>&));
    std::tuple< std::vector< std::vector< float> >, std::vector<int>, std::vector< std::vector< float> >, std::vector<int> > split(std::vector< std::vector<float> > const& data, std::vector<int> const& labels, int feature_index, float feature_val);
    std::vector<int> select_features(std::vector< std::vector<float> > const& data);
    std::tuple<std::vector< std::vector<float> >, std::vector<int>, std::vector< std::vector<float> >, std::vector<int>, int, float, float> optimal_split(std::vector< std::vector<float> > const& data, std::vector<int> const& labels, float (DecisionTreeClassifier::*imp_measure)(const std::vector<float>&));
    Node* create_tree(std::vector< std::vector<float> > const& data, std::vector<int> const& labels, int curr_depth, float (DecisionTreeClassifier::*imp_measure)(const std::vector<float>&));
    int classify_single_sample(std::vector<float> const& probs);

};

#endif