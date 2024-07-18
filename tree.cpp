#include "tree.hpp"
#include "node.hpp"

#include <vector>
#include <tuple>
#include <map>
#include <cmath>
#include <random>
#include <algorithm>
#include <queue>
#include <string>

#include <iostream>

using std::vector;
using std::tuple;
using std::map;
using std::string;
using std::get;

using std::cout;
using std::endl;

DecisionTreeClassifier::DecisionTreeClassifier(int max_depth, int min_samples_leaf, float min_information_gain, int max_features, string criterion, int num_classes) : max_depth(max_depth), min_samples_leaf(min_samples_leaf), min_information_gain(min_information_gain), max_features(max_features), criterion(criterion), num_classes(num_classes), head(nullptr) {}

DecisionTreeClassifier::~DecisionTreeClassifier() {
    destroy_tree(head);
}

void DecisionTreeClassifier::destroy_tree(Node* node) {
    if (node != nullptr) {
        destroy_tree(node->getLeft());
        node->setLeft(nullptr);
        destroy_tree(node->getRight());
        node->setRight(nullptr);
        delete node;
        node = nullptr;
    }
}

float DecisionTreeClassifier::gini(const vector<float>& class_probs) {
    float sum_squares = 0.0f;
    for (vector<float>::const_iterator cit = class_probs.cbegin(); cit != class_probs.cend(); cit++) {
        sum_squares += *cit * *cit;
    }
    return 1.0f - sum_squares;
}

float DecisionTreeClassifier::entropy(const vector<float>& class_probs) {
    float entropy = 0.0f;
    for (vector<float>::const_iterator it = class_probs.cbegin(); it != class_probs.cend(); it++) {
        if (*it > 0.0f) {
            entropy -= *it * log2(*it);
        }
    }
    return entropy;
}

vector<float> DecisionTreeClassifier::class_probs(vector<int> const& labels) {
    vector<int> counts(num_classes, 0);
    int len = labels.size();

    for (vector<int>::const_iterator it = labels.cbegin(); it != labels.cend(); it++) {
        if (*it < num_classes && *it >= 0) {
            counts.at(*it)++;
        }
    }

    vector<float> probs(num_classes, 0.0f);

    for (int i = 0; i < num_classes; i++) {
        float prob = static_cast<float>(counts.at(i)) / len;
        probs.at(i) = prob;
    }

    return probs;
}

float DecisionTreeClassifier::partition_impurity(vector< vector<int> > const& subsets_labels, float (DecisionTreeClassifier::*imp_measure)(const vector<float>&)) {
    int total(0);
    for (int i = 0; i < int(subsets_labels.size()); i++) {
        total += subsets_labels.at(i).size();
    }

    float impurity_total(0.0f);

    for (int i = 0; i < int(subsets_labels.size()); i++) {
        vector<int> subset = subsets_labels.at(i);
        if (!subset.empty()) {
            impurity_total += (this->*imp_measure)(class_probs(subset)) * int(subset.size()) / total;
        }
    }

    return impurity_total;
}

tuple< vector< vector<float> >, vector<int>, vector< vector<float> >, vector<int> > DecisionTreeClassifier::split(vector< vector<float> > const& data, vector<int> const& labels, int feature_index, float feature_val) {
    vector< vector< vector<float> > > split_groups;
    vector< vector<float> > g1, g2;
    vector<int> g1_labels, g2_labels;

    for (size_t i = 0; i < data.size(); i++) {
        vector<float> copy = data.at(i);
        if (copy.at(feature_index) < feature_val) {
            g1.push_back(copy);
            g1_labels.push_back(labels.at(i));
        } else {
            g2.push_back(copy);
            g2_labels.push_back(labels.at(i));
        }
    }
    return {g1, g1_labels, g2, g2_labels};
}

vector<int> DecisionTreeClassifier::select_features(vector< vector<float> > const& data) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);

    int len = data.at(0).size();
        
    vector<int> indices;
    for (int i = 0; i < len; i++) {
        indices.push_back(i);
    }

    std::shuffle(indices.begin(), indices.end(), gen);

    std::vector<int> samples(indices.begin(), indices.begin() + max_features);

    return samples;
}

tuple<vector< vector<float> >, vector<int>, vector< vector<float> >, vector<int>, int, float, float> DecisionTreeClassifier::optimal_split(std::vector< std::vector<float> > const& data, vector<int> const& labels, float (DecisionTreeClassifier::*imp_measure)(const vector<float>&)) {

    float min_impurity = 1000.0f;
    vector< vector<float> > g1, g2;
    vector< int> g1_labels, g2_labels;
    int feature_idx;
    float feature_val;
    
    vector<int> features = select_features(data);

    for (vector<int>::const_iterator ft_idx = features.cbegin(); ft_idx != features.cend(); ft_idx++) {
        vector<float> features;
        for (vector<vector<float> >::const_iterator cit = data.cbegin(); cit != data.cend(); cit++) {
            features.push_back(cit->at(*ft_idx));
        }

        std::sort(features.begin(), features.end());

        vector<int> percentiles;

        for (float i = 0.1; i < 1; i += 0.1) {
            percentiles.push_back(features.at(int(i * features.size())));
        }

        for (vector<int>::const_iterator cit = percentiles.cbegin(); cit != percentiles.cend(); cit++) {            

            tuple<vector< vector<float> >, vector<int>, vector< vector<float> >, vector<int> > ft_split = split(data, labels, *ft_idx, *cit);

            vector<int> temp_g1 = get<1>(ft_split);
            vector<int> temp_g2 = get<3>(ft_split);

            float split_impurity = partition_impurity({temp_g1, temp_g2}, imp_measure);

            if (split_impurity < min_impurity) {
                min_impurity = split_impurity;
                g1 = get<0>(ft_split);
                g1_labels = temp_g1;
                g2 = get<2>(ft_split);
                g2_labels = temp_g2;
                feature_idx = *ft_idx;
                feature_val = *cit;
            }
        }
    }

    return {g1, g1_labels, g2, g2_labels, feature_idx, feature_val, min_impurity};
}

Node* DecisionTreeClassifier::create_tree(vector< vector<float> > const& data, vector<int> const& labels, int curr_depth, float (DecisionTreeClassifier::*imp_measure)(const vector<float>&)) {
    if (curr_depth >= max_depth) {return nullptr;}

    auto split_results = optimal_split(data, labels, imp_measure);
    
    vector<int> temp_g1 = get<1>(split_results);
    vector<int> temp_g2 = get<3>(split_results);

    float node_entropy = (this->*imp_measure)(class_probs(labels));

    float split_entropy = partition_impurity({get<1>(split_results), get<3>(split_results)}, imp_measure);

    float info_gain = node_entropy - split_entropy;

    Node* node = new Node(data, labels, get<4>(split_results), get<5>(split_results), class_probs(labels), info_gain);

    if (info_gain < min_information_gain || int(get<1>(split_results).size()) < min_samples_leaf || int(get<3>(split_results).size()) < min_samples_leaf) {
        return node;
    } else {
        node->setLeft(create_tree(get<0>(split_results), get<1>(split_results), curr_depth + 1, imp_measure));
        node->setRight(create_tree(get<2>(split_results), get<3>(split_results), curr_depth + 1, imp_measure));
    }

    return node;
}

void DecisionTreeClassifier::train(vector< vector< float> > const& data, vector<int> const& labels) {
    float (DecisionTreeClassifier::*imp_measure)(const vector<float>&);
    if (criterion == "gini") {
        imp_measure = &DecisionTreeClassifier::gini;
    } else if (criterion == "info") {
        imp_measure = &DecisionTreeClassifier::entropy;
    } else {
        imp_measure = &DecisionTreeClassifier::entropy;
    }
    head = create_tree(data, labels, 0, imp_measure);
}

vector<float> DecisionTreeClassifier::predict_single_sample(vector<float> const& sample) {
    Node* node = head;
    vector<float> pred_probs;

    while (node != nullptr) {
        pred_probs = node->getPredProbs();
        if (sample.at(node->getFtIdx()) < node->getFtVal()) {
            node = node->getLeft();
        } else {
            node = node->getRight();
        }
    }
    return pred_probs;
}

vector< vector<float> > DecisionTreeClassifier::predict_proba(vector< vector<float> > const& X) {
    vector<vector<float> > probs;
    for (vector< vector<float> >::const_iterator sample = X.cbegin(); sample != X.cend(); sample++) {
        probs.push_back(predict_single_sample(*sample));
    }
    return probs;
}

int DecisionTreeClassifier::classify_single_sample(vector<float> const& probs) {
    float max = -1;
    int max_idx = -1;
    for (int i = 0; i < int(probs.size()); i++) {
        float temp = probs.at(i);
        if (temp > max) {
            max = temp;
            max_idx = i;
        }
    }
    return max_idx;
}

vector<int> DecisionTreeClassifier::predict(vector< vector<float> > const& X) {
    vector<int> results;
    vector<vector<float> > probs = predict_proba(X);
    for (vector<vector<float> >::const_iterator cit = probs.cbegin(); cit != probs.cend(); cit++) {
        results.push_back(classify_single_sample(*cit));
    }
    return results;
}


//CHATGPT DEBUGGING FUNCTION
void DecisionTreeClassifier::printTree() {
    if (!head) return;
    
    std::queue<Node*> q;
    q.push(head);
    
    while (!q.empty()) {
        int size = q.size();
        while (size > 0) {
            Node* current = q.front();
            q.pop();
            std::cout << "(" << current->getData().size() << ", " << current->getFtIdx() << ", " << current->getFtVal() << ") ";
            if (current->getLeft()) q.push(current->getLeft());
            if (current->getRight()) q.push(current->getRight());
            size--;
        }
        std::cout << std::endl; // Move to the next level
    }
}