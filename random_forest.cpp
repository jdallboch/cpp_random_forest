#include "random_forest.hpp"
#include "tree.hpp"

#include <vector>
#include <random>
#include <tuple>
#include <numeric>
#include <chrono>
#include <string>

#include <iostream>

using std::tuple;
using std::vector;
using std::get;
using std::string;

RandomForestClassifier::RandomForestClassifier(int n_estimators, int max_depth, int min_samples_leaf, float min_information_gain, int max_features, int bootstrap_sample_size, string criterion, int num_classes) : n_estimators(n_estimators), max_depth(max_depth), min_samples_leaf(min_samples_leaf), min_information_gain(min_information_gain), max_features(max_features), bootstrap_sample_size(bootstrap_sample_size), criterion(criterion), num_classes(num_classes) {
    for (int i = 0; i < n_estimators; i++) {
        forest.push_back(DecisionTreeClassifier(max_depth, min_samples_leaf, min_information_gain, max_features, criterion, num_classes));
    }
}

tuple<vector< vector< vector<float> > >, vector< vector<int> > > RandomForestClassifier::bootstrap(vector< vector<float> > const& X, vector<int> const& y) {
    vector< vector< vector<float> > > bootstrap_samples;
    vector< vector<int> > bootstrap_labels;
    
    for (int i = 0; i < n_estimators; i++) {
        unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count() + i;
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dis(0, X.size() - 1);
        
        vector< vector<float> > dataset;
        vector<int> labels;
        for (int j = 0; j < bootstrap_sample_size; j++) {
            int idx = dis(gen);
            dataset.push_back(X.at(idx));
            labels.push_back(y.at(idx));
        }
        bootstrap_samples.push_back(dataset);
        bootstrap_labels.push_back(labels);
    }
    return {bootstrap_samples, bootstrap_labels};
}


void RandomForestClassifier::train(vector< vector<float> > const& data, vector<int> const& labels) {
    
    tuple<vector< vector< vector<float> > >, vector< vector<int> > > bootstrapped = bootstrap(data, labels);
    
    vector< vector< vector<float> > > bootstrapped_data = get<0>(bootstrapped);
    vector< vector<int> > bootstrapped_labels = get<1>(bootstrapped);

    int i = 0;
    for (vector<DecisionTreeClassifier>::iterator tree = forest.begin(); tree != forest.end(); tree++) {
        tree->train(bootstrapped_data.at(i), bootstrapped_labels.at(i));
        i++;
    }

}

vector< vector<float> > RandomForestClassifier::predict_proba(vector< vector<float> > const& X) {
    vector< vector< float> > out;
    
    for (vector< vector<float> >::const_iterator sample = X.cbegin(); sample != X.cend(); sample++) {
        vector< vector<float> > tree_results;
        for (vector<DecisionTreeClassifier>::iterator tree = forest.begin(); tree != forest.end(); tree++) {
            tree_results.push_back(tree->predict_single_sample(*sample));
        }
        vector<float> composite_prob;
        for (int i = 0; i < num_classes - 1; i++) {
            float total(0);
            for (vector< vector<float> >::iterator sample = tree_results.begin(); sample != tree_results.end(); sample++) {
                total += sample->at(i);
            }
            total /= int(tree_results.size());
            composite_prob.push_back(total);
        }
        float sum = std::accumulate(composite_prob.begin(), composite_prob.end(), 0.0f);
        composite_prob.push_back(1.0f - sum);
        out.push_back(composite_prob);  
    }
    return out;
}

vector<int> RandomForestClassifier::predict(vector< vector<float> > const& X) {
    vector< vector<float> > probs = predict_proba(X);
    vector<int> classifications;
    for (vector<vector<float> >::iterator dist = probs.begin(); dist != probs.end(); dist++) {
        float max = -1;
        int max_idx = -1;
        for (int i = 0; i < num_classes; i++) {
            if (dist->at(i) > max) {
                max = dist->at(i);
                max_idx = i;
            }
        }
        classifications.push_back(max_idx);
    }
    return classifications;
}

void RandomForestClassifier::printRF() {
    for (vector<DecisionTreeClassifier>::iterator tree = forest.begin(); tree != forest.end(); tree++) {
        tree->printTree();
        std::cout << "-------------------" << std::endl;
    }
}


