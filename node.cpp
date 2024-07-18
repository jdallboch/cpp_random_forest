#include "node.hpp"
#include <vector>


Node::Node(std::vector< std::vector<float> > data, std::vector< int > labels, int feature_index,
    float feature_val, std::vector< float > prediction_probs, float it_shift) : data(data), labels(labels), feature_index(feature_index), feature_val(feature_val), prediction_probs(prediction_probs), it_shift(it_shift), left(nullptr), right(nullptr) {}
    
Node::~Node() {
    delete left;
    delete right;
}