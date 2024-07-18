//node.hpp
#ifndef NODE_HPP
#define NODE_HPP

#include <vector>

class Node {

public:
    
    Node(std::vector< std::vector<float> > data, std::vector< int > labels, int feature_index, float feature_val, std::vector< float > prediction_probs, float it_shift);
    ~Node();

    Node* getLeft() const {return left;}
    Node* getRight() const {return right;}
    void setLeft(Node* node) {left = node;}
    void setRight(Node* node) {right = node;}
    std::vector<float> getPredProbs() {return prediction_probs;}
    int getFtIdx() {return feature_index;}
    float getFtVal() {return feature_val;}
    std::vector< std::vector<float> > getData() {return data;}
    float getInfoTheoryChg() {return it_shift;}


private:
    std::vector< std::vector<float> > data;
    std::vector< int > labels;
    int feature_index;
    float feature_val;
    std::vector< float > prediction_probs;
    float it_shift;
    Node * left;
    Node * right;

};

#endif 