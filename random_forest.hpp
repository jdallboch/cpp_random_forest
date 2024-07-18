#include "tree.hpp"

#include <vector>
#include <tuple>
#include <string>

class RandomForestClassifier {
public:
    void train(std::vector< std::vector<float> > const& data, std::vector<int> const& labels);
    std::vector< std::vector<float> > predict_proba(std::vector< std::vector<float> > const& X);
    std::vector<int> predict(std::vector< std::vector<float> > const& X);

    RandomForestClassifier(int n_estimators=10, int max_depth=4, int min_samples_leaf=1, float min_information_gain=0.0001f, int max_features=1, int bootstrap_sample_size=10, std::string criterion="info", int num_classes=2);

    void printRF();

    int getMaxDepth() const { return max_depth; }
    int getMinSamplesLeaf() const { return min_samples_leaf; }
    float getMinInformationGain() const { return min_information_gain; }
    int getMaxFeatures() const { return max_features; }

private:
    int n_estimators;
    int max_depth;
    int min_samples_leaf;
    float min_information_gain;
    int max_features;
    int bootstrap_sample_size;
    std::string criterion;
    int num_classes;
    std::vector<DecisionTreeClassifier> forest;

    std::tuple<std::vector< std::vector< std::vector<float> > >, std::vector<std::vector<int> > > bootstrap(std::vector< std::vector<float> > const& X, std::vector<int> const& y);

};