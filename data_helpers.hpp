#ifndef DATA_HELPERS_HPP
#define DATA_HELPERS_HPP

#include <vector>

void load_iris(std::vector<std::vector<float> >& data, std::vector<int>& labels);
void train_test_split(std::vector<std::vector<float> > const& data, std::vector<int> const& labels, std::vector<std::vector<float> >& X_train, std::vector<std::vector<float> >& X_test, std::vector<int>& y_train, std::vector<int>& y_test, float test_split, unsigned int random_state);

#endif