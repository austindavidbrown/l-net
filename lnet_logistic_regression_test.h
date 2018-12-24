/*
Testing for logistic regression
*/

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Dense>

#include <time.h>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "lnet_logistic_regression.h"

using namespace Eigen;
using std::cout;

namespace lnet_logistic_regression_test {

// CV parser
template<typename M>
M load_csv(const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

void test_logistic_regression() {
  MatrixXd X_train = load_csv<MatrixXd>("data/X_train.csv");
  VectorXd z_train_ = load_csv<MatrixXd>("data/z_train.csv");
  VectorXi z_train = z_train_.cast<int>();

  MatrixXd X_test = load_csv<MatrixXd>("data/X_test.csv");
  VectorXd z_test_ = load_csv<MatrixXd>("data/z_test.csv");
  VectorXi z_test = z_test_.cast<int>();
}

} // end namespace