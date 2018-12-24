/*
Testing for logistic regression
*/

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
#include "lnet.h"

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
  VectorXd z_train = load_csv<MatrixXd>("data/z_train.csv");
  //VectorXi z_train = z_train_.cast<int>();

  MatrixXd X_test = load_csv<MatrixXd>("data/X_test.csv");
  VectorXd z_test = load_csv<MatrixXd>("data/z_test.csv");
  //VectorXi z_test = z_test_.cast<int>();

  // create alpha
  double alpha_data[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  Map<Vector6d> alpha(alpha_data);

  double step_size = 1/((double) 10000);
  int max_iter = 10000;
  double tolerance = pow(10, -6);
  int random_seed = 0;

  cout << R"(
  Test fit_logistic_proximal_gradient_cd
  -------
  )";
  
  double lambda = .1;
  VectorXd B_0 = VectorXd::Zero(X_train.cols());
  //FitType fit = fit_logistic_proximal_gradient_cd(B_0, X_train, y_train, alpha, lambda, step_size, max_iter, tolerance, random_seed);



}

} // end namespace