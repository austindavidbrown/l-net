/*
Testing for logistic regression
*/
#pragma once // guard header

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>

#include <time.h>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <Eigen/Dense>
#include "lnet.h"
#include "lnet_classification.h"

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

/*
Tests for proximal gradient coordinate descent
*/
void test_fit_logistic_proximal_gradient_coordinate_descent(MatrixXd& X_train, VectorXd& y_train, MatrixXd& X_test, VectorXd& y_test, Vector6d alpha, double lambda, double step_size) {
  cout << R"(
  Test fit_logistic_proximal_gradient_coordinate_descent
  -------
  )";

  VectorXd B_0 = VectorXd::Zero(X_train.cols());
  LogitFitType fit = fit_logistic_proximal_gradient_coordinate_descent(B_0, X_train, y_train, alpha, lambda, step_size, 
                                                                       10000, pow(10, -6), 0);

  cout << "\nintercept:\n" << fit.intercept << "\n";
  cout << "\nB:\n" << fit.B << "\n";

  cout << "\nTrain accuracy : " << accuracy(y_train, predict_class(X_train, fit.intercept, fit.B)) << "\n";
  cout << "\nTest accuracy: " << accuracy(y_test, predict_class(X_test, fit.intercept, fit.B)) << "\n";
}

void test_logistic_regression() {
  cout << R"(
  Test logistic_regression
  -------
  )";

  MatrixXd X_train = load_csv<MatrixXd>("data/X_train.csv");
  VectorXd y_train = load_csv<MatrixXd>("data/binary_y_train.csv");

  MatrixXd X_test = load_csv<MatrixXd>("data/X_test.csv");
  VectorXd y_test = load_csv<MatrixXd>("data/binary_y_test.csv");

  // create alpha
  double alpha_data[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  Map<Vector6d> alpha(alpha_data);

  double step_size = .01;
  double lambda = .5;
  test_fit_logistic_proximal_gradient_coordinate_descent(X_train, y_train, X_test, y_test, alpha, lambda, step_size);
  cout << INFINITY;
}

} // end namespace