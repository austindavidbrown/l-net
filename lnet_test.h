/*
Testing for lnet
*/
#pragma once // guard header

#include <Eigen/Dense> 
#include "lnet.h"

#include <time.h>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace Eigen;
using std::cout;

namespace lnet_test {

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
Tests for proximal gradient descent with line search
*/
void test_fit_proximal_gradient(MatrixXd& X_train, VectorXd& y_train, MatrixXd& X_test, VectorXd& y_test, Vector6d alpha, double lambda) {
  cout << R"(
  Test fit_proximal_gradient
  -------
  )";

  int max_iter = 10000;
  double tolerance = pow(10, -6);

  VectorXd B_0 = VectorXd::Zero(X_train.cols());
  FitType fit = fit_proximal_gradient(B_0, X_train, y_train, alpha, lambda, max_iter, tolerance);

  cout << "\nintercept:\n" << fit.intercept << "\n";
  cout << "\nB:\n" << fit.B << "\n";

  cout << "\nMSE: " << mean_squared_error(y_train, predict(X_train, fit.intercept, fit.B)) << "\n";
  cout << "\nTest MSE: " << mean_squared_error(y_test, predict(X_test, fit.intercept, fit.B)) << "\n";
}

void test_fit_warm_start_proximal_gradient(MatrixXd& X, VectorXd& y, Vector6d alpha, vector<double> lambdas) {
  cout << R"(
  Test fit_warm_start_proximal_gradient
  -------
  )";

  int max_iter = 10000;
  double tolerance = pow(10, -6);

  vector<FitType> fits = fit_warm_start_proximal_gradient(X, y, alpha, lambdas, max_iter, tolerance);
  FitType last_fit = fits.at(fits.size() - 1);
  cout << "\nlast fit intercept:\n" << last_fit.intercept << "\n";
  cout << "\nlast fit B:\n" << last_fit.B << "\n";
}

void test_cross_validation_proximal_gradient(MatrixXd& X_train, VectorXd& y_train, MatrixXd& X_test, VectorXd& y_test, Vector6d alpha, vector<double> lambdas, int K_fold) {
  cout << R"(
  Test cross_validation_proximal_gradient
  -------
  )";

  int max_iter = 10000;
  double tolerance = pow(10, -6);
  int random_seed = 0;

  CVType cv = cross_validation_proximal_gradient(X_train, y_train, K_fold, alpha, lambdas, max_iter, tolerance, random_seed);
  cout << "\nCV Risks:\n" << cv.risks << "\n";

  cout << "\nOrdered Lambdas\n";
  for (auto& lambda : cv.lambdas) {
    cout << lambda << " ";
  }
  cout << "\n";

  // get the best lambda
  MatrixXf::Index min_row;
  cv.risks.minCoeff(&min_row);
  double best_lambda = cv.lambdas[min_row];
  cout << "\nBest Lambda:\n" << best_lambda << "\n";

  VectorXd B_0 = VectorXd::Zero(X_train.cols());
  FitType best_fit = fit_proximal_gradient(B_0, X_train, y_train, alpha, best_lambda, max_iter, tolerance);
  cout << "\nTest MSE: " << mean_squared_error(y_test, predict(X_test, best_fit.intercept, best_fit.B)) << "\n";
}


/*
Tests for proximal gradient coordinate descent
*/
void test_fit_proximal_gradient_cd(MatrixXd& X_train, VectorXd& y_train, MatrixXd& X_test, VectorXd& y_test, Vector6d alpha, double lambda, double step_size) {
  cout << R"(
  Test fit_proximal_gradient_cd
  -------
  )";

  int max_iter = 10000;
  double tolerance = pow(10, -6);
  int random_seed = 0;

  VectorXd B_0 = VectorXd::Zero(X_train.cols());
  FitType fit = fit_proximal_gradient_cd(B_0, X_train, y_train, alpha, lambda, step_size, max_iter, tolerance, random_seed);

  cout << "\nintercept:\n" << fit.intercept << "\n";
  cout << "\nB:\n" << fit.B << "\n";

  cout << "\nMSE: " << mean_squared_error(y_train, predict(X_train, fit.intercept, fit.B)) << "\n";
  cout << "\nTest MSE: " << mean_squared_error(y_test, predict(X_test, fit.intercept, fit.B)) << "\n";
}

void test_fit_warm_start_proximal_gradient_cd(MatrixXd& X, VectorXd& y, Vector6d alpha, vector<double> lambdas, double step_size) {
  cout << R"(
  Test fit_warm_start_proximal_gradient_cd
  -------
  )";

  int max_iter = 10000;
  double tolerance = pow(10, -6);
  int random_seed = 0;

  vector<FitType> fits = fit_warm_start_proximal_gradient_cd(X, y, alpha, lambdas, step_size, max_iter, tolerance, random_seed);
  FitType last_fit = fits.at(fits.size() - 1);
  cout << "\nlast fit intercept:\n" << last_fit.intercept << "\n";
  cout << "\nlast fit B:\n" << last_fit.B << "\n";
}

void test_cross_validation_proximal_gradient_cd(MatrixXd& X_train, VectorXd& y_train, MatrixXd& X_test, VectorXd& y_test, Vector6d alpha, vector<double> lambdas, double step_size, int K_fold) {
  cout << R"(
  Test cross_validation_proximal_gradient_cd
  -------
  )";

  int max_iter = 10000;
  double tolerance = pow(10, -6);
  int random_seed = 0;

  CVType cv = cross_validation_proximal_gradient_cd(X_train, y_train, K_fold, alpha, lambdas, step_size, max_iter, tolerance, random_seed);
  cout << "\nCV Risks:\n" << cv.risks << "\n";

  cout << "\nOrdered Lambdas\n";
  for (auto& lambda : cv.lambdas) {
    cout << lambda << " ";
  }
  cout << "\n";

  // get the best lambda
  MatrixXf::Index min_row;
  cv.risks.minCoeff(&min_row);
  double best_lambda = cv.lambdas[min_row];
  cout << "\nBest Lambda:\n" << best_lambda << "\n";

  VectorXd B_0 = VectorXd::Zero(X_train.cols());
  FitType best_fit = fit_proximal_gradient_cd(B_0, X_train, y_train, alpha, best_lambda, step_size, max_iter, tolerance, random_seed);
  cout << "\nTest MSE: " << mean_squared_error(y_test, predict(X_test, best_fit.intercept, best_fit.B)) << "\n";
}

void test_regression() {
  //std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1); // set precision

  MatrixXd X_train = load_csv<MatrixXd>("data/X_train.csv");
  VectorXd y_train = load_csv<MatrixXd>("data/y_train.csv");
  MatrixXd X_test = load_csv<MatrixXd>("data/X_test.csv");
  VectorXd y_test = load_csv<MatrixXd>("data/y_test.csv");

  // create alpha
  double alpha_data[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  Map<Vector6d> alpha(alpha_data);

  double step_size = 1/((double) 10000);

  //
  // fit test
  //
  double lambda = .1;
  test_fit_proximal_gradient(X_train, y_train, X_test, y_test, alpha, lambda);
  test_fit_proximal_gradient_cd(X_train, y_train, X_test, y_test, alpha, lambda, step_size);

  //
  // warm start test
  //
  // create lambdas
  vector<double> lambdas;
  lambdas.push_back(pow(10, -3));
  for (int i = 1; i < 10; i++) {
    lambdas.push_back(lambdas[i - 1] + .1);
  }
  test_fit_warm_start_proximal_gradient(X_train, y_train, alpha, lambdas);
  test_fit_warm_start_proximal_gradient_cd(X_train, y_train, alpha, lambdas, step_size);

  //
  // cv test
  //
  int K_fold = 10;
  test_cross_validation_proximal_gradient(X_train, y_train, X_test, y_test, alpha, lambdas, K_fold);
  test_cross_validation_proximal_gradient_cd(X_train, y_train, X_test, y_test, alpha, lambdas, step_size, K_fold);
}

void test_regression_prostate() {
  MatrixXd X_train = load_csv<MatrixXd>("data/prostate_X_train.csv");
  VectorXd y_train = load_csv<MatrixXd>("data/prostate_y_train.csv");

  MatrixXd X_test = load_csv<MatrixXd>("data/prostate_X_test.csv");
  VectorXd y_test = load_csv<MatrixXd>("data/prostate_y_test.csv");

  // create alpha
  double alpha_data[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  Map<Vector6d> alpha(alpha_data);

  double step_size = 1.0f/((double) 80);

  //
  // fit test
  //
  double lambda = 11;
  test_fit_proximal_gradient(X_train, y_train, X_test, y_test, alpha, lambda);
  test_fit_proximal_gradient_cd(X_train, y_train, X_test, y_test, alpha, lambda, step_size);

  //
  // warm start test
  //
  // Create lambdas
  vector<double> lambdas;
  lambdas.push_back(X_train.cols());
  for (int i = 1; i < 100; i++) {
    lambdas.push_back(lambdas[i - 1] + .1);
  }

  test_fit_proximal_gradient(X_train, y_train, X_test, y_test, alpha, lambda);
  test_fit_proximal_gradient_cd(X_train, y_train, X_test, y_test, alpha, lambda, step_size);

  //
  // cv test
  //
  int K_fold = 10;
  test_cross_validation_proximal_gradient(X_train, y_train, X_test, y_test, alpha, lambdas, K_fold);
  test_cross_validation_proximal_gradient_cd(X_train, y_train, X_test, y_test, alpha, lambdas, step_size, K_fold);
}

} // end namespace
