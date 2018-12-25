/*
lnet logistic regression
*/
#pragma once // guard header

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

#include <Eigen/Dense>

using namespace Eigen;
using std::vector;
using std::sort;
using std::cout;

// Notes: We purposefully use doubles instead of integers everywhere to avoid any conversion.

struct LogitFitType {
  double intercept;
  VectorXd B;
};

struct LogitCVType {
  VectorXd risks;
  vector<double> lambdas;
};

double sigmoid(const double x) {
  return pow(1 + exp(-1 * x), -1);
}

VectorXd predict_class(const MatrixXd& X, const double intercept, const VectorXd& B) {
  const int n = X.rows();
  VectorXd pred = VectorXd::Zero(n);

  for (int i = 0; i < n; i++) {
     double p = sigmoid(intercept + X.row(i) * B);
    if (p > 1 - p) {
      pred(i) = 1.0;
    } else {
      pred(i) = 0.0;
    }
  }
  return pred;
}

VectorXd predict_prob(const MatrixXd& X, const double intercept, const VectorXd& B) {
  const int n = X.rows();
  VectorXd pred_prob = VectorXd::Zero(n);
  for (int i = 0; i < n; i++) {
    pred_prob(i) = sigmoid(intercept + X.row(i) * B);
  }
  return pred_prob;
}

// We purposefully use doubles
double accuracy(const VectorXd& v, const VectorXd& w) {
  if (v.rows() != w.rows()) {
    throw std::invalid_argument("Vectors must be the same length.");
  }
  const int n = v.rows();
  double m = 0;
  for (int i = 0; i < n; i++) {
    m += (v(i) == w(i));
  }
  return 1.0/((double)n) * m;
}

LogitFitType fit_logistic_proximal_gradient_coordinate_descent(const VectorXd& B_0, const MatrixXd& X, const VectorXd& y, 
                              const Matrix<double, 6, 1>& alpha, const double lambda, const double step_size,
                              const int max_iter, const double tolerance, const int random_seed) {
  LogitFitType fit; // return value

  const int n = X.rows();
  const int p = X.cols();

  // Create return values
  VectorXd B = B_0;
  double intercept = 0;
  for (int j = 0; j < max_iter; j++) {
    // Keep track of the current values
    const VectorXd B_old = B;
    const double intercept_old = intercept;

    const double h_j = step_size; // step size

    // Update intercept without penalization
    VectorXd Isigmoid_intercept_XB = VectorXd::Zero(n);
    for (int l = 0; l < n; l++) {
      Isigmoid_intercept_XB(l) = sigmoid(intercept + X.row(l) * B);
    }
    intercept = intercept + h_j * VectorXd::Ones(n).transpose() * (y - Isigmoid_intercept_XB);

    // Update the rest with penalization
    for (int i = 0; i < p; i++) {
      //
      // derivative of loss + differentiable penalizations
      //
      VectorXd sigmoid_intercept_XB = VectorXd::Zero(n);
      for (int k = 0; k < n; k++) {
        sigmoid_intercept_XB(k) = sigmoid(intercept + X.row(k) * B);
      }
      const double DL_i = X.col(i).transpose() * (y - sigmoid_intercept_XB);

      // Proximal Mapping: Soft Thresholding
      const double v_i = B(i) + h_j * DL_i; // gradient step
      if (v_i < -h_j * alpha(0) * lambda) {
        B(i) = v_i + h_j * alpha(0) * lambda;
      } else if (v_i >= -h_j * alpha(0) * lambda && v_i <= h_j * alpha(0) * lambda) {
        B(i) = 0;
      } else if (v_i > h_j * alpha(0) * lambda) {
        B(i) = v_i - h_j * alpha(0) * lambda;
      }
    }

    // Stopping criterion
    if ( pow(intercept - intercept_old, 2) + (B - B_old).squaredNorm() < tolerance ) {
      // Build return value
      fit.intercept = intercept;
      fit.B = B;
      return fit;
    }
  }

  std::cout << "Failed to converge! Tune the step size.\n";
  // Build return
  fit.intercept = intercept;
  fit.B = B;
  return fit;
}