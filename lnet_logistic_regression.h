/*
lnet logistic regression
*/
#pragma once // guard header

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
#include "lnet.h"

#include <iostream>

using namespace Eigen;
using std::vector;
using std::sort;
using std::cout;


FitType fit_logistic_proximal_gradient_cd(const VectorXd& B_0, const MatrixXd& X, const VectorXd& y, 
                              const Vector6d& alpha, const double lambda, const double step_size,
                              const int max_iter, const double tolerance, const int random_seed) {
  VectorXd B = B_0; // create return value
  const int n = X.rows();
  const int p = X.cols();

  // Create random permutation for coordinate descent using the Mersenne twister random number engine 64 bit
  vector<int> I(p);
  std::iota (std::begin(I), std::end(I), 0);
  std::mt19937_64 rng(random_seed);

  // Center: Convert X, y to mean 0
  MatrixXd cX = MatrixXd(n, p);
  for (int j = 0; j < X.cols(); j++) {
    cX.col(j) = X.col(j) - (X.col(j).mean() * VectorXd::Ones(n));
  }
  VectorXd cy = y - (y.mean() * VectorXd::Ones(n));

  for (int j = 0; j < max_iter; j++) {
    const VectorXd B_old = B; // Copy B for stopping criterion
    const double h_j = step_size; // step size

    std::shuffle(std::begin(I), std::end(I), rng); // permute
    for (int& i : I) {
      // derivative of loss + differentiable penalizations
      const double DL_i = -1 * cX.col(i).transpose() * (cy - (cX * B))
                + alpha(1) * lambda * B(i) + alpha(2) * lambda * pow(B(i), 3) 
                + alpha(3) * lambda * pow(B(i), 5) + alpha(4) * lambda * pow(B(i), 7) + alpha(5) * lambda * pow(B(i), 9);

      const double v_i = B(i) - h_j * DL_i; // gradient step

      // Proximal Mapping: Soft Thresholding
      if (v_i < -h_j * alpha(0) * lambda) {
        B(i) = v_i + h_j * alpha(0) * lambda;
      } else if (v_i >= -h_j * alpha(0) * lambda && v_i <= h_j * alpha(0) * lambda) {
        B(i) = 0;
      } else if (v_i > h_j * alpha(0) * lambda) {
        B(i) = v_i - h_j * alpha(0) * lambda;
      }
    }

    // Stop if the norm of the Moreau-Yoshida convolution gradient is less than tolerance
    if ( (1/h_j * (B_old - B)).squaredNorm() < tolerance ) {
      const double intercept = 1.0/((double)n) *  VectorXd::Ones(n).transpose() * (y.mean() * VectorXd::Ones(n) - (X * B));
      // Build return value
      FitType fit;
      fit.intercept = intercept;
      fit.B = B;
      return fit;
    }
  }

  std::cout << "Failed to converge! Tune the step size.\n";
  // Build return value
  const double intercept = 1/((double)n) *  VectorXd::Ones(n).transpose() * (y.mean() * VectorXd::Ones(n) - (X * B));
  FitType fit;
  fit.intercept = intercept;
  fit.B = B;
  return fit;
}