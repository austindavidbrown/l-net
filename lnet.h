/*
lnet

The optimization functions fail silently and return the value if the max iterations is reached. 
This is to avoid complicated logic. 
It will be easily noticed on the test error.
*/
#pragma once // guard header

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Dense>

#include <iostream>

using namespace Eigen;
using std::vector;
using std::sort;
using std::cout;

struct FitType {
  double intercept;
  VectorXd B;
};

struct CVType {
  VectorXd risks;
  vector<double> lambdas;
};

//
// Utils
//
MatrixXd standardize(MatrixXd M) {
  const int n = M.rows();
  const int m = M.cols();
  for (int j = 0; j < m; j++) {
    M.col(j) = (M.col(j) - M.col(j).mean() * VectorXd::Ones(n)).normalized();
  }
  return M;
}

template<typename T>
vector<vector<T>> partition(const vector<T>& S, const size_t n) {
  vector<vector<T>> partitions;

  size_t length = S.size() / n;
  size_t remainder = S.size() % n;

  size_t begin = 0;
  size_t end = 0;
  for (size_t i = 0; i < n; ++i) {
    if (remainder > 0) {
      end += length + !!(remainder--);
    } else {
      end += length;
    }
    partitions.push_back(vector<T>(S.begin() + begin, S.begin() + end));
    begin = end;
  }

  return partitions;
}

/*
================================

Lnet Proximal Gradient Regression

================================
*/

double mean_squared_error(const VectorXd& v, const VectorXd& w) {
  return 1/((double)v.rows()) * (v - w).squaredNorm(); 
}

VectorXd predict_regression(const MatrixXd& X, const double intercept, const VectorXd& B) {
  const int n = X.rows();
  return intercept * VectorXd::Ones(n) + (X * B);
}

FitType fit_regression_proximal_gradient(const VectorXd& B_0, const MatrixXd& X, const VectorXd& y, 
                              const Matrix<double, 6, 1>& alpha, const double lambda,
                              const int max_iter, const double tolerance) {
  FitType fit;

  const int n = X.rows();
  const int p = X.cols();
  const VectorXd One_n = VectorXd::Ones(n);

  // Create the values to optimize
  double intercept = 0;
  VectorXd B = B_0;
  for (int j = 0; j < max_iter; j++) {
    // Keep track of current B and intercept
    const double intercept_old = intercept;
    const VectorXd B_old = B;

    //
    // Compute derivative of f
    //
    const VectorXd cache_d = y - intercept_old * One_n - X * B_old; // compute only once for speed

    // Compute the derivative of intercept with no penalization
    double Df_intercept = -1 * One_n.transpose() * cache_d;

    // Compute derivative of the rest: Derivative of loss + differentiable penalizations
    VectorXd Df = VectorXd::Zero(p);
    for (int i = 0; i < p; i++) {
      Df(i) = -1 * X.col(i).transpose() * cache_d
                + alpha(1) * lambda * B_old(i) + alpha(2) * lambda * pow(B_old(i), 3) 
                + alpha(3) * lambda * pow(B_old(i), 5) + alpha(4) * lambda * pow(B_old(i), 7) + alpha(5) * lambda * pow(B_old(i), 9);
    }

    // Line search
    double h_j = 0.5; // initial step size
    bool line_searching = true;
    while (line_searching) {
      // Gradient step for the intercept
      intercept = intercept_old - h_j * Df_intercept;

      // Gradient step for the rest
      for (int i = 0; i < p; i++) {
        // Proximal Mapping of l1: Soft Thresholding
        const double v_i = B_old(i) - h_j * Df(i); // gradient step
        if (v_i < -h_j * alpha(0) * lambda) {
          B(i) = v_i + h_j * alpha(0) * lambda;
        } else if (v_i >= -h_j * alpha(0) * lambda && v_i <= h_j * alpha(0) * lambda) {
          B(i) = 0;
        } else if (v_i > h_j * alpha(0) * lambda) {
          B(i) = v_i - h_j * alpha(0) * lambda;
        }
      }

      // Line search criterion from Boyd: Proximal algorithms
      const double f_B = (y - intercept * One_n - (X * B)).squaredNorm();
      const double f_B_old = cache_d.squaredNorm();
      if ( f_B <= f_B_old + Df_intercept * (intercept - intercept_old) + Df.transpose() * (B - B_old) + 1.0/((double)2.0 * h_j) * (pow(intercept - intercept_old, 2) + (B - B_old).squaredNorm()) ) {
        // If we are descenting, break out of line search
        line_searching = false;
      } else if (h_j == 0){
        // We cannot descent, so we have converged and break out of line search
        line_searching = false;
      } else {
        h_j = 0.5 * h_j; // half step size
      } 
    }

    // Stopping criterion
    if ( pow(intercept - intercept_old, 2) + ((B - B_old)).squaredNorm() < tolerance ) {
      // Build return value
      fit.intercept = intercept;
      fit.B = B;
      return fit;
    }
  }

  cout << "Failed to converge!\n";
  // Build return value
  fit.intercept = intercept;
  fit.B = B;
  return fit;
}

// Returns a vector of B corresponding to lambdas using warm-start.
// We do not sort the lambdas here, they are ordered how you want them
vector<FitType> fit_regression_warm_start_proximal_gradient(const MatrixXd& X, const VectorXd& y, 
                                         const Matrix<double, 6, 1>& alpha, const vector<double>& lambdas,
                                         const int max_iter, const double tolerance) {
  const int p = X.cols();
  const int L = lambdas.size();
  vector<FitType> fit_vector;

  // do the first one normally
  VectorXd B_0 = VectorXd::Zero(p);
  fit_vector.push_back(fit_regression_proximal_gradient(B_0, X, y, alpha, lambdas[0], max_iter, tolerance));

  // Warm start after the first one
  for (int l = 1; l < L; l++) {
    const FitType fit_warm = fit_vector.at(l - 1); // warm start
    const VectorXd B_warm = fit_warm.B;
    fit_vector.push_back(fit_regression_proximal_gradient(B_warm, X, y, alpha, lambdas[l], max_iter, tolerance));
  }
  return fit_vector;
}

CVType cross_validation_regression_proximal_gradient(const MatrixXd& X, const VectorXd& y, 
                                             const double K_fold, const Matrix<double, 6, 1>& alpha, const vector<double>& arg_lambdas,
                                             const int max_iter, const double tolerance, const int random_seed) {
  const int n = X.rows();
  const int p = X.cols();
  vector<double> lambdas = arg_lambdas; // copy argument
  const int L = lambdas.size();
  MatrixXd test_risks_matrix = MatrixXd::Zero(L, K_fold);

  sort(lambdas.begin(), lambdas.end(), std::greater<double>()); // sort the lambdas in place descending

  // Create random permutation using the Mersenne twister random number engine 64 bit
  vector<int> I(n);
  std::iota (std::begin(I), std::end(I), 0);
  std::mt19937_64 rng(random_seed);
  std::shuffle(std::begin(I), std::end(I), rng); // permute

  vector<vector<int>> partitions = partition(I, K_fold);
  for (size_t k = 0; k < partitions.size(); k++) {
    vector<int> TEST = partitions[k];

    // Build training indices
    vector<int> TRAIN;
    for (int& i : I) {
      bool exists = false;
      for (int& j : TEST) {
        if (j == i) {
          exists = true;
        }
      }
      if (exists == false) {
        TRAIN.push_back(i);
      }
    }

    // Build X_train, y_train
    MatrixXd X_train = MatrixXd(TRAIN.size(), p);
    VectorXd y_train = VectorXd(TRAIN.size());
    for (size_t i = 0; i < TRAIN.size(); i++) {
      X_train.row(i) = X.row(TRAIN[i]);
      y_train.row(i) = y.row(TRAIN[i]);
    }

    // Build X_test, y_test
    MatrixXd X_test = MatrixXd(TEST.size(), p);
    VectorXd y_test = VectorXd(TEST.size());
    for (size_t i = 0; i < TEST.size(); i++) {
      X_test.row(i) = X.row(TEST[i]);
      y_test.row(i) = y.row(TEST[i]);
    }

    // Do the computation
    const vector<FitType> fit_vector = fit_regression_warm_start_proximal_gradient(X_train, y_train, alpha, lambdas, max_iter, tolerance);
    for (size_t l = 0; l < fit_vector.size(); l++) {
      const FitType fit = fit_vector.at(l);
      const VectorXd B = fit.B;
      const double intercept = fit.intercept;

      test_risks_matrix(l, k) = mean_squared_error(y_test, predict_regression(X_test, intercept, B));
    }
  }

  // build return
  CVType cv;
  cv.risks = test_risks_matrix.rowwise().mean();
  cv.lambdas = lambdas;
  return cv;
}

/*
================================

Lnet Proximal Gradient Binary Classification

================================
*/

// Notes: We purposefully use doubles instead of integers everywhere to avoid any conversion.

double sigmoid(const double x) {
  return pow(1 + exp(-1 * x), -1);
}

// We purposefully use doubles
double accuracy(const VectorXd& v, const VectorXd& w) {
  if (v.rows() != w.rows()) {
    throw std::invalid_argument("Vectors must be the same length.");
  }
  const int n = v.rows();
  double m = 0;
  for (int i = 0; i < n; i++) {
    m += ((int)v(i) == (int)w(i));
  }
  return 1.0/((double)n) * m;
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


FitType fit_logistic_proximal_gradient(const VectorXd& B_0, const MatrixXd& X, const VectorXd& y, 
                              const Matrix<double, 6, 1>& alpha, const double lambda,
                              const int max_iter, const double tolerance) {
  FitType fit; // return value

  const int n = X.rows();
  const int p = X.cols();
  const VectorXd One_n = VectorXd::Ones(n);

  // Create the values to optimize
  double intercept = 0;
  VectorXd B = B_0;
  for (int j = 0; j < max_iter; j++) {
    // Keep track of the current values
    const double intercept_old = intercept;
    const VectorXd B_old = B;

    // Compute once and store
    VectorXd sigmoid_intercept_XB_old = VectorXd::Zero(n);
    for (int i = 0; i < n; i++) {
      sigmoid_intercept_XB_old(i) = sigmoid(intercept + X.row(i) * B_old);
    }
    const VectorXd cache_d = (y - sigmoid_intercept_XB_old); // for speed

    //
    // Compute derivative of f
    //
    double Df_intercept = One_n.transpose() * cache_d;

    VectorXd Df = VectorXd::Zero(p); // Derivative of loss + differentiable penalizations
    for (int i = 0; i < p; i++) {
      Df(i) = X.col(i).transpose() * cache_d
              + alpha(1) * lambda * B_old(i) + alpha(2) * lambda * pow(B_old(i), 3) 
              + alpha(3) * lambda * pow(B_old(i), 5) + alpha(4) * lambda * pow(B_old(i), 7) + alpha(5) * lambda * pow(B_old(i), 9);
    }

    // Line search
    double h_j = .5; // initial step size
    bool line_searching = true;
    while (line_searching) {
      // Gradient step for the intercept
      intercept = intercept_old + h_j * Df_intercept;

      // Gradient step for the rest
      for (int i = 0; i < p; i++) {
        // Proximal Mapping: Soft Thresholding
        const double v_i = B_old(i) + h_j * Df(i); // gradient step
        if (v_i < -h_j * alpha(0) * lambda) {
          B(i) = v_i + h_j * alpha(0) * lambda;
        } else if (v_i >= -h_j * alpha(0) * lambda && v_i <= h_j * alpha(0) * lambda) {
          B(i) = 0;
        } else if (v_i > h_j * alpha(0) * lambda) {
          B(i) = v_i - h_j * alpha(0) * lambda;
        }
      }

      // Check if the the step size is small enough for descent
      double f_B_old = 0;
      double f_B = 0;
      for (int i = 0; i < n; i++) {
        f_B_old += y(i) * log(sigmoid_intercept_XB_old(i)) + (1 - y(i)) * log(1 - sigmoid_intercept_XB_old(i));
        f_B += y(i) * log(sigmoid(intercept + X.row(i) * B)) + (1 - y(i)) * log(1 - sigmoid(intercept + X.row(i) * B));
      }
      if ( -f_B <= -f_B_old - Df_intercept * (intercept - intercept_old) - Df.transpose() * (B - B_old) + 1.0/((double)2.0 * h_j) * ( pow(intercept - intercept_old, 2) + (B - B_old).squaredNorm() ) ) {
        // If we are descenting, break out of line search
        line_searching = false;
      } else if (h_j == 0){
        // We cannot descent, so we have converged and break out of line search
        line_searching = false;
      } else {
        h_j = 0.5 * h_j; // half step size
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

  std::cout << "Failed to converge!\n";
  // Build return
  fit.intercept = intercept;
  fit.B = B;
  return fit;
}

// Returns a vector of B corresponding to lambdas using warm-start.
// We do not sort the lambdas here, they are ordered how you want them
vector<FitType> fit_logistic_warm_start_proximal_gradient(const MatrixXd& X, const VectorXd& y, 
                                         const Matrix<double, 6, 1>& alpha, const vector<double>& lambdas,
                                         const int max_iter, const double tolerance) {
  const int p = X.cols();
  const int L = lambdas.size();
  vector<FitType> fit_vector;

  // do the first one normally
  VectorXd B_0 = VectorXd::Zero(p);
  fit_vector.push_back(fit_logistic_proximal_gradient(B_0, X, y, alpha, lambdas[0], max_iter, tolerance));

  // Warm start after the first one
  for (int l = 1; l < L; l++) {
    const FitType fit_warm = fit_vector.at(l - 1); // warm start
    const VectorXd B_warm = fit_warm.B;
    fit_vector.push_back(fit_logistic_proximal_gradient(B_warm, X, y, alpha, lambdas[l], max_iter, tolerance));
  }
  return fit_vector;
}


CVType cross_validation_logistic_proximal_gradient(const MatrixXd& X, const VectorXd& y, 
                                             const double K_fold, const Matrix<double, 6, 1>& alpha, const vector<double>& arg_lambdas,
                                             const int max_iter, const double tolerance, const int random_seed) {
  const int n = X.rows();
  const int p = X.cols();
  vector<double> lambdas = arg_lambdas; // copy argument
  const int L = lambdas.size();
  MatrixXd test_risks_matrix = MatrixXd::Zero(L, K_fold);

  sort(lambdas.begin(), lambdas.end(), std::greater<double>()); // sort the lambdas in place descending

  // Create random permutation using the Mersenne twister random number engine 64 bit
  vector<int> I(n);
  std::iota (std::begin(I), std::end(I), 0);
  std::mt19937_64 rng(random_seed);
  std::shuffle(std::begin(I), std::end(I), rng); // permute

  vector<vector<int>> partitions = partition(I, K_fold);
  for (size_t k = 0; k < partitions.size(); k++) {
    vector<int> TEST = partitions[k];

    // Build training indices
    vector<int> TRAIN;
    for (int& i : I) {
      bool exists = false;
      for (int& j : TEST) {
        if (j == i) {
          exists = true;
        }
      }
      if (exists == false) {
        TRAIN.push_back(i);
      }
    }

    // Build X_train, y_train
    MatrixXd X_train = MatrixXd(TRAIN.size(), p);
    VectorXd y_train = VectorXd(TRAIN.size());
    for (size_t i = 0; i < TRAIN.size(); i++) {
      X_train.row(i) = X.row(TRAIN[i]);
      y_train.row(i) = y.row(TRAIN[i]);
    }

    // Build X_test, y_test
    MatrixXd X_test = MatrixXd(TEST.size(), p);
    VectorXd y_test = VectorXd(TEST.size());
    for (size_t i = 0; i < TEST.size(); i++) {
      X_test.row(i) = X.row(TEST[i]);
      y_test.row(i) = y.row(TEST[i]);
    }

    // Do the computation
    const vector<FitType> fit_vector = fit_logistic_warm_start_proximal_gradient(X_train, y_train, alpha, lambdas, max_iter, tolerance);
    for (size_t l = 0; l < fit_vector.size(); l++) {
      const FitType fit = fit_vector.at(l);
      test_risks_matrix(l, k) = accuracy(y_test, predict_class(X_test, fit.intercept, fit.B));
    }
  }

  // build return
  CVType cv;
  cv.risks = test_risks_matrix.rowwise().mean();
  cv.lambdas = lambdas;
  return cv;
}































/*
================================

Lnet Classification Coordinate Descent

Not used.
================================
*/

/*
Logistic proximal gradient coordinate descent
*/
FitType fit_logistic_proximal_gradient_coordinate_descent(const VectorXd& B_0, const MatrixXd& X, const VectorXd& y, 
                              const Matrix<double, 6, 1>& alpha, const double lambda, const double step_size,
                              const int max_iter, const double tolerance) {
  FitType fit; // return value

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





/*
================================

Lnet Regression Coordinate Descent

Not used.

================================
*/


/*

Proximal Gradient Coordinate Descent

*/
FitType fit_regression_proximal_gradient_cd(const VectorXd& B_0, const MatrixXd& X, const VectorXd& y, 
                              const Matrix<double, 6, 1>& alpha, const double lambda, const double step_size,
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

// Returns a vector of B corresponding to lambdas using warm-start.
// We do not sort the lambdas here, they are ordered how you want them
vector<FitType> fit_regression_warm_start_proximal_gradient_cd(const MatrixXd& X, const VectorXd& y, 
                                         const Matrix<double, 6, 1>& alpha, const vector<double>& lambdas, const double step_size,
                                         const int max_iter, const double tolerance, const int random_seed) {
  const int p = X.cols();
  const int L = lambdas.size();
  vector<FitType> fit_vector;

  // do the first one normally
  VectorXd B_0 = VectorXd::Zero(p);
  fit_vector.push_back(fit_regression_proximal_gradient_cd(B_0, X, y, alpha, lambdas[0], step_size, max_iter, tolerance, random_seed));

  // Warm start after the first one
  for (int l = 1; l < L; l++) {
    const FitType fit_warm = fit_vector.at(l - 1); // warm start
    const VectorXd B_warm = fit_warm.B;
    fit_vector.push_back(fit_regression_proximal_gradient_cd(B_warm, X, y, alpha, lambdas[l], step_size, max_iter, tolerance, random_seed));
  }
  return fit_vector;
}

// Prox Gradient Cross Validation
CVType cross_validation_regression_proximal_gradient_cd(const MatrixXd& X, const VectorXd& y, 
                                             const double K_fold, const Matrix<double, 6, 1>& alpha, const vector<double>& arg_lambdas, const double step_size,
                                             const int max_iter, const double tolerance, const int random_seed) {
  int n = X.rows();
  int p = X.cols();
  vector<double> lambdas = arg_lambdas; // copy argument
  int L = lambdas.size();
  MatrixXd test_risks_matrix = MatrixXd::Zero(L, K_fold);

  sort(lambdas.begin(), lambdas.end(), std::greater<double>()); // sort the lambdas in place descending

  // Create random permutation using the Mersenne twister random number engine 64 bit
  vector<int> I(n);
  std::iota (std::begin(I), std::end(I), 0);
  std::mt19937_64 rng(random_seed);
  std::shuffle(std::begin(I), std::end(I), rng); // permute

  vector<vector<int>> partitions = partition(I, K_fold);
  for (size_t k = 0; k < partitions.size(); k++) {
    vector<int> TEST = partitions[k];

    // Build training indices
    vector<int> TRAIN;
    for (int& i : I) {
      bool exists = false;
      for (int& j : TEST) {
        if (j == i) {
          exists = true;
        }
      }
      if (exists == false) {
        TRAIN.push_back(i);
      }
    }

    // Build X_train, y_train
    MatrixXd X_train = MatrixXd(TRAIN.size(), p);
    VectorXd y_train = VectorXd(TRAIN.size());
    for (size_t i = 0; i < TRAIN.size(); i++) {
      X_train.row(i) = X.row(TRAIN[i]);
      y_train.row(i) = y.row(TRAIN[i]);
    }

    // Build X_test, y_test
    MatrixXd X_test = MatrixXd(TEST.size(), p);
    VectorXd y_test = VectorXd(TEST.size());
    for (size_t i = 0; i < TEST.size(); i++) {
      X_test.row(i) = X.row(TEST[i]);
      y_test.row(i) = y.row(TEST[i]);
    }

    // Do the computation
    const vector<FitType> fit_vector = fit_regression_warm_start_proximal_gradient_cd(X_train, y_train, alpha, lambdas, step_size, max_iter, tolerance, random_seed);
    for (size_t l = 0; l < fit_vector.size(); l++) {
      const FitType fit = fit_vector.at(l);
      const VectorXd B = fit.B;
      const double intercept = fit.intercept;

      test_risks_matrix(l, k) = mean_squared_error(y_test, predict_regression(X_test, intercept, B));
    }
  }

  // build return
  CVType cv;
  cv.risks = test_risks_matrix.rowwise().mean();
  cv.lambdas = lambdas;
  return cv;
}
