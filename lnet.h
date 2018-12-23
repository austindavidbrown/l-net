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

// TODO

// Change return type to fit function

// long double matrix, vector, scalar
typedef long double ld;
typedef Matrix<long double, Dynamic, Dynamic> MatrixXld;
typedef Matrix<long double, Dynamic, 1> VectorXld;

typedef Matrix<double, 6, 1> Vector6d;

struct CVType {
  VectorXd risks;
  vector<double> lambdas;
};

//
// Utils
//
double mean_squared_error(const VectorXd& v, const VectorXd& w) {
  return 1/((double)v.rows()) * (v - w).squaredNorm(); 
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


// Proximal Gradient Coordinate Descent
/// =====================================================================================

VectorXd predict(const MatrixXd& X, const double intercept, const VectorXd& B) {
  const int n = X.rows();
  return intercept * VectorXd::Ones(n) + (X * B);
}

//
// Proximal Gradient Coordinate Descent
//
VectorXd fit_proximal_gradient_cd(const VectorXd& B_0, const MatrixXd& X, const VectorXd& y, 
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
      return B;
    }
  }

  std::cout << "Failed to converge! Tune the step size.\n";
  return B;
}

// Returns a vector of B corresponding to lambdas using warm-start.
// We do not sort the lambdas here, they are ordered how you want them
vector<VectorXd> fit_warm_start_proximal_gradient_cd(const MatrixXd& X, const VectorXd& y, 
                                         const Vector6d& alpha, const vector<double>& lambdas, const double step_size,
                                         const int max_iter, const double tolerance, const int random_seed) {
  const int p = X.cols();
  const int L = lambdas.size();
  vector<VectorXd> B_vector;

  // do the first one normally
  VectorXd B_0 = VectorXd::Zero(p);
  B_vector.push_back(fit_proximal_gradient_cd(B_0, X, y, alpha, lambdas[0], step_size, max_iter, tolerance, random_seed));

  // Warm start after the first one
  for (int l = 1; l < L; l++) {
    VectorXd B_warm = B_vector.at(l - 1); // warm start
    B_vector.push_back(fit_proximal_gradient_cd(B_warm, X, y, alpha, lambdas[l], step_size, max_iter, tolerance, random_seed));
  }
  return B_vector;
}

// Prox Gradient Cross Validation
CVType cross_validation_proximal_gradient_cd(const MatrixXd& X, const VectorXd& y, 
                                             const double K_fold, const Vector6d& alpha, const vector<double>& arg_lambdas, const double step_size,
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

    // So the computation
    const vector<VectorXd> B_vector = fit_warm_start_proximal_gradient_cd(X_train, y_train, alpha, lambdas, step_size, max_iter, tolerance, random_seed);
    for (size_t l = 0; l < B_vector.size(); l++) {
      const VectorXd B = B_vector.at(l);

      // compute the intercept
      int n = X_train.rows();
      double intercept = 1/((double)n) *  VectorXd::Ones(n).transpose() * (y_train.mean() * VectorXd::Ones(n) - (X_train * B));

      test_risks_matrix(l, k) = mean_squared_error(y_test, predict(X_test, intercept, B));
    }
  }

  // build return
  CVType cv;
  cv.risks = test_risks_matrix.rowwise().mean();
  cv.lambdas = lambdas;
  return cv;
}

