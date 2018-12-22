#include "pros.h"

#include <time.h>
#include <iostream>
#include <fstream>
#include <iomanip>

using std::cout;

//
// Test
//

// CV parser
template<typename M>
M load_csv (const std::string & path) {
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

//
// Test
//
void test_prostate() {
  MatrixXd X_train = load_csv<MatrixXd>("data/prostate_X_train.csv");
  VectorXd y_train = load_csv<MatrixXd>("data/prostate_y_train.csv");

  MatrixXd X_test = load_csv<MatrixXd>("data/prostate_X_test.csv");
  VectorXd y_test = load_csv<MatrixXd>("data/prostate_y_test.csv");
  VectorXd B_0 = VectorXd::Zero(X_train.cols());

  int n_train = X_train.rows();

  int K_fold = 10;
  double lambda = 11;
  double step_size = 1/((double) 80);
  int max_iter = 10000;
  double tolerance = pow(10, -8);
  int random_seed = 0;

  // create alpha
  double alpha_data[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  Map<Vector6d> alpha(alpha_data);

  // create lambdas
  vector<double> lambdas;
  lambdas.push_back(X_train.cols());
  for (int i = 1; i < 100; i++) {
    lambdas.push_back(lambdas[i - 1] + .1);
  }

  //
  // Single fit test
  //
  cout << "\nProstate Single fit test\n";
  VectorXd B = proximal_gradient_cd(B_0, X_train, y_train, alpha, lambda, step_size, max_iter, tolerance, random_seed);
  cout << "\nB:\n" << B << "\n";

  double intercept = 1/((double)n_train) *  VectorXd::Ones(n_train).transpose() * (y_train.mean() * VectorXd::Ones(n_train) - (X_train * B)); // mean
  cout << "\nintercept:\n" << intercept << "\n";

  cout << "\nMSE: " << mean_squared_error(y_train, predict(B, intercept, X_train)) << "\n";
  cout << "\nTest MSE: " << mean_squared_error(y_test, predict(B, intercept, X_test)) << "\n";

  //
  // Warm start test
  //
  cout << "\nWarm start test\n";
  MatrixXd B_matrix = warm_start_proximal_gradient_cd(X_train, y_train, alpha, lambdas, step_size, max_iter, tolerance, random_seed);
  cout << "\nB Matrix last:\n" << B_matrix.row(B_matrix.rows() - 1).transpose() << "\n";

  //
  // CV test
  //
  cout << "\nCV test\n";
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

  VectorXd B_best = proximal_gradient_cd(B_0, X_train, y_train, alpha, best_lambda, step_size, max_iter, tolerance, random_seed);
  double intercept_best = 1/((double)n_train) *  VectorXd::Ones(n_train).transpose() * (y_train.mean() * VectorXd::Ones(n_train) - (X_train * B_best));
  cout << "\nTest MSE: " << mean_squared_error(y_test, predict(B_best, intercept_best, X_test)) << "\n";


}

void test_prox() {
  //std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1); // set precision

  MatrixXd X_train = load_csv<MatrixXd>("data/X_train.csv");
  VectorXd y_train = load_csv<MatrixXd>("data/y_train.csv");
  MatrixXd X_test = load_csv<MatrixXd>("data/X_test.csv");
  VectorXd y_test = load_csv<MatrixXd>("data/y_test.csv");
  VectorXd B_0 = VectorXd::Zero(X_train.cols());

  int K_fold = 10;
  double lambda = .1;
  double step_size = 1/((double) 10000);
  int max_iter = 10000;
  double tolerance = pow(10, -3);
  int random_seed = 0;

  // create alpha
  double alpha_data[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  Map<Vector6d> alpha(alpha_data);

  // create lambdas
  vector<double> lambdas;
  lambdas.push_back(pow(10, -3));
  for (int i = 1; i < 10; i++) {
    lambdas.push_back(lambdas[i - 1] + .1);
  }


  // -----------------
  // Proximal Gradient Testing
  // -----------------

  //
  // Single fit test
  //
  cout << "\nSingle fit test\n";
  VectorXd B = proximal_gradient_cd(B_0, X_train, y_train, alpha, lambda, step_size, max_iter, tolerance, random_seed);

  int n = X_train.rows();
  double intercept = 1/((double)n) *  VectorXd::Ones(n).transpose() * (y_train.mean() * VectorXd::Ones(n) - (X_train * B));
  cout << "\nintercept:\n" << intercept << "\n";

  cout << "\nB:\n" << B << "\n";
  cout << "\ntest MSE: " << mean_squared_error(y_test, predict(B, intercept, X_test)) << "\n";

  
  //
  // Warm start test
  //
  cout << "\nWarm start test\n";
  MatrixXd B_matrix = warm_start_proximal_gradient_cd(X_train, y_train, alpha, lambdas, step_size, max_iter, tolerance, random_seed);
  cout << "\nB Matrix last:\n" << B_matrix.row(B_matrix.rows() - 1).transpose() << "\n";

  //
  // CV test
  //
  cout << "\nCV test\n";
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
}

// Benchmark for compiler optimization
void bench() {
  MatrixXf a = MatrixXf::Random(5000, 5000);
  MatrixXf b = MatrixXf::Random(5000, 5000);
  time_t start = clock();
  MatrixXf c = a * b;
  std::cout << (double)(clock() - start) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;
}

void test_random_gen() {
  int n = 10;
  vector<int> I(n);
  std::iota (std::begin(I), std::end(I), 0);
  std::random_device rd;
  cout << rd();
  std::seed_seq random_seed{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
  std::mt19937_64 rng(random_seed);
  time_t start = clock();
  std::shuffle(std::begin(I), std::end(I), rng); // permute
  std::cout << (double)(clock() - start) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;

  cout << "\n";
  for (auto& i : I) {
    cout << i << " ";
  }
  cout << "\n";
}

int main() {
  test_prostate();
  // test_prox();
  // test_random_gen();
  // bench();
}

