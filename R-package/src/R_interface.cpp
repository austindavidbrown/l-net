// Libraries here ORDER MATTERS
#include <Eigen/Dense>
#include <vector>
#include "pros.h"

// R stuff here ORDER MATTERS
#include <R.h>
#include <Rinternals.h>

using namespace Eigen;
using std::vector;

// This allows visibility
extern "C" {


SEXP R_fit(SEXP X_, SEXP y_, 
           SEXP alpha_, SEXP lambda_, SEXP step_size_,
           SEXP max_iter_, SEXP tolerance_, SEXP random_seed_){
  SEXP res;

  // Handle X
  SEXP dim_X = getAttrib(X_, R_DimSymbol);
  const int nrow_X = INTEGER(dim_X)[0];
  const int ncol_X = INTEGER(dim_X)[1];
  double* p_X = REAL(X_); // pointer

  // Handle y
  SEXP dim_y = getAttrib(y_, R_DimSymbol);
  const int nrow_y = INTEGER(dim_y)[0];
  double* p_y = REAL(y_); // pointer

  // Handle alpha
  double* p_alpha = REAL(alpha_); // pointer

  // Handle lambda
  double lambda = REAL(lambda_)[0];

  // Handle step size
  double step_size = REAL(step_size_)[0];

  // Handle max_iter
  const int max_iter = INTEGER(max_iter_)[0];

  // Handle tolerance
  const double tolerance = REAL(tolerance_)[0];

  // Handle random seed
  const int random_seed = INTEGER(random_seed_)[0];

  // Setup
  Map<Matrix<double, Dynamic, Dynamic, ColMajor>> X(p_X, nrow_X, ncol_X); // R is laid out in memory column major
  Map<VectorXd> y(p_y, nrow_y);
  Map<Vector6d> alpha(p_alpha);

  // fit
  VectorXd B_0 = VectorXd::Zero(X.cols());
  VectorXd B = proximal_gradient_cd(B_0, X, y, alpha, lambda, step_size, max_iter, tolerance, random_seed);

  // compute intercept
  int n = X.rows();
  double intercept = 1/((double)n) *  VectorXd::Ones(n).transpose() * (y.mean() * VectorXd::Ones(n) - (X * B));


  //
  // Copy to R
  //
  const char *names[] = {"B", "intercept", ""};
  res = PROTECT(mkNamed(VECSXP, names)); // create response

  // Copy B
  SEXP res_B;
  PROTECT(res_B = Rf_allocVector(REALSXP, B.rows()));
  for (int i = 0; i < B.rows(); i++) {
    REAL(res_B)[i] = B(i);
  }
  SET_VECTOR_ELT(res, 0, res_B);

  // Copy intercept
  SET_VECTOR_ELT(res, 1, ScalarReal(intercept));

  UNPROTECT(2);
  return res;
}

SEXP R_cross_validation(SEXP X_, SEXP y_, 
                        SEXP K_fold_, SEXP alpha_, SEXP lambdas_, SEXP step_size_, 
                        SEXP max_iter_, SEXP tolerance_, SEXP random_seed_){  
  SEXP res;

  // Handle X
  SEXP dim_X = getAttrib(X_, R_DimSymbol);
  const int nrow_X = INTEGER(dim_X)[0];
  const int ncol_X = INTEGER(dim_X)[1];
  double* p_X = REAL(X_); // pointer

  // Handle y
  SEXP dim_y = getAttrib(y_, R_DimSymbol);
  const int nrow_y = INTEGER(dim_y)[0];
  double* p_y = REAL(y_); // pointer

  // Handle alpha
  double* p_alpha = REAL(alpha_); // pointer

  // Handle lambdas
  const int L = Rf_length(lambdas_);

  // Handle step size
  double step_size = REAL(step_size_)[0];

  // Handle max_iter
  const int max_iter = INTEGER(max_iter_)[0];

 // Handle tolerance
  const double tolerance = REAL(tolerance_)[0];

  // Handle random seed
  const int random_seed = INTEGER(random_seed_)[0];

  // Setup
  Map<Matrix<double, Dynamic, Dynamic, ColMajor>> X(p_X, nrow_X, ncol_X);
  Map<VectorXd> y(p_y, nrow_y);
  double K_fold = REAL(K_fold_)[0];
  Map<Vector6d> alpha(p_alpha);

  // Build lambdas
  vector<double> lambdas;
  lambdas.assign(REAL(lambdas_), REAL(lambdas_) + L);

  // CV
  CVType cv = cross_validation_proximal_gradient_cd(X, y, K_fold, alpha, lambdas, step_size, max_iter, tolerance, random_seed);
  vector<double> cv_lambdas = cv.lambdas;
  VectorXd cv_risks = cv.risks;

  // get location of minimum
  MatrixXf::Index min_row;
  cv.risks.minCoeff(&min_row);
  double best_lambda = cv.lambdas[min_row];

  //
  // Copy to R
  //
  const char *names[] = {"best_lambda", "lambdas", "risks", ""};
  res = PROTECT(mkNamed(VECSXP, names)); // create response

  SET_VECTOR_ELT(res, 0, ScalarReal(best_lambda));

  // Copy lambdas
  SEXP res_lambdas;
  PROTECT(res_lambdas = Rf_allocVector(REALSXP, cv_lambdas.size()));
  for (size_t i = 0; i < cv_lambdas.size(); i++) {
    REAL(res_lambdas)[i] = cv_lambdas[i];
  }
  SET_VECTOR_ELT(res, 1, res_lambdas);

  // Copy Risks
  SEXP res_risks;
  PROTECT(res_risks = Rf_allocVector(REALSXP, cv_risks.rows()));
  for (int i = 0; i < cv_risks.rows(); i++) {
    REAL(res_risks)[i] = cv_risks(i);
  }
  SET_VECTOR_ELT(res, 2, res_risks);

  UNPROTECT(3);
  return res;
}

SEXP R_predict(SEXP B_, SEXP intercept_, SEXP X_){
  SEXP result;

  // Handle X
  SEXP dim_X = getAttrib(X_, R_DimSymbol) ;
  const int nrow_X = INTEGER(dim_X)[0];
  const int ncol_X = INTEGER(dim_X)[1];
  double* p_X = REAL(X_); // pointer

  // Handle B
  SEXP dim_B = getAttrib(B_, R_DimSymbol);
  const int nrow_B = INTEGER(dim_B)[0];
  double* p_B = REAL(B_); // pointer

  // Setup
  Map<Matrix<double, Dynamic, Dynamic, ColMajor>> X(p_X, nrow_X, ncol_X);
  Map<VectorXd> B(p_B, nrow_B);
  double intercept = REAL(intercept_)[0];

  // Predict
  VectorXd pred = predict(B, intercept, X);

  //
  // Copy to R
  //
  PROTECT(result = Rf_allocVector(REALSXP, pred.rows()));

  // Copy
  for (int i = 0; i < pred.rows(); i++) {
    REAL(result)[i] = pred(i);
  }

  UNPROTECT(1);
  return result;
}

}

