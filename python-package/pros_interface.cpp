#include <math.h>
#include <vector>
#include <iostream>

#include <Python.h>
#include <numpy/arrayobject.h>

#include <Eigen/Dense>
#include "pros.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // use new numpy api

using namespace Eigen;
using std::cout;
using std::vector;

static PyObject* python_fit(PyObject *self, PyObject *args, PyObject* kwargs) {
  char* keywords[] = {"X", "y", "alpha", "lambda_", "step_size", 
                      "max_iter", "tolerance", "random_seed", NULL};

  // Required arguments
  PyArrayObject* arg_y = NULL;
  PyArrayObject* arg_X = NULL;
  PyArrayObject* arg_alpha = NULL;
  double arg_lambda;
  double arg_step_size;

  // Arguments with default values
  int arg_max_iter = 10000;
  double arg_tolerance = pow(10, -8);
  int arg_random_seed = 0;

  // Parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!dd|idi", keywords,
                        &PyArray_Type, &arg_X, &PyArray_Type, &arg_y,
                        &PyArray_Type, &arg_alpha, &arg_lambda, &arg_step_size, 
                        &arg_max_iter, &arg_tolerance, &arg_random_seed)) {
    return NULL;
  }

  // Handle X argument
  arg_X = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_X), NPY_DOUBLE, NPY_IN_ARRAY));
  double* ptr_arg_X = reinterpret_cast<double*>(arg_X->data);
  const int nrow_X = (arg_X->dimensions)[0];
  const int ncol_X = (arg_X->dimensions)[1];

  // Handle y argument
  arg_y = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_y), NPY_DOUBLE, NPY_IN_ARRAY));
  double* ptr_arg_y = reinterpret_cast<double*>(arg_y->data);
  const int nrow_y = (arg_y->dimensions)[0];

  // Handle alpha argument
  arg_alpha = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_alpha), NPY_DOUBLE, NPY_IN_ARRAY));
  double* ptr_arg_alpha = reinterpret_cast<double*>(arg_alpha->data);

  // Setup
  const Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(ptr_arg_X, nrow_X, ncol_X);
  const Map<VectorXd> y(ptr_arg_y, nrow_y);
  const Map<Vector6d> alpha(ptr_arg_alpha);

  // Fit
  const VectorXd B_0 = VectorXd::Zero(X.cols());
  VectorXd B = proximal_gradient_cd(B_0, X, y, alpha, arg_lambda, arg_step_size, arg_max_iter, arg_tolerance, arg_random_seed);

  // Compute intercept
  const int n = X.rows();
  const double intercept = 1/((double)n) *  VectorXd::Ones(n).transpose() * (y.mean() * VectorXd::Ones(n) - (X * B));

  //
  // Copy to Python
  //
  long res_dims[1];
  res_dims[0] = B.rows();
  PyArrayObject* res = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, res_dims, NPY_DOUBLE)); // 1 is for vector
  double* ptr_res_data = (reinterpret_cast<double*>(res->data));

  for (int i = 0; i < B.rows(); i++) {
    ptr_res_data[i] = B(i);
  }
  return Py_BuildValue("Od", res, intercept);
}

static PyObject* python_predict(PyObject *self, PyObject *args, PyObject* kwargs) {
  char* keywords[] = {"X", "intercept", "B", NULL};

  // Required arguments
  PyArrayObject* arg_X = NULL;
  double arg_intercept;
  PyArrayObject* arg_B = NULL;

  // Parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!dO!", keywords, 
                                   &PyArray_Type, &arg_X,  &arg_intercept, &PyArray_Type, &arg_B)) {
    return NULL;
  }

  // Handle X argument
  arg_X = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_X), NPY_DOUBLE, NPY_IN_ARRAY));
  double* ptr_arg_X = reinterpret_cast<double*>(arg_X->data);
  const int nrow_X = (arg_X->dimensions)[0];
  const int ncol_X = (arg_X->dimensions)[1];

  // Handle B argument
  arg_B = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_B), NPY_DOUBLE, NPY_IN_ARRAY));
  double* ptr_arg_B = reinterpret_cast<double*>(arg_B->data);
  const int nrow_B = (arg_B->dimensions)[0];

  // Setup
  const Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(ptr_arg_X, nrow_X, ncol_X);
  const Map<VectorXd> B(ptr_arg_B, nrow_B);

  // Predict
  const VectorXd pred = predict(B, arg_intercept, X);

  //
  // Copy to Python
  //
  long res_dims[1];
  res_dims[0] = pred.rows();
  PyArrayObject* res = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, res_dims, NPY_DOUBLE)); // 1 is for vector
  double* ptr_res_data = (reinterpret_cast<double*>(res->data));

  for (int i = 0; i < pred.rows(); i++) {
    ptr_res_data[i] = pred(i);
  }

  return Py_BuildValue("O", res);
}

static PyObject* python_cross_validation(PyObject *self, PyObject *args, PyObject* kwargs) {
  char* keywords[] = {"X", "y", "alpha", "lambdas", "step_size", 
                      "K_fold", "max_iter", "tolerance", "random_seed", NULL};

  // Required arguments
  PyArrayObject* arg_y = NULL;
  PyArrayObject* arg_X = NULL;
  PyArrayObject* arg_alpha = NULL;
  PyArrayObject* arg_lambdas = NULL;
  double arg_step_size;

  // Arguments with default values
  int arg_K_fold = 10;
  int arg_max_iter = 10000;
  double arg_tolerance = pow(10, -8);
  int arg_random_seed = 0;

  // Parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!O!d|iidi", keywords,
                        &PyArray_Type, &arg_X, &PyArray_Type, &arg_y,
                        &PyArray_Type, &arg_alpha, &PyArray_Type, &arg_lambdas, &arg_step_size, 
                        &arg_K_fold, &arg_max_iter, &arg_tolerance, &arg_random_seed)) {
    return NULL;
  }

  // Handle X argument
  arg_X = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_X), NPY_DOUBLE, NPY_IN_ARRAY));
  double* ptr_arg_X = reinterpret_cast<double*>(arg_X->data);
  const int nrow_X = (arg_X->dimensions)[0];
  const int ncol_X = (arg_X->dimensions)[1];

  // Handle y argument
  arg_y = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_y), NPY_DOUBLE, NPY_IN_ARRAY));
  double* ptr_arg_y = reinterpret_cast<double*>(arg_y->data);
  const int nrow_y = (arg_y->dimensions)[0];

  // Handle alpha argument
  arg_alpha = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_alpha), NPY_DOUBLE, NPY_IN_ARRAY));
  double* ptr_arg_alpha = reinterpret_cast<double*>(arg_alpha->data);

  // Handle lambdas argument
  arg_lambdas = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_lambdas), NPY_DOUBLE, NPY_IN_ARRAY));
  double* ptr_arg_lambdas = reinterpret_cast<double*>(arg_lambdas->data);
  const int nrow_lambdas = (arg_lambdas->dimensions)[0];

  // Build lambdas
  vector<double> lambdas;
  lambdas.assign(ptr_arg_lambdas, ptr_arg_lambdas + nrow_lambdas);

  // Setup
  const Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(ptr_arg_X, nrow_X, ncol_X);
  const Map<VectorXd> y(ptr_arg_y, nrow_y);
  const Map<Vector6d> alpha(ptr_arg_alpha);

  // CV
  CVType cv = cross_validation_proximal_gradient_cd(X, y, arg_K_fold, alpha, lambdas, arg_step_size, arg_max_iter, arg_tolerance, arg_random_seed);

  // Get location of best lambda
  MatrixXf::Index min_row;
  cv.risks.minCoeff(&min_row);
  double best_lambda = cv.lambdas[min_row];

  // TODO implement

  //
  // Copy to Python
  //
  // Copy risks
  long res_risks_dims[1];
  res_risks_dims[0] = cv.risks.rows();
  PyArrayObject* res_risks = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, res_risks_dims, NPY_DOUBLE));
  double* ptr_res_risks = (reinterpret_cast<double*>(res_risks->data));

  for (int i = 0; i < cv.risks.rows(); i++) {
    ptr_res_risks[i] = cv.risks(i);
  }

  // Copy lambdas
  long res_lambdas_dims[1];
  res_lambdas_dims[0] = cv.lambdas.size();
  PyArrayObject* res_lambdas = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, res_lambdas_dims, NPY_DOUBLE));
  double* ptr_res_lambdas = (reinterpret_cast<double*>(res_lambdas->data));

  for (size_t i = 0; i < cv.lambdas.size(); i++) {
    ptr_res_lambdas[i] = cv.lambdas[i];
  }

  return Py_BuildValue("OOd", res_risks, res_lambdas, best_lambda);
}

static PyMethodDef methods[] = {
    {"fit", reinterpret_cast<PyCFunction>(python_fit), METH_VARARGS|METH_KEYWORDS, ""},
    {"predict", reinterpret_cast<PyCFunction>(python_predict), METH_VARARGS|METH_KEYWORDS, ""},
    {"cross_validation", reinterpret_cast<PyCFunction>(python_cross_validation), METH_VARARGS|METH_KEYWORDS, ""},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "pros",
    NULL,
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_pros(void) {
  Py_Initialize();
  PyObject *m = PyModule_Create(&module_definition);
  import_array(); // Numpy requirement
  return m;
}