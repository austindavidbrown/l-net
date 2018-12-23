/*
Refer to this documentation: https://docs.python.org/3/extending/newtypes_tutorial.html
*/

#include <math.h>
#include <vector>
#include <iostream>

#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>

#include <Eigen/Dense>
#include "lnet.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // use new numpy api

using namespace Eigen;
using std::cout;
using std::vector;

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

  // return dictionary
  return Py_BuildValue("{s:O, s:O, s:d}",
                "cv_risks", res_risks, 
                "cv_lambdas", res_lambdas,
                "best_lambda", best_lambda);
}









/*
Lnet python class
*/
typedef struct {
  PyObject_HEAD

  VectorXd B;
  double intercept;
} LnetObject;

static PyObject* Lnet_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    LnetObject *self;
    self = (LnetObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
      // initialization goes here
    }
    return (PyObject *) self;
}

static void Lnet_dealloc(LnetObject *self) {
  Py_TYPE(self)->tp_free((PyObject *) self);
}

static int python_Lnet_fit(LnetObject *self, PyObject *args, PyObject *kwargs) {
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
    return -1;
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
  const FitType fit = fit_proximal_gradient_cd(B_0, X, y, alpha, arg_lambda, arg_step_size, arg_max_iter, arg_tolerance, arg_random_seed);

  // Set to the class
  self->B = fit.B;
  self->intercept = fit.intercept;

  return 0;
}

static PyObject* python_Lnet_coeff(LnetObject *self, PyObject *Py_UNUSED(ignored)) { 
  //
  // Copy to Python
  //
  long B_res_dims[1];
  B_res_dims[0] = self->B.rows();
  PyArrayObject* B_res = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, B_res_dims, NPY_DOUBLE)); // 1 is for vector
  double* ptr_B_res = (reinterpret_cast<double*>(B_res->data));

  for (int i = 0; i < self->B.rows(); i++) {
    ptr_B_res[i] = self->B(i);
  }

  // return dictionary
  return Py_BuildValue("{s:d, s:O}",
                "intercept", self->intercept, 
                "B", B_res);
}

static PyObject* python_Lnet_predict(LnetObject *self, PyObject *args, PyObject* kwargs) {
  char* keywords[] = {"X", NULL};

  // Required arguments
  PyArrayObject* arg_X = NULL;

  // Parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", keywords, 
                                   &PyArray_Type, &arg_X)) {
    return NULL;
  }

  // Handle X argument
  arg_X = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_X), NPY_DOUBLE, NPY_IN_ARRAY));
  double* ptr_arg_X = reinterpret_cast<double*>(arg_X->data);
  const int nrow_X = (arg_X->dimensions)[0];
  const int ncol_X = (arg_X->dimensions)[1];

  // Setup
  const Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(ptr_arg_X, nrow_X, ncol_X);

  // Predict
  const VectorXd pred = predict(X, self->intercept, self->B);

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

/*
Lnet python class definition
*/
static PyMemberDef Lnet_members[] = {
  {NULL}  /* Sentinel */
};

static PyMethodDef Lnet_methods[] = {
  {"coeff", reinterpret_cast<PyCFunction>(python_Lnet_coeff), METH_NOARGS, "doc string"},
  {"predict", reinterpret_cast<PyCFunction>(python_Lnet_predict), METH_VARARGS|METH_KEYWORDS, "doc string"},
  {NULL}  /* Sentinel */
};

static PyTypeObject LnetType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Lnet",
    .tp_doc = "doc string",
    .tp_basicsize = sizeof(LnetObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Lnet_new,
    .tp_init = (initproc) python_Lnet_fit,
    .tp_dealloc = (destructor) Lnet_dealloc,
    .tp_members = Lnet_members,
    .tp_methods = Lnet_methods,
};


/*
lnet python module definition
*/
static PyMethodDef lnet_methods[] = {
    {"cross_validation", reinterpret_cast<PyCFunction>(python_cross_validation), METH_VARARGS|METH_KEYWORDS, "doc string"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef lnet_module = {
    PyModuleDef_HEAD_INIT,
    .m_name =  "lnet",
    .m_doc = "doc string",
    .m_size = -1,
    lnet_methods,
};

PyMODINIT_FUNC PyInit_lnet(void) {
  PyObject *m;
  import_array(); // Numpy requirement

  if (PyType_Ready(&LnetType) < 0) {
    return NULL;
  }

  m = PyModule_Create(&lnet_module);
  if (m == NULL) {
    return NULL;
  }

  Py_INCREF(&LnetType);
  PyModule_AddObject(m, "Lnet", reinterpret_cast<PyObject*>(&LnetType));
  return m;
}