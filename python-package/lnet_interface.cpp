/*
Refer to this documentation: https://docs.python.org/3/extending/newtypes_tutorial.html
Reference:
  PySys_WriteStdout(std::to_string(self->max_iter).c_str());
*/

// TODO document
// TODO change ptr variables to data_ptr
// Change self->X and the X in CV it is confusing

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

/*
LnetCV python class
*/
typedef struct {
  PyObject_HEAD

  double best_lambda;
  VectorXd cv_risks;
  vector<double> cv_lambdas;

  MatrixXd X;
  VectorXd y;
  Vector6d alpha;
  double step_size;

  int K_fold;
  int max_iter;
  double tolerance;
  int random_seed;
} LnetCVObject;

static PyObject* LnetCV_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    LnetCVObject *self;
    self = (LnetCVObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
      // Set default values
      self->K_fold = 10;
      self->max_iter = 10000;
      self->tolerance = pow(10, -8);
      self->random_seed = 0;
    }
    return (PyObject *) self;
}

static void LnetCV_dealloc(LnetCVObject *self) {
  Py_TYPE(self)->tp_free((PyObject *) self);
}

static int python_LnetCV_cross_validation(LnetCVObject *self, PyObject *args, PyObject* kwargs) {
  char* keywords[] = {"X", "y", "alpha", "lambdas", "step_size", 
                      "K_fold", "max_iter", "tolerance", "random_seed", NULL};

  // Required arguments
  PyArrayObject* arg_y = NULL;
  PyArrayObject* arg_X = NULL;
  PyArrayObject* arg_alpha = NULL;
  PyArrayObject* arg_lambdas = NULL;

  // Parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!O!d|iidi", keywords,
                        &PyArray_Type, &arg_X, &PyArray_Type, &arg_y,
                        &PyArray_Type, &arg_alpha, &PyArray_Type, &arg_lambdas, &(self->step_size), 
                        &(self->K_fold), &(self->max_iter), &(self->tolerance), &(self->random_seed))) {
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

  // Handle lambdas argument
  arg_lambdas = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_lambdas), NPY_DOUBLE, NPY_IN_ARRAY));
  double* ptr_arg_lambdas = reinterpret_cast<double*>(arg_lambdas->data);
  const int nrow_lambdas = (arg_lambdas->dimensions)[0];

  // Build unordered lambdas
  vector<double> lambdas;
  lambdas.assign(ptr_arg_lambdas, ptr_arg_lambdas + nrow_lambdas);

  // Setup
  const Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(ptr_arg_X, nrow_X, ncol_X);
  const Map<VectorXd> y(ptr_arg_y, nrow_y);
  const Map<Vector6d> alpha(ptr_arg_alpha);

  // Assign to class
  self->X = X;
  self->y = y;
  self->alpha = alpha;

  // CV
  CVType cv = cross_validation_proximal_gradient_cd(self->X, self->y, self->K_fold, self->alpha, lambdas, self->step_size, self->max_iter, self->tolerance, self->random_seed);

  // Get location of best lambda
  MatrixXf::Index min_row;
  cv.risks.minCoeff(&min_row);

  // Assign to class
  self->best_lambda = cv.lambdas[min_row];
  self->cv_risks = cv.risks;
  self->cv_lambdas = cv.lambdas;
  return 0;
}

static PyObject* python_LnetCV_data(LnetCVObject *self, PyObject *Py_UNUSED(ignored)) { 
  //
  // Copy to Python
  //
  // Copy cv risks
  long res_risks_dims[1];
  res_risks_dims[0] = self->cv_risks.rows();
  PyArrayObject* res_risks = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, res_risks_dims, NPY_DOUBLE));
  double* ptr_res_risks = (reinterpret_cast<double*>(res_risks->data));

  for (int i = 0; i < self->cv_risks.rows(); i++) {
    ptr_res_risks[i] = self->cv_risks(i);
  }

  // Copy cv lambdas
  long res_lambdas_dims[1];
  res_lambdas_dims[0] = self->cv_lambdas.size();
  PyArrayObject* res_lambdas = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, res_lambdas_dims, NPY_DOUBLE));
  double* ptr_res_lambdas = (reinterpret_cast<double*>(res_lambdas->data));

  for (size_t i = 0; i < self->cv_lambdas.size(); i++) {
    ptr_res_lambdas[i] = self->cv_lambdas[i];
  }

  // return dictionary
  return Py_BuildValue("{s:O, s:O, s:d}",
                "risks", res_risks, 
                "lambdas", res_lambdas,
                "best_lambda", self->best_lambda);
}

static PyObject* python_LnetCV_predict(LnetCVObject *self, PyObject *args, PyObject* kwargs) {
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

  // Fit
  const VectorXd B_0 = VectorXd::Zero(self->X.cols());
  const FitType fit = fit_proximal_gradient_cd(B_0, self->X, self->y, self->alpha, self->best_lambda, self->step_size, self->max_iter, self->tolerance, self->random_seed);

  // Predict
  const VectorXd pred = predict(X, fit.intercept, fit.B);

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
LnetCV python class definition
*/
static PyMemberDef LnetCV_members[] = {
  {NULL}  /* Sentinel */
};

static PyMethodDef LnetCV_methods[] = {
  {"data", reinterpret_cast<PyCFunction>(python_LnetCV_data), METH_NOARGS, "doc string"},
  {"predict", reinterpret_cast<PyCFunction>(python_LnetCV_predict), METH_VARARGS|METH_KEYWORDS, "doc string"},
  {NULL}  /* Sentinel */
};

static PyTypeObject LnetCVType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "LnetCV",
    .tp_doc = "doc string",
    .tp_basicsize = sizeof(LnetCVObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = LnetCV_new,
    .tp_init = (initproc) python_LnetCV_cross_validation,
    .tp_dealloc = (destructor) LnetCV_dealloc,
    .tp_members = LnetCV_members,
    .tp_methods = LnetCV_methods,
};



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

  if (PyType_Ready(&LnetCVType) < 0) {
    return NULL;
  }

  m = PyModule_Create(&lnet_module);
  if (m == NULL) {
    return NULL;
  }

  Py_INCREF(&LnetType);
  PyModule_AddObject(m, "Lnet", reinterpret_cast<PyObject*>(&LnetType));

  Py_INCREF(&LnetCVType);
  PyModule_AddObject(m, "LnetCV", reinterpret_cast<PyObject*>(&LnetCVType));
  return m;
}