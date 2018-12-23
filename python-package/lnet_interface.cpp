/*
Refer to this documentation: https://docs.python.org/3/extending/newtypes_tutorial.html
Reference:
  PySys_WriteStdout(std::to_string(self->max_iter).c_str());
*/

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
using std::vector;
using std::cout;


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
      self->tolerance = pow(10, -6);
      self->random_seed = 0;
    }
    return (PyObject *) self;
}

static void LnetCV_dealloc(LnetCVObject *self) {
  Py_TYPE(self)->tp_free((PyObject *) self);
}

static int python_LnetCV_cross_validation(LnetCVObject *self, PyObject *args, PyObject* kwargs) {
  const char* keywords[] = {"X", "y", "alpha", "lambdas", 
                            "K_fold", "max_iter", "tolerance", "random_seed", NULL};

  // Required arguments
  PyArrayObject* arg_y = NULL;
  PyArrayObject* arg_X = NULL;
  PyArrayObject* arg_alpha = NULL;
  PyArrayObject* arg_lambdas = NULL;

  // Parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!|O!iidi", (char**) keywords,
                        &PyArray_Type, &arg_X, &PyArray_Type, &arg_y, &PyArray_Type, &arg_alpha, 
                        &PyArray_Type, &arg_lambdas, &(self->K_fold), &(self->max_iter), &(self->tolerance), &(self->random_seed))) {
    return -1;
  }

  // Handle X argument
  arg_X = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_X), NPY_DOUBLE, NPY_IN_ARRAY));
  double* data_ptr_arg_X = reinterpret_cast<double*>(arg_X->data);
  const int nrow_X = (arg_X->dimensions)[0];
  const int ncol_X = (arg_X->dimensions)[1];

  // Handle y argument
  arg_y = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_y), NPY_DOUBLE, NPY_IN_ARRAY));
  double* data_ptr_arg_y = reinterpret_cast<double*>(arg_y->data);
  const int nrow_y = (arg_y->dimensions)[0];

  // Handle alpha argument
  arg_alpha = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_alpha), NPY_DOUBLE, NPY_IN_ARRAY));
  double* data_ptr_arg_alpha = reinterpret_cast<double*>(arg_alpha->data);

  /*
  Handle lambdas argument
  If not specified create a default sequence, otherwise convert the user specified lambdas
  */
  vector<double> lambdas;
  if (arg_lambdas == NULL) {
    lambdas.push_back(ncol_X);
    for (int i = 1; i < 100; i++) {
      lambdas.push_back(lambdas[i - 1] + .1);
    }
  } else {
    arg_lambdas = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_lambdas), NPY_DOUBLE, NPY_IN_ARRAY));
    double* data_ptr_arg_lambdas = reinterpret_cast<double*>(arg_lambdas->data);
    const int nrow_lambdas = (arg_lambdas->dimensions)[0];
    lambdas.assign(data_ptr_arg_lambdas, data_ptr_arg_lambdas + nrow_lambdas);
  }

  // Setup
  const Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(data_ptr_arg_X, nrow_X, ncol_X);
  const Map<VectorXd> y(data_ptr_arg_y, nrow_y);
  const Map<Vector6d> alpha(data_ptr_arg_alpha);

  // Assign to class
  self->X = X;
  self->y = y;
  self->alpha = alpha;

  // CV
  CVType cv = cross_validation_proximal_gradient(self->X, self->y, self->K_fold, self->alpha, lambdas, self->max_iter, self->tolerance, self->random_seed);

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
  double* data_ptr_res_risks = (reinterpret_cast<double*>(res_risks->data));

  for (int i = 0; i < self->cv_risks.rows(); i++) {
    data_ptr_res_risks[i] = self->cv_risks(i);
  }

  // Copy cv lambdas
  long res_lambdas_dims[1];
  res_lambdas_dims[0] = self->cv_lambdas.size();
  PyArrayObject* res_lambdas = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, res_lambdas_dims, NPY_DOUBLE));
  double* data_ptr_res_lambdas = (reinterpret_cast<double*>(res_lambdas->data));

  for (size_t i = 0; i < self->cv_lambdas.size(); i++) {
    data_ptr_res_lambdas[i] = self->cv_lambdas[i];
  }

  // return dictionary
  return Py_BuildValue("{s:O, s:O, s:d}",
                "risks", res_risks, 
                "lambdas", res_lambdas,
                "best_lambda", self->best_lambda);
}

static PyObject* python_LnetCV_predict(LnetCVObject *self, PyObject *args, PyObject* kwargs) {
  const char* keywords[] = {"X", NULL};

  // Required arguments
  PyArrayObject* arg_X = NULL;

  // Parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", (char**) keywords, 
                                   &PyArray_Type, &arg_X)) {
    return NULL;
  }

  // Handle X argument
  arg_X = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_X), NPY_DOUBLE, NPY_IN_ARRAY));
  double* data_ptr_arg_X = reinterpret_cast<double*>(arg_X->data);
  const int nrow_X = (arg_X->dimensions)[0];
  const int ncol_X = (arg_X->dimensions)[1];

  // Setup
  const Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(data_ptr_arg_X, nrow_X, ncol_X);

  // Fit
  const VectorXd B_0 = VectorXd::Zero(self->X.cols());
  const FitType fit = fit_proximal_gradient(B_0, self->X, self->y, self->alpha, self->best_lambda, self->max_iter, self->tolerance);

  // Predict
  const VectorXd pred = predict(X, fit.intercept, fit.B);

  //
  // Copy to Pythonc
  //
  long res_dims[1];
  res_dims[0] = pred.rows();
  PyArrayObject* res = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, res_dims, NPY_DOUBLE)); // 1 is for vector
  double* data_ptr_res_data = (reinterpret_cast<double*>(res->data));

  for (int i = 0; i < pred.rows(); i++) {
    data_ptr_res_data[i] = pred(i);
  }

  return Py_BuildValue("O", res);
}

/*
LnetCV python class definition
*/

// documentation
static const char* DOC_LnetCV_fit = R"(
Cross-validation

The \code{LnetCV} function is used for K-fold cross-validation.
 
@param X is an \eqn{n \times m}-dimensional matrix of the data.
@param y is an \eqn{n \times m}-dimensional matrix of the data.
@param alpha is a \eqn{6}-dimensional vector of the convex combination corresponding to the penalization:
\itemize{
   \item \eqn{\alpha_1} is the \eqn{l^1} penalty.
   \item \eqn{\alpha_2} is the \eqn{l^2} penalty.
   \item \eqn{\alpha_3} is the \eqn{l^4} penalty.
   \item \eqn{\alpha_4} is the \eqn{l^6} penalty.
   \item \eqn{\alpha_5} is the \eqn{l^8} penalty.
   \item \eqn{\alpha_6} is the \eqn{l^{10}} penalty.
}
@param K_fold is the number of folds in cross-validation.
@param lambdas is a vector of dual penalization values to be evaluated.
@param max_iter is the maximum iterations the algorithm will run regardless of convergence.
@param tolerance is the accuracy of the stopping criterion.
@param random_seed is the random seed used in the algorithms.

@return A class \code{LnetCV}
)";

static const char* DOC_LnetCV_predict = R"(
Cross-validation Prediction
 
The prediction function for \code{cv.pros}.
 
@param object an object of class \code{cv_pros}
@param X_new is an \eqn{n \times m}-dimensional matrix of the data.
@param ... Other parameters (this is required by R)
 
@return 
A \code{vector} of prediction values.

@references
Zou, Hui. “Regularization and variable selection via the elastic net.” (2004).
)";

static const char* DOC_LnetCV_data = R"(
Returns data from the CV class.
)";

static PyMemberDef LnetCV_members[] = {
  {NULL}  /* Sentinel */
};

static PyMethodDef LnetCV_methods[] = {
  {"data", reinterpret_cast<PyCFunction>(python_LnetCV_data), METH_NOARGS, DOC_LnetCV_data},
  {"predict", reinterpret_cast<PyCFunction>(python_LnetCV_predict), METH_VARARGS|METH_KEYWORDS, DOC_LnetCV_predict},
  {NULL}  /* Sentinel */
};

static PyTypeObject LnetCVType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "lnet.CV",
    .tp_doc = DOC_LnetCV_fit,
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

Lnet python Fit class

*/
typedef struct {
  PyObject_HEAD

  VectorXd B;
  double intercept;
} LnetFitObject;

static PyObject* LnetFit_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    LnetFitObject *self;
    self = (LnetFitObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
      // initialization goes here
    }
    return (PyObject *) self;
}

static void LnetFit_dealloc(LnetFitObject *self) {
  Py_TYPE(self)->tp_free((PyObject *) self);
}

static int LnetFit_fit(LnetFitObject *self, PyObject *args, PyObject *kwargs) {
  const char* keywords[] = {"X", "y", "alpha", "lambda_", 
                      "max_iter", "tolerance", "random_seed", NULL};

  // Required arguments
  PyArrayObject* arg_y = NULL;
  PyArrayObject* arg_X = NULL;
  PyArrayObject* arg_alpha = NULL;
  double arg_lambda;

  // Optional arguments
  int arg_max_iter = 10000;
  double arg_tolerance = pow(10, -6);
  int arg_random_seed = 0;

  // Parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!d|idi", (char**) keywords,
                        &PyArray_Type, &arg_X, &PyArray_Type, &arg_y, 
                        &PyArray_Type, &arg_alpha, &arg_lambda, 
                        &arg_max_iter, &arg_tolerance, &arg_random_seed)) {
    return -1;
  }

  // Handle X argument
  arg_X = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_X), NPY_DOUBLE, NPY_IN_ARRAY));
  double* data_ptr_arg_X = reinterpret_cast<double*>(arg_X->data);
  const int nrow_X = (arg_X->dimensions)[0];
  const int ncol_X = (arg_X->dimensions)[1];

  // Handle y argument
  arg_y = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_y), NPY_DOUBLE, NPY_IN_ARRAY));
  double* data_ptr_arg_y = reinterpret_cast<double*>(arg_y->data);
  const int nrow_y = (arg_y->dimensions)[0];

  // Handle alpha argument
  arg_alpha = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_alpha), NPY_DOUBLE, NPY_IN_ARRAY));
  double* data_ptr_arg_alpha = reinterpret_cast<double*>(arg_alpha->data);

  // Setup
  const Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(data_ptr_arg_X, nrow_X, ncol_X);
  const Map<VectorXd> y(data_ptr_arg_y, nrow_y);
  const Map<Vector6d> alpha(data_ptr_arg_alpha);

  // Fit
  const VectorXd B_0 = VectorXd::Zero(X.cols());
  const FitType fit = fit_proximal_gradient(B_0, X, y, alpha, arg_lambda, arg_max_iter, arg_tolerance);

  // Assign to the class
  self->B = fit.B;
  self->intercept = fit.intercept;

  return 0;
}

static PyObject* LnetFit_coeff(LnetFitObject *self, PyObject *Py_UNUSED(ignored)) { 
  //
  // Copy to Python
  //
  long B_res_dims[1];
  B_res_dims[0] = self->B.rows();
  PyArrayObject* B_res = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, B_res_dims, NPY_DOUBLE)); // 1 is for vector
  double* data_ptr_B_res = (reinterpret_cast<double*>(B_res->data));

  for (int i = 0; i < self->B.rows(); i++) {
    data_ptr_B_res[i] = self->B(i);
  }

  // return dictionary
  return Py_BuildValue("{s:d, s:O}",
                "intercept", self->intercept, 
                "B", B_res);
}

static PyObject* LnetFit_predict(LnetFitObject *self, PyObject *args, PyObject* kwargs) {
  const char* keywords[] = {"X", NULL};

  // Required arguments
  PyArrayObject* arg_X = NULL;

  // Parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", (char**) keywords, 
                                   &PyArray_Type, &arg_X)) {
    return NULL;
  }

  // Handle X argument
  arg_X = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(reinterpret_cast<PyObject*>(arg_X), NPY_DOUBLE, NPY_IN_ARRAY));
  double* data_ptr_arg_X = reinterpret_cast<double*>(arg_X->data);
  const int nrow_X = (arg_X->dimensions)[0];
  const int ncol_X = (arg_X->dimensions)[1];

  // Setup
  const Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(data_ptr_arg_X, nrow_X, ncol_X);

  // Predict
  const VectorXd pred = predict(X, self->intercept, self->B);

  //
  // Copy to Python
  //
  long res_dims[1];
  res_dims[0] = pred.rows();
  PyArrayObject* res = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, res_dims, NPY_DOUBLE)); // 1 is for vector
  double* data_ptr_res_data = (reinterpret_cast<double*>(res->data));

  for (int i = 0; i < pred.rows(); i++) {
    data_ptr_res_data[i] = pred(i);
  }

  return Py_BuildValue("O", res);
}

/*

Lnet python class definition

*/

// Documentation
static const char* DOC_LnetFit_init = R"(
Lnet
 
The \code{Lnet} class is used to fit a single regression model with a specified penalization. 

@param X is an \eqn{n \times m}-dimensional matrix of the data.
@param y is an \eqn{n \times m}-dimensional matrix of the data.
@param alpha is a \eqn{6}-dimensional vector of the convex combination corresponding to the penalization:
\itemize{
\item \eqn{\alpha_1} is the \eqn{l^1} penalty.
  \item \eqn{\alpha_2} is the \eqn{l^2} penalty.
  \item \eqn{\alpha_3} is the \eqn{l^4} penalty.
  \item \eqn{\alpha_4} is the \eqn{l^6} penalty.
  \item \eqn{\alpha_5} is the \eqn{l^8} penalty.
  \item \eqn{\alpha_6} is the \eqn{l^{10}} penalty.
}
@param lambda is the Lagrangian dual penalization parameter.
@param max_iter is the maximum iterations the algorithm will run regardless of convergence.
@param tolerance is the accuracy of the stopping criterion.
@param random_seed is the random seed used in the algorithms.
 
@return 
A class \code{Lnet}

@references
Zou, Hui. “Regularization and variable selection via the elastic net.” (2004).
)";

static const char* DOC_LnetFit_predict = R"(
Lnet Prediction

The prediction function for \code{pros}.
 
@param object an object of class \code{pros}
@param X is an \eqn{n \times m}-dimensional matrix of the data.
@param ... Other parameters (this is required by R)
 
@return A \code{vector} of prediction values.
)";

static const char* DOC_LnetFit_coeff = R"(
Lnet coeff

Returns the coefficients.
)";

static PyMemberDef LnetFit_members[] = {
  {NULL}  /* Sentinel */
};

static PyMethodDef LnetFit_methods[] = {
  {"coeff", reinterpret_cast<PyCFunction>(LnetFit_coeff), METH_NOARGS, DOC_LnetFit_coeff},
  {"predict", reinterpret_cast<PyCFunction>(LnetFit_predict), METH_VARARGS|METH_KEYWORDS, DOC_LnetFit_predict},
  {NULL}  /* Sentinel */
};

static PyTypeObject LnetFitType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "lnet.Fit",
    .tp_doc = DOC_LnetFit_init,
    .tp_basicsize = sizeof(LnetFitObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = LnetFit_new,
    .tp_init = (initproc) LnetFit_fit,
    .tp_dealloc = (destructor) LnetFit_dealloc,
    .tp_members = LnetFit_members,
    .tp_methods = LnetFit_methods,
};


/*

lnet python module

*/

// Documentation
static const char* DOC_lnet_module = R"(
Adds the ability to combine l1 to l10 penalties in regression extending the elastic-net.
)";

static PyMethodDef lnet_methods[] = {
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef lnet_module = {
    PyModuleDef_HEAD_INIT,
    .m_name =  "lnet",
    .m_doc = DOC_lnet_module,
    .m_size = -1,
    lnet_methods,
};

PyMODINIT_FUNC PyInit_lnet(void) {
  PyObject *m;
  import_array(); // Numpy requirement

  if (PyType_Ready(&LnetFitType) < 0) {
    return NULL;
  }

  if (PyType_Ready(&LnetCVType) < 0) {
    return NULL;
  }

  m = PyModule_Create(&lnet_module);
  if (m == NULL) {
    return NULL;
  }

  Py_INCREF(&LnetFitType);
  PyModule_AddObject(m, "Fit", reinterpret_cast<PyObject*>(&LnetFitType));

  Py_INCREF(&LnetCVType);
  PyModule_AddObject(m, "CV", reinterpret_cast<PyObject*>(&LnetCVType));
  return m;
}