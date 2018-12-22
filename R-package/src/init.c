#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>


/* .Call calls */
extern SEXP R_cross_validation(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_fit(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_predict(SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"R_cross_validation", (DL_FUNC) &R_cross_validation, 10},
    {"R_fit",              (DL_FUNC) &R_fit,               9},
    {"R_predict",          (DL_FUNC) &R_predict,           3},
    {NULL, NULL, 0}
};

void R_init_pros(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
