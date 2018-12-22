#' Pros
#' 
#' The \code{pros} function is used to fit a single regression model with a specified penalization. 
#' 
#' @param X is an \eqn{n \times m}-dimensional matrix of the data.
#' @param y is an \eqn{n \times m}-dimensional matrix of the data.
#' @param alpha is a \eqn{6}-dimensional vector of the convex combination corresponding to the penalization:
#' \itemize{
#'   \item \eqn{\alpha_1} is the \eqn{l^1} penalty.
#'   \item \eqn{\alpha_2} is the \eqn{l^2} penalty.
#'   \item \eqn{\alpha_3} is the \eqn{l^4} penalty.
#'   \item \eqn{\alpha_4} is the \eqn{l^6} penalty.
#'   \item \eqn{\alpha_5} is the \eqn{l^8} penalty.
#'   \item \eqn{\alpha_6} is the \eqn{l^{10}} penalty.
#' }
#' @param lambda is the Lagrangian dual penalization parameter.
#' @param step_size is a tuning parameter defining the step size. Larger values are more aggressive and smaller values are less aggressive.
#' @param max_iter is the maximum iterations the algorithm will run regardless of convergence.
#' @param tolerance is the accuracy of the stopping criterion.
#' @param random_seed is the random seed used in the algorithms.
#' 
#' @return 
#' A class \code{pros}
#'
#' @references
#' Zou, Hui. “Regularization and variable selection via the elastic net.” (2004).
#'
#' @examples
#' n = 1000
#' X1 <- rnorm(n)
#' X2 <- rnorm(n)
#' y = 3 + 5 * X1 + 0 * X2
#' X = matrix(c(X1, X2), ncol = 2)
#' fit = pros(X, y, lambda = 2, step_size = .001)
#' predict(fit, X)
#'
#' @export
pros = function(X, y, 
                alpha = c(1, 0, 0, 0, 0, 0), lambda, step_size,
                max_iter = 10000, tolerance = 10^(-8), random_seed = 0) {
  if (!is.matrix(X)) {
    stop("X must be a matrix")
  }

  if (!is.vector(y)) {
    stop("y must be a vector")
  }
  y = matrix(as.vector(t(y)), ncol = 1) # convert to column vector

  if (length(alpha) != 6) {
    stop("alpha needs to be length 6")
  }
  alpha = matrix(as.vector(t(alpha)), ncol = 1) # convert to column vector

  if (!(is.atomic(lambda) && length(lambda) == 1L)) {
    stop("lambda must be a scalar")
  }
  if (!(is.atomic(step_size) && length(step_size) == 1L)) {
    stop("step_size must be a scalar")
  }
  if (!(is.atomic(max_iter) && length(max_iter) == 1L)) {
    stop("max_iter must be a integer")
  }
  if (!(is.atomic(tolerance) && length(tolerance) == 1L)) {
    stop("tolerance must be a scalar")
  }
  if (!(is.atomic(random_seed) && length(random_seed) == 1L)) {
    stop("random_seed must be a integer")
  }

  res = .Call("R_fit", as.matrix(X), y, 
            alpha, as.double(lambda), as.double(step_size),
            as.integer(max_iter), as.double(tolerance), as.integer(random_seed))
  class(res) = "pros"
  return ( res )
}


#' Pros Prediction
#' 
#' The prediction function for \code{pros}.
#' 
#' @param object an object of class \code{pros}
#' @param X is an \eqn{n \times m}-dimensional matrix of the data.
#' @param ... Other parameters (this is required by R)
#' 
#' @return 
#' A \code{vector} of prediction values.
#'
#' @export
predict.pros = function(object, X, ...) {
  if (!is.matrix(X)) {
    stop("X must be a matrix")
  }

  B = matrix(as.vector(t(object$B)), ncol = 1) # convert to column vector
  intercept = object$intercept
  return ( .Call("R_predict", B, as.double(intercept), as.matrix(X)) )
}


#' Cross-validation
#' 
#' The \code{cv.pros} function is used for K-fold cross-validation.
#' 
#' @param X is an \eqn{n \times m}-dimensional matrix of the data.
#' @param y is an \eqn{n \times m}-dimensional matrix of the data.
#' @param alpha is a \eqn{6}-dimensional vector of the convex combination corresponding to the penalization:
#' \itemize{
#'   \item \eqn{\alpha_1} is the \eqn{l^1} penalty.
#'   \item \eqn{\alpha_2} is the \eqn{l^2} penalty.
#'   \item \eqn{\alpha_3} is the \eqn{l^4} penalty.
#'   \item \eqn{\alpha_4} is the \eqn{l^6} penalty.
#'   \item \eqn{\alpha_5} is the \eqn{l^8} penalty.
#'   \item \eqn{\alpha_6} is the \eqn{l^{10}} penalty.
#' }
#' @param K_fold is the number of folds in cross-validation.
#' @param lambdas is a vector of dual penalization values to be evaluated.
#' @param step_size is a tuning parameter defining the step size. Larger values are more aggressive and smaller values are less aggressive.
#' @param max_iter is the maximum iterations the algorithm will run regardless of convergence.
#' @param tolerance is the accuracy of the stopping criterion.
#' @param random_seed is the random seed used in the algorithms.
#'
#'
#' @return 
#' A class \code{cv_pros}
#'
#' @export
cv.pros = function(X, y, 
                   K_fold = 10, alpha = c(1, 0, 0, 0, 0, 0), lambdas = c(), step_size,
                   max_iter = 10000, tolerance = 10^(-8), random_seed = 0) {
  if (!is.matrix(X)) {
    stop("X must be a matrix")
  }
  if (!is.vector(y)) {
    stop("y must be a vector")
  }
  # Set default lambdas for the user
  # There is no theoretical justification here.
  if (length(lambdas) == 0) {
    lambdas = seq(3/2 * ncol(X), 3 * ncol(X), .1)
  } else if (!is.vector(lambdas)) {
    stop("lambdas must be a vector")
  }
  if (!(is.atomic(step_size) && length(step_size) == 1L)) {
    stop("step_size must be a scalar")
  }
  if (!(is.atomic(max_iter) && length(max_iter) == 1L)) {
    stop("max_iter must be a integer")
  }
  if (!(is.atomic(tolerance) && length(tolerance) == 1L)) {
    stop("tolerance must be a scalar")
  }
  if (!(is.atomic(random_seed) && length(random_seed) == 1L)) {
    stop("random_seed must be a integer")
  }

  y = matrix(as.vector(t(y)), ncol = 1) # convert to column vector
  alpha = matrix(as.vector(t(alpha)), ncol = 1) # convert to column vector

  res = .Call("R_cross_validation", as.matrix(X), y, 
              as.double(K_fold), alpha, as.vector(lambdas), as.double(step_size),
              as.integer(max_iter), as.double(tolerance), as.integer(random_seed))
  res$X = X
  res$y = y
  res$alpha = alpha
  res$step_size = step_size
  res$max_iter = max_iter
  res$tolerance = tolerance
  res$random_seed = random_seed
  class(res) = "cv_pros"
  return ( res )
}

#' Cross-validation Prediction
#' 
#' The prediction function for \code{cv.pros}.
#' 
#' @param object an object of class \code{cv_pros}
#' @param X_new is an \eqn{n \times m}-dimensional matrix of the data.
#' @param ... Other parameters (this is required by R)
#' 
#' @return 
#' A \code{vector} of prediction values.
#'
#' @references
#' Zou, Hui. “Regularization and variable selection via the elastic net.” (2004).
#'
#' @examples
#' n = 1000
#' X1 <- rnorm(n)
#' X2 <- rnorm(n)
#' y = 3 + 5 * X1 + 0 * X2
#' X = matrix(c(X1, X2), ncol = 2)
#' cv = cv.pros(X, y, step_size = .001)
#' predict(cv, X)
#'
#' @export
predict.cv_pros = function(object, X_new, ...) {
  if (!is.matrix(X_new)) {
    stop("X_new must be a matrix")
  }

  X = object$X
  y = object$y
  alpha = object$alpha
  lambda = object$best_lambda
  step_size = object$step_size
  max_iter = object$max_iter
  tolerance = object$tolerance
  random_seed = object$random_seed
  fit = pros(X, as.vector(y), alpha = alpha, lambda = lambda, step_size = step_size, max_iter = max_iter, tolerance = tolerance, random_seed = random_seed)
  res = predict(fit, X_new)
  return ( res )
}
