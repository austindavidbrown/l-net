###
# Test
###
test = function() {
  source("pros.R")
  system("cd ../src; R CMD SHLIB R_interface.cpp")
  dyn.load("../src/R_interface.so")

  X_train = data.matrix(read.csv("../../data/X_train.csv", header = F))
  y_train = as.vector(data.matrix(read.csv("../../data/y_train.csv", header = F)))

  X_test = data.matrix(read.csv("../../data/X_test.csv", header = F))
  y_test = as.vector(data.matrix(read.csv("../../data/y_test.csv", header = F)))

  max_iter = 10000
  alpha = c(1, 0, 0, 0, 0, 0)
  lambda = .01
  step_size = 1/1000
  lambdas = seq(.001, 1.001, .01)
  K_fold = 10
  tolerance = 10^(-8)
  random_seed = 1032432;

  ###
  # fit test
  ###
  .Call("R_fit", as.matrix(X_train), matrix(as.vector(t(y_train)), ncol = 1),
                 matrix(as.vector(t(alpha)), ncol = 1), as.double(lambda), as.double(step_size),
                 as.integer(max_iter), as.double(tolerance), as.integer(random_seed))
  fit = pros(X_train, y_train, lambda = .1, step_size = 1/1000)
  fit

  ###
  # predict test
  ###
  .Call("R_predict", matrix(c(1, 1, 1, 1, 1), ncol = 1), as.double(mean(t(y_test))), as.matrix(X_test))
  pred = predict(fit, X_train)
  
  print(mean((y_train - predict(fit, X_train))^2))
  print(mean((y_test - predict(fit, X_test))^2))

  ###
  # CV test
  ###
  .Call("R_cross_validation", as.matrix(X_train), matrix(as.vector(t(y_train)), ncol = 1), as.double(K_fold), 
        matrix(as.vector(t(alpha)), ncol = 1), lambdas, as.double(step_size),
        as.integer(max_iter), as.double(tolerance), as.integer(random_seed))
  cv = cv.pros(X_train, y_train, lambdas = lambdas, step_size = 1/1000)

  print(mean((y_train - predict(cv, X_train))^2))
  print(mean((y_test - predict(cv, X_test))^2))

}