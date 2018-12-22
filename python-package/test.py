import pros
import numpy as np

###
# Generate data
###
n = 100
p = 5

A = np.array(np.repeat(3, p - int(1/2 * p))) # nonzero features
B_true = np.append(A, np.zeros(p - len(A))) # add sparsity
intercept_true = 5

X_train = np.zeros((n, p))
for i in range(0, n):
  X_train[i, :] = np.random.multivariate_normal(np.zeros(p), np.identity(p))
y_train = np.repeat(intercept_true, n) + X_train @ B_true

n_test = int(n/3)
X_test = np.zeros((n_test, p))
for i in range(0, n_test):
  X_test[i, :] = np.random.multivariate_normal(np.zeros(p), np.identity(p))
y_test = np.repeat(intercept_true, n_test) + (X_test @ B_true)


alpha = np.array([1, 0, 0, 0, 0, 0])
lambda_ = 1;
lambdas = np.array([.1, .5, 1, 2, 3, 4])
step_size = 1/100;

K_fold = 10;
max_iter = 1000;
tolerance = 10**(-8);
random_seed = 777;

###
# Single fit test
###
B, intercept = pros.fit(X = X_train, y = y_train, alpha = alpha, lambda_ = lambda_, step_size = step_size)
print("mse:", ((y_test - pros.predict(X = X_test, intercept = intercept, B = B))**2).mean())

###
# CV test
###
cv_risks, cv_lambdas, best_lambda = pros.cross_validation(X = X_train, y = y_train, alpha = alpha, lambdas = lambdas, step_size = step_size)

B, intercept = pros.fit(X = X_train, y = y_train, alpha = alpha, lambda_ = best_lambda, step_size = step_size)
print("mse:", ((y_test - pros.predict(X = X_test, intercept = intercept, B = B))**2).mean())

