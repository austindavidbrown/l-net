import numpy as np
import lnet

###
# Generate data
###
n = 100
p = 5

SPLIT_RATIO = .8
SPARSITY_LEVEL = p // 2

B = np.repeat(3, p)
B[np.random.choice(p, SPARSITY_LEVEL, replace = False)] = 0 # sparsify
intercept = 5

X = np.zeros((n, p))
for i in range(0, n):
  X[i, :] = np.random.multivariate_normal(np.zeros(p), np.identity(p))
y = np.repeat(intercept, n) + X @ B

TRAIN = np.arange(0, int(np.floor(SPLIT_RATIO * n)))
TEST = np.arange(int(np.floor(SPLIT_RATIO * n)), n)
X_train, y_train = X[TRAIN, :], y[TRAIN]
X_test, y_test = X[TEST, :], y[TEST]

alpha = np.array([1, 0, 0, 0, 0, 0])

###
# Single fit test
###
lambda_ = 1;
fit = lnet.Fit(X = X_train, y = y_train, alpha = alpha, lambda_ = lambda_)
print("Test MSE:", ((y_test - fit.predict(X = X_test))**2).mean())

###
# CV test
###
lambdas = np.array([.1, .5, 1, 2, 3, 4])
K_fold = 10;
cv = lnet.CV(X = X_train, y = y_train, alpha = alpha, lambdas = lambdas)
print("CV Test MSE:", ((y_test - cv.predict(X = X_test))**2).mean())

###
# Coeff test
###
print(fit.coeff())
