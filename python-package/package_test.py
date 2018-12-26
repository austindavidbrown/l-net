import numpy as np
import lnet
from sklearn.metrics import accuracy_score, zero_one_loss, mean_squared_error

###
# Generate Sparse Regression data
###
n = 1000
p = 10

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


###
# Generate logistic regression data
###
p = 1/(1 + np.exp(-1 * y))

binary_y = np.zeros(n)
for i in range(0, n):
  binary_y[i] = np.random.binomial(1, p[i])

binary_y_train = binary_y[TRAIN]
binary_y_test = binary_y[TEST]


###
# Test Regression
###
print("""
      Testing Regression
      """)

alpha = np.array([1, 0, 0, 0, 0, 0])

###
# Single fit test
###
lambda_ = 1;
fit = lnet.Fit(X = X_train, y = y_train, alpha = alpha, lambda_ = lambda_)
print("\nCoefficient: ", fit.coeff())

print("\nFit MSE:", ((y_test - fit.predict(X = X_test))**2).mean())

###
# CV test
###
cv = lnet.CV(X = X_train, y = y_train, alpha = alpha, lambdas = np.array([.1, .5, 1, 2, 3, 4]))
print("\nCV MSE, specified lambdas:", ((y_test - cv.predict(X = X_test))**2).mean())

cv = lnet.CV(X = X_train, y = y_train, alpha = alpha)
print("\nCV MSE, default lambdas:", ((y_test - cv.predict(X = X_test))**2).mean())


###
# Test Logistic Regression
###
print("""
      Testing Classification
      """)

alpha = np.array([1, 0, 0, 0, 0, 0])

###
# Single fit test
###
lambda_ = 1;
fit = lnet.Fit(X = X_train, y = binary_y_train, alpha = alpha, lambda_ = lambda_, objective = "classification:binary")
print("\nCoefficient: ", fit.coeff())

print("\nFit Accuracy: ", 1/binary_y_test.shape[0] * np.sum(binary_y_test == np.round(fit.predict(X = X_test))).astype(int))

cv = lnet.CV(X = X_train, y = y_train, alpha = alpha, lambdas = np.array([.1, .5, 1, 2, 3, 4]), objective = "classification:binary")
print("\nCV Accuracy: ", 1/binary_y_test.shape[0] * np.sum(binary_y_test == np.round(cv.predict(X = X_test))).astype(int))

