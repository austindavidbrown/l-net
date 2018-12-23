import numpy as np

# Generate data
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

# Save to file
np.savetxt("data/X_train.csv", np.asarray(X_train), delimiter=",")
np.savetxt("data/y_train.csv", np.asarray(y_train), delimiter=",")

np.savetxt("data/X_test.csv", np.asarray(X_test), delimiter=",")
np.savetxt("data/y_test.csv", np.asarray(y_test), delimiter=",")