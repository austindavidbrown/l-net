import numpy as np

# Generate data
n = 1000
p = 5

A = np.array(np.repeat(3, p - int(1/2 * p))) # nonzero features
B_true = np.append(A, np.zeros(p - len(A))) # add sparsity
intercept_true = 5

X_train = np.zeros((n, p))
for i in range(0, n):
  X_train[i, :] = np.random.multivariate_normal(np.zeros(p), np.identity(p))
# err_train = np.random.multivariate_normal(np.zeros(n), np.identity(n)) # error 
y_train = np.repeat(intercept_true, n) + (X_train @ B_true)

n_test = int(n/3)
X_test = np.zeros((n_test, p))
for i in range(0, n_test):
  X_test[i, :] = np.random.multivariate_normal(np.zeros(p), np.identity(p))
# err_test = np.random.multivariate_normal(np.zeros(n_test), np.identity(n_test)) # error
y_test = np.repeat(intercept_true, n_test) + (X_test @ B_true)

# Save to file
np.savetxt("X_train.csv", np.asarray(X_train), delimiter=",")
np.savetxt("y_train.csv", np.asarray(y_train), delimiter=",")

np.savetxt("X_test.csv", np.asarray(X_test), delimiter=",")
np.savetxt("y_test.csv", np.asarray(y_test), delimiter=",")