import numpy as np
from sklearn.metrics import *

# Returns matrix of Betas
# Each row is a Beta for the specified lambda
def warm_start_B_matrix(X_, y_, alpha, lambdas__):
  n = X_.shape[0]
  p = X_.shape[1]
  L = len(lambdas__)
  lambdas_ = -np.sort(-lambdas__) # Order lambdas from largest to smallest (sparest to densest)
  B_matrix = np.zeros((L, p))

  # do the first one normally
  B_0 = np.zeros(p)
  B_matrix[0, :] = penalize_sgcd(B_0, X_, y_, alpha, lambdas_[0])

  # Warm start after the first one
  for l in range(1, L):
    B_warm = B_matrix[(l - 1), :] # warm start
    B_matrix[l, :] = penalize_sgcd(B_warm, X_, y_, alpha, lambdas_[l])  
  return B_matrix

def cross_validation_risks(X, y, K, alpha, lambdas_):
  N = X.shape[0]
  L = len(lambdas_)
  test_risks_matrix = np.zeros((L, K))

  I = np.random.permutation([i for i in range(0, N)])
  partitions = np.array_split(I, K)
  for k in range(0, K):
    partition = partitions[k]
    TRAIN = np.delete(I, partition)
    TEST = partition
    X_train = X[TRAIN, :]
    y_train = y[TRAIN]

    X_test = X[TEST, :]
    y_test = y[TEST]
    # compute error
    intercept = np.mean(y_train)
    B_matrix = warm_start_B_matrix(X_train, y_train, alpha, lambdas_)
    for l in range(0, L):
      B = B_matrix[l, :]
      test_risks_matrix[l, k] = np.linalg.norm(y_test - predict(X_test, B, intercept))

  return { "risks": np.mean(test_risks_matrix, axis = 1), "lambdas_": -np.sort(-lambdas_) } 

###
# Subgradient Coordinate descent
# step size is a really sensitive (be careful)
###
def penalize_sgcd(B, X_, y_, alpha, lambda_):
  n = X_.shape[0]
  p = X_.shape[1]
  max_iter = 100000
  tolerance = 10**(-5)
  I = np.array([i for i in range(0, p)])

  # Center
  C = np.identity(n) - 1/n * np.outer(np.ones(n),np.ones(n))
  cX = C @ X_
  cy = y_ - np.mean(y_)

  for j in range(0, max_iter):
    s = 10**(-(np.log(n)/np.log(10)))
    B_old = np.copy(B) # this is current B; keep track for stopping criterion
    for i in np.random.permutation(I):
      Bi_current = B[i] # current B[i]
      cX_i = cX[:, i]
      D_Li = -1 * cX_i.T @ (cy - (cX @ B)) # derivative of loss
      D_Pi = alpha[0] * lambda_ * Bi_current # derivative of penalization
      sub_grad_l1 = np.sign(Bi_current)

      # update
      # handle subgradient cases of l1
      if sub_grad_l1 != 0:
        B[i] = Bi_current - s * (D_Li + D_Pi + alpha[-1] * lambda_ * sub_grad_l1)
      elif sub_grad_l1 == 0:
        if D_Li + D_Pi < - alpha[-1] * lambda_:
          B[i] = Bi_current - s * (D_Li + D_Pi + alpha[-1] * lambda_)
        if D_Li + D_Pi > alpha[-1] * lambda_:
          B[i] = Bi_current - s * (D_Li + D_Pi - alpha[-1] * lambda_)
        if D_Li + D_Pi >= -alpha[-1] * lambda_ and D_Li + D_Pi <= alpha[-1] * lambda_:
          B[i] = 0

    # stop if loss is not changing much
    if (np.abs(np.linalg.norm(cy - cX @ B) - np.linalg.norm(cy - cX @ B_old)) < tolerance):
      print(j)
      return B
  return B

def sparsify(B):
  tolerance = .01
  for i in range(0, len(B)):
    if B[i] < tolerance:
      B[i] = 0.0
  return B

def predict(X, B, intercept):
  n = X.shape[0]
  return np.repeat(intercept, n) + X @ B


# Generate data
n = 1000
p = 10

A = np.array(np.repeat(3, p - int(1/2 * p))) # nonzero features
B_true = np.append(A, np.zeros(p - len(A))) # add sparsity
intercept_true = 5

X_train = np.zeros((n, p))
for i in range(0, n):
  X_train[i, :] = np.random.multivariate_normal(np.zeros(p), np.identity(p))
err_train = np.random.multivariate_normal(np.zeros(n), np.identity(n)) # error 
y_train = np.repeat(intercept_true, n) + (X_train @ B_true) + err_train

n_test = int(n/3)
X_test = np.zeros((n_test, p))
for i in range(0, n_test):
  X_test[i, :] = np.random.multivariate_normal(np.zeros(p), np.identity(p))
err_test = np.random.multivariate_normal(np.zeros(n_test), np.identity(n_test)) # error
y_test = np.repeat(intercept_true, n_test) + (X_test @ B_true) + err_test

### CV test
# CV
alpha = [1/2, 1/2]
lambdas_ = np.arange(10**(-3), 1, .1)
K_fold = 5

cv = cross_validation_risks(X_train, y_train, K_fold, alpha, lambdas_)
best_lambda =  cv["lambdas_"][np.argmin(cv["risks"])]

intercept = np.mean(y_train)
B_elnet = penalize_sgcd(np.zeros(p), X_train, y_train, alpha, best_lambda)

# train error
print(mean_squared_error(y_train, predict(X_train, B_elnet, intercept)))

# test error
print(mean_squared_error(y_test, predict(X_test, B_elnet, intercept)))

# Elnet
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV
elnet_model = ElasticNetCV(l1_ratio = 1/2)
elnet_model.fit(X_train, y_train)
print(mean_squared_error(y_test, predict(X_test, elnet_model.coef_, intercept)))




### Single fit test
alpha = [0, 1]
lambda_ = .1
intercept = np.mean(y_train)
B = sparsify(penalize_sgcd(np.zeros(p), X_train, y_train, alpha, lambda_))
B

# train error
np.linalg.norm(y_train - predict(X_train, B, intercept))

# test error
np.linalg.norm(y_test - predict(X_test, B, intercept))




