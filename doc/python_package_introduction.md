Python Package Introduction
===

Install l-net
---

```bash
pip install numpy # install numpy
pip install git+https://github.com/austindavidbrown/l-net/#egg=l-net\&subdirectory=python-package
```

Generate sparse data with numpy or load a dataset
---

```python
import numpy as np

n = 100
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
```

Train and predict
---

```python
import lnet as lnet

alpha = np.array([1, 0, 0, 0, 0, 0])
lambdas = np.array([.1, .5, 1, 2, 3, 4])
step_size = 1/100;

cv = lnet.LnetCV(X = X_train, y = y_train, alpha = alpha, lambdas = lambdas, step_size = step_size)
pred = cv.predict(X = X_test)

print("MSE on the test set: ", ((y_test - pred)**2).mean())

## MSE on the test set:  6.9760444370031585e-06
```
