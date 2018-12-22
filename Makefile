
develop:
	clang++ -Wall -Wextra -std=c++17 -I./include/eigen/ lnet_test.cpp -O3 -march=native -mfpmath=sse -o lnet_test
	./lnet_test

build_R:
	R -f R_build.R

build_python:
	pip install ./python-package







# References:
# -DNDEBUG
# clang++ -Wall -Wextra -std=c++17 -I ./eigen/ -O3 -march=native -mfpmath=sse sgcd.cpp -o sgcd && ./sgcd

# vectorize and improve math
# -mfpmath=sse -march=native -funroll-loops
# Use BLAS/LAPACK
# -DEIGEN_USE_BLAS -framework Accelerate
# Use openmp
# -L/usr/local/opt/llvm/lib -I/usr/local/opt/llvm/include -fopenmp