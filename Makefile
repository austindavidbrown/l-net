
develop:
	clang++ -Wall -Wextra -std=c++17 -I ./eigen/ pros_test.cpp -O3 -march=native -o pros_test
	./pros_test

production:
	clang++ -std=c++17 -I ./eigen/ test.cpp -o test -DNDEBUG -O3 -march=native -mfpmath=sse && ./test

build_R:
	R -f R_build.R

build_python:
	pip install ./python-package







# References:
# clang++ -Wall -Wextra -std=c++17 -I ./eigen/ -O3 -march=native -mfpmath=sse sgcd.cpp -o sgcd && ./sgcd

# vectorize and improve math
# -mfpmath=sse -march=native -funroll-loops
# Use BLAS/LAPACK
# -DEIGEN_USE_BLAS -framework Accelerate
# Use openmp
# -L/usr/local/opt/llvm/lib -I/usr/local/opt/llvm/include -fopenmp