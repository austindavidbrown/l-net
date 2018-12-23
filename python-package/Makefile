develop:
	./env/bin/python3 setup.py check build_ext --inplace
	./env/bin/python3 package_test.py

pip_install:
	./env/bin/pip3 install ./
	./env/bin/python3 package_test.py

python_setup:
	virtualenv -p python3 ./env
	./env/bin/pip3 install numpy

build_docs:
	./env/bin/python3 -m pydoc -w lnet
