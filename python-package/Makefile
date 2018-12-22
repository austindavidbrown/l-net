develop:
	./env/bin/python3 setup.py check build_ext --inplace
	./env/bin/python3 test.py

pip_install:
	./env/bin/pip3 install ./
	./env/bin/python3 test.py

python_setup:
	virtualenv -p python3 ./env