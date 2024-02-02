# cython:
# 	rm -rf build && python setup.py build_ext && python setup.py bdist_wheel
# cython_infer:
# 	rm -rf build && python setup_infer_only.py build_ext && python setup.py bdist_wheel
build_cython:
	rm -rf build && python setup.py bdist_wheel
build:
	poetry build
# build_private:
# 	BUILD_TYPE=PRIVATE python build.py bdist_wheel
clean_build:
	rm -rf build
test:
	nox