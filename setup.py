# from setuptools import setup, find_packages, Extension
# from Cython.Build import cythonize
# import os
#
#
# # recursively find all *.py files in the directory 'x'
# to_compile = [f for f in os.popen('find optimus -name "*.py"')]
# print(to_compile)
# to_compile = [Extension("*", [f[:-1]], ) for f in to_compile]
#
# setup(
#     name="optimus",
#     version="0.4.1",
#     packages=find_packages(),  # set this to include any pure python packages
#     ext_modules=cythonize(to_compile, compiler_directives={'language_level': '3'}),
# )

import os

# from setuptools.command.build_py import build_py as build_py_orig
from Cython.Build import cythonize

# import fnmatch
# NOTE: import setuptools first: https://stackoverflow.com/a/53356077/19204579
from setuptools import find_packages, setup

from dlkit.__init__ import __version__

# ENCRYPTED = [
#     "optimus/__init__.py",
#     "optimus/cli.py",
#     "optimus/models/__init__.py",
#     "optimus/models/bases.py",
#     "optimus/models/gpt.py",
# ]

# encrypted = ENCRYPTED
package_name = "dlkit"
to_be_compiled = [f.rstrip() for f in os.popen(f'find {package_name} -name "*.py"')]
print(to_be_compiled)
# encrypted = [
#     'dlkit/__init__.py',
#     'dlkit/checks.py',
# ]
private_list = [
    # "dlkit/train.py",
]

to_be_compiled = [f for f in to_be_compiled if f not in private_list]

# NOTE: overwrite build_py to exclude file in bdist:
# https://github.com/pypa/setuptools/issues/511#issuecomment-570177072
# https://stackoverflow.com/a/50517893
# class build_py(build_py_orig):
#     def find_package_modules(self, package, package_dir):
#         modules = super().find_package_modules(package, package_dir)
#         return [
#             (pkg, mod, file)
#             for (pkg, mod, file) in modules
#             if file in to_be_compiled #and file.endswith('.so')
#             # if not any(fnmatch.fnmatchcase(file, pat=pattern) for pattern in to_be_compiled)
#         ]


setup(
    name=package_name + "_cython",
    version=__version__,
    ext_modules=cythonize(to_be_compiled, compiler_directives={"language_level": "3"}),
    # cmdclass={"build_py": build_py},
)
