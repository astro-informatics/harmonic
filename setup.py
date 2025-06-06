import os
import shutil
from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

# clean previous build
for root, dirs, files in os.walk("./harmonic/", topdown=False):
    for name in dirs:
        if name == "build":
            shutil.rmtree(name)

from os import path

this_directory = path.abspath(path.dirname(__file__))


def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()


def read_file(file):
    with open(file) as f:
        return f.read()


long_description = read_file(".pip_readme.rst")
required = read_requirements("requirements/requirements-core.txt")

include_dirs = [
    numpy.get_include(),
]

extra_link_args = []

setup(
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    name="harmonic",
    version="1.3.0",
    prefix=".",
    url="https://github.com/astro-informatics/harmonic",
    author="Jason D. McEwen, Alicja Polanska, Christopher G. R. Wallis, Matthew A. Price, Matthew M. Docherty & Contributors",
    author_email="jason.mcewen@ucl.ac.uk",
    license="GNU General Public License v3 (GPLv3)",
    install_requires=required,
    description="Python package for efficient Bayesian evidence computation",
    long_description_content_type="text/x-rst",
    long_description=long_description,
    packages=["harmonic"],
    include_package_data=True,
    package_data={"harmonic": ["default-logging-config.yaml"]},
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(
        [
            Extension(
                "harmonic.model_legacy",
                package_dir=["harmonic"],
                sources=["harmonic/model_legacy.pyx"],
                define_macros=[("CYTHON_TRACE", "1")],
                include_dirs=include_dirs,
                libraries=[],
                extra_link_args=extra_link_args,
                extra_compile_args=[],
            ),
        ],
        compiler_directives={"linetrace": True, "language_level": "2"},
    ),
)
