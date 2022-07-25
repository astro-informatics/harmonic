import sys
import os
import shutil
import setuptools
from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

# clean previous build
for root, dirs, files in os.walk("./harmonic/", topdown=False):
    for name in dirs:
        if (name == "build"):
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

include_dirs = [numpy.get_include(),]

extra_link_args=[]

setup(
    classifiers=['Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 'Operating System :: OS Independent',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research'
                 ],
    name = "harmonic",
    version = "1.1.1",
    prefix='.',
    url='https://github.com/astro-informatics/harmonic',
    author='Jason D. McEwen, Christopher G. R. Wallis, Matthew A. Price, Matthew M. Docherty & Contributors',
    author_email='jason.mcewen@ucl.ac.uk',
    license='GNU General Public License v3 (GPLv3)',
    install_requires=required,
    description='Python package for efficient Bayesian evidence computation',
    long_description_content_type = "text/x-rst",
    long_description = long_description,
    packages=['harmonic'],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([
    Extension(
        "harmonic.model",
        package_dir=['harmonic'],
        sources=["harmonic/model.pyx"],
        include_dirs=include_dirs,
        libraries=[],
        extra_link_args=extra_link_args,
        extra_compile_args=[]
    ),        
    Extension(
        "harmonic.chains",
        package_dir=['harmonic'],
        sources=["harmonic/chains.pyx"],
        include_dirs=include_dirs,
        libraries=[],
        extra_link_args=extra_link_args,
        extra_compile_args=[]
    ),
    Extension(
        "harmonic.utils",
        package_dir=['harmonic'],
        sources=["harmonic/utils.pyx"],
        include_dirs=include_dirs,
        libraries=[],
        extra_link_args=extra_link_args,
        extra_compile_args=[]
    ),
    Extension(
        "harmonic.logs",
        package_dir=['harmonic'],
        sources=["harmonic/logs.py"],
        include_dirs=include_dirs,
        libraries=[],
        extra_link_args=extra_link_args,
        extra_compile_args=[]
    ),
    Extension(
        "harmonic.evidence",
        package_dir=['harmonic'],
        sources=["harmonic/evidence.pyx"],
        include_dirs=include_dirs,
        libraries=[],
        extra_link_args=extra_link_args,
        extra_compile_args=[]
    )],
    compiler_directives={'linetrace': True})
)
