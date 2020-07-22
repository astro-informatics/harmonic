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
with open(path.join(this_directory, '.pip_readme.md'), encoding='utf-8') as f:
    long_description = f.read()
with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    required = f.read().splitlines()


include_dirs = [
    numpy.get_include(),
#    os.environ['code']+"ssht/include/c",
    ]

extra_link_args=[
#    "-L"+os.environ['code']+"ssht/lib/c",
#    "-L"+os.environ['FFTW']+"/lib",
]

setup(
    classifiers=['Programming Language :: Python :: 3',
                 'Operating System :: OS Independent',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
                 ],
    name = "harmonic",
    version = "0.13",
    prefix='.',
    url='https://github.com/astro-informatics/harmonic',
    author='Jason McEwen & Constributors',
    author_email='jason.mcewen@ucl.ac.uk',
    install_requires=required,
    description='Python package for efficient Bayesian evidence computation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(where='src'),
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([
    Extension(
        "harmonic.model",
        package_dir=['src'],
        sources=["harmonic/model.pyx"],
        include_dirs=include_dirs,
        libraries=[],
        extra_link_args=extra_link_args,
        extra_compile_args=[]
    ),        
    Extension(
        "harmonic.chains",
        package_dir=['src'],
        sources=["harmonic/chains.pyx"],
        include_dirs=include_dirs,
        libraries=[],
        extra_link_args=extra_link_args,
        extra_compile_args=[]
    ),
    Extension(
        "harmonic.utils",
        package_dir=['src'],
        sources=["harmonic/utils.pyx"],
        include_dirs=include_dirs,
        libraries=[],
        extra_link_args=extra_link_args,
        extra_compile_args=[]
    ),
    Extension(
        "harmonic.logs",
        package_dir=['src'],
        sources=["harmonic/logs.py"],
        include_dirs=include_dirs,
        libraries=[],
        extra_link_args=extra_link_args,
        extra_compile_args=[]
    ),
    Extension(
        "harmonic.evidence",
        package_dir=['src'],
        sources=["harmonic/evidence.pyx"],
        include_dirs=include_dirs,
        libraries=[],
        extra_link_args=extra_link_args,
        extra_compile_args=[]
    )],
    compiler_directives={'linetrace': True})
)
