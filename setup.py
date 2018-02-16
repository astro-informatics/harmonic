import sys
import os
import shutil

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

# clean previous build
for root, dirs, files in os.walk("./harmonic/", topdown=False):
    for name in dirs:
        if (name == "build"):
            shutil.rmtree(name)


include_dirs = [
    numpy.get_include(),
#    os.environ['code']+"ssht/include/c",
    ]

extra_link_args=[
#    "-L"+os.environ['code']+"ssht/lib/c",
#    "-L"+os.environ['FFTW']+"/lib",
]

setup(
    classifiers=['Programming Language :: Python :: 3.x'],
    name = "harmonic",
    version = "0.0",
    prefix='.',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([Extension(
        "harmonic/harmonic",
        package_dir=['src'],
        sources=["harmonic/harmonic.pyx"],
        include_dirs=include_dirs,
        libraries=[],#["ssht", "fftw3"],
        extra_link_args=extra_link_args,
        extra_compile_args=[]
    ),
    Extension(
        "harmonic/model",
        package_dir=['src'],
        sources=["harmonic/model.pyx"],
        include_dirs=include_dirs,
        libraries=[],#["ssht", "fftw3"],
        extra_link_args=extra_link_args,
        extra_compile_args=[]
    ),        
    Extension(
        "harmonic/tmp_poly_class_example",
        package_dir=['src'],
        sources=["harmonic/tmp_poly_class_example.pyx"],
        include_dirs=include_dirs,
        libraries=[],#["ssht", "fftw3"],
        extra_link_args=extra_link_args,
        extra_compile_args=[]
    )])
)