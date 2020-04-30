# Harmonic


[![Build Status](https://travis-ci.com/astro-informatics/src_harmonic.svg?token=quDUMr3yVpQwGYxko5xh&branch=master)](https://travis-ci.com/astro-informatics/src_harmonic)


[![Arxiv Status ](http://img.shields.io/badge/arXiv-2004.07855-orange.svg?style=flat)](https://arxiv.org/abs/2004.07855)




# Installation

## Set up conda environment

```conda create --name harmonic python=3.6```

## Install dependencies for `harmonic` package

`pip install -r requirements.txt`


## Install dependencies for `harmonic` examples

`pip install -r requirements-examples.txt`

## Build `harmonic`

`python setup.py build_ext --inplace`

Run tests

`pytest`

Run examples

From within the harmonic root directory

`python example/<example_name>`

e.g.

`python examples/rastrigin.py`


