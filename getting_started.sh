# <This script needs to be ran from within Harmonic root directory>

# Install core and extra requirements

pip install -r requirements/requirements-core.txt
pip install -r requirements/requirements-examples.txt
pip install -r requirements/requirements-extra.txt
pip install -r requirements/requirements-docs.txt

# Install specific converter for building tutorial documentation

conda install pandoc=1.19.2.1 -y

# Build Harmonic

python setup.py build_ext --inplace