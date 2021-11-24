# <This script needs to be ran from within Harmonic root directory>

# Install core and extra requirements

echo -ne 'Building Dependencies... \r'
pip install -q -r requirements/requirements-core.txt
echo -ne 'Building Dependencies... #####               (25%)\r'
pip install -q -r requirements/requirements-examples.txt
echo -ne 'Building Dependencies... ##########          (50%)\r'
pip install -q -r requirements/requirements-test.txt
echo -ne 'Building Dependencies... ###############     (75%)\r'
pip install -q -r requirements/requirements-docs.txt
echo -ne 'Building Dependencies... ####################(100%)\r'
echo -ne '\n'

# Install specific converter for building tutorial documentation

conda install pandoc=1.19.2.1 -y

# Build Harmonic

python setup.py build_ext --inplace
pip install .