# <This script needs to be ran from within Harmonic root directory>

# Install core and extra requirements

echo -ne 'Building Dependencies... \r'

# Install jax and TFP on jax substrates (on M1 mac)
# conda install -q -c conda-forge jax==0.4.1 -y
# conda install -q -c conda-forge flax==0.6.1 chex==0.1.6 -y

pip install -q -r requirements/requirements-core.txt
echo -ne 'Building Dependencies... #####               (25%)\r'
pip install -q -r requirements/requirements-examples.txt
echo -ne 'Building Dependencies... ##########          (50%)\r'
pip install -q -r requirements/requirements-test.txt
echo -ne 'Building Dependencies... ###############     (75%)\r'
pip install -q -r requirements/requirements-docs.txt
echo -ne 'Building Dependencies... ####################(100%)\r'
echo -ne '\n'

# pip install -Uq tfp-nightly[jax]==0.20.0.dev20230801 > /dev/null

# Install specific converter for building tutorial documentation

# conda install pandoc=1.19.2.1 -y

# Build Harmonic

python setup.py build_ext --inplace
pip install -e .