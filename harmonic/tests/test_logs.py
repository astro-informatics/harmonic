import harmonic.logs as lg
import pytest
import numpy as np

def test_add_chains_2d_and_copy():

    dir_name = "random/incorrect/filepath/"

    # Check cannot add samples with different ndim.
    with pytest.raises(ValueError):
        lg.setup_logging(custom_yaml_path=dir_name)
        



        