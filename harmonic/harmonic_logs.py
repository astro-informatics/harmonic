import os
import logging.config
import logging
import yaml


def setup_logging(

    default_path='/Users/matt/Downloads/Software/src_harmonic/logs/logging.yaml',
    default_level=logging.DEBUG,
    env_key='LOG_CFG'
):
    """Call at the begining of code to initialize and configure
    the desired logging level.

    Args:
        os path: Directory location of .yaml configure file.
        int: logging level at which to configure.
        string: Environment key. Do not touch this.

    Returns:
        None.

    Raises:
        None.
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

"""
Custom low-level logger for Harmonic: (Cyan) Use for standard debug prints.

"""
def Harmonic_low_log(message):
    """Custom low-level logger for Harmonic

    Args:
        string: message which should be logged.

    Returns:
        None.

    Raises:
        None.
    """
    logger = logging.getLogger('Harmonic')
    logger.debug('\033[0;36;40m' + message + '\033[0;0m')


def Harmonic_high_log(message):
    """Custom high-level logger for Harmonic

    Args:
        string: message which should be logged.

    Returns:
        None.

    Raises:
        None.
    """
    logger = logging.getLogger('Harmonic')
    logger.critical('\033[1;31;40m' + message + '\033[0;0m')

""" 
In main code, call lines (1) and (3) to create and initialize the logger:

(1) import logging
(2) import harmonic.harmonic_logs as lg 
(3) lg.setup_logging(default_level=[level that you want to log at e.g. logging.DEBUG])

examples of use:
        lg.Harmonic_low_log('a debug level message')
        lg.Harmonic_high_log('a critical level message')
"""


