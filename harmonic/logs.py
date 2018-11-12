import os
import logging.config
import logging
import yaml
import harmonic
import os


def setup_logging(

    default_level=logging.DEBUG,
    env_key='LOG_CFG'
):
    """Call at the begining of code to initialize and configure
    the desired logging level.

    Args:
        int: logging level at which to configure.
        string: Environment key. Do not touch this.

    Returns:
        None.

    Raises:
        None.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(harmonic.__file__))) + '/logs/logging.yaml')
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        config['handlers']['info_file_handler']['filename'] = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(harmonic.__file__))) + '/logs/info.logs')
        config['handlers']['error_file_handler']['filename'] = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(harmonic.__file__))) + '/logs/errors.logs')
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

"""
Custom low-level logger for Harmonic: (Cyan) Use for standard debug prints.

"""
def low_log(message):
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


def high_log(message):
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
In main code, call lines (1) and (2) to create and initialize the logger:

(1) from harmonic import logs as log
(2) log.setup_logging(default_level=[level that you want to log at e.g. logging.DEBUG])

examples of use:
        log.Harmonic_low_log('a debug level message')
        log.Harmonic_high_log('a critical level message')
"""


