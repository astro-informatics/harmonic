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
    """
    .. note:: Call at the begining of code to initialize and configure
              the desired logging level.

    Args:
        - int: 
            logging level at which to configure.
        - string:  
            Environment key. Do not touch this.

    Raises:
        - ValueError:
            Raised if logging.yaml is not in src_harmonic/logs/ directory.
        
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(harmonic.__file__))) + '/logs/logging.yaml')
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        config['handlers']['info_file_handler']['filename'] = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(harmonic.__file__))) + '/logs/info.log')
        config['handlers']['error_file_handler']['filename'] = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(harmonic.__file__))) + '/logs/errors.log')
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        raise ValueError("Logging config pathway incorrect.")


def low_log(message):
    """
    .. note:: Custom low-level logger for Harmonic

    Args:
        - string: 
            message which should be logged.

    """
    logger = logging.getLogger('Harmonic')
    logger.debug('\033[0;36;40m' + message + '\033[0;0m')


def high_log(message):
    """
    .. note:: Custom high-level logger for Harmonic

    Args:
        - string: 
            message which should be logged.

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


