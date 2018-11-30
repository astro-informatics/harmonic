import os
import logging.config
import logging
import yaml
import harmonic
import os


def setup_logging(custom_yaml_path=None, default_level=logging.DEBUG):
    """
    Initialize and configure logging.
    
    Should be called at the beginning of code to initialize and configure the 
    desired logging level. Logging levels can be ints in [0,50] where 10 is 
    debug logging and 50 is critical logging.
    Args:
        - custom_yaml_path: 
            Complete pathname of desired yaml logging configuration. If empty 
            will provide Harmonics default logging config.
        - default_level: 
            Logging level at which to configure.
    Raises:
        - ValueError:
            Raised if Harmonic's logging.yaml is not in src_harmonic/logs/ 
            directory.
        
    """
    if custom_yaml_path == None:
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.realpath(harmonic.__file__))) + '/logs/logging.yaml')
    if custom_yaml_path != None:
        path = custom_yaml_path
    value = os.getenv('LOG_CFG', None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        if custom_yaml_path == None:
            config['handlers']['info_file_handler']['filename'] = os.path.join(
                os.path.dirname(os.path.dirname(
                    os.path.realpath(harmonic.__file__))) + '/logs/info.log')
            config['handlers']['error_file_handler']['filename'] = os.path.join(
                os.path.dirname(os.path.dirname(
                    os.path.realpath(harmonic.__file__))) + '/logs/errors.log')
            config['handlers']['critical_file_handler']['filename'] = os.path.join(
                os.path.dirname(os.path.dirname(
                    os.path.realpath(harmonic.__file__))) + '/logs/critical.log')
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        raise ValueError("Logging config pathway incorrect.")
    high_log('Using config from {}'.format(path))


def low_log(message):
    """
    Log low-level (DEBUG) message. 
    Args:
        - message: 
            Message to log.
    """
    logger = logging.getLogger('Harmonic')
    logger.debug('\033[0;36;40m ' + message + ' \033[0;0m')


def high_log(message):
    """
    Log high-level (CRITICAL) message
    Args:
        - message: 
            Message to log.
    """
    logger = logging.getLogger('Harmonic')
    logger.critical('\033[1;31;40m' + message + '\033[0;0m')

""" 
In main code, call lines (1) and (2) to create and initialize the logger:
(1) import harmonic as hm 
(2) hm.logs.setup_logging()
(note) if you wish to use a custom logging configuration simply provide the 
pathname to your yaml as the argument to setup_logging('pathname').
examples of use:
        hm.logs.low_log('a debug level message')
        hm.logs.high_log('a critical level message')
"""