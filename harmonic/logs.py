import os
import logging.config
import logging
from pathlib import Path
import yaml
import harmonic


def setup_logging(custom_yaml_path=None, default_level=logging.DEBUG):
    """initialise and configure logging.

    Should be called at the beginning of code to initialise and configure the
    desired logging level. Logging levels can be ints in [0,50] where 10 is
    debug logging and 50 is critical logging.

    Args:

        custom_yaml_path (string): Complete pathname of desired yaml logging
            configuration. If empty will provide default logging config.

        default_level (int): Logging level at which to configure.

    Raises:

        ValueError: Raised if logging.yaml is not in ./logs/ directory.

    """
    if "LOG_CFG" in os.environ:
        path = Path(os.environ["LOG_CFG"])
    elif custom_yaml_path is None:
        path = Path(harmonic.__file__).parent / "default-logging-config.yaml"
    else:
        path = Path(custom_yaml_path)
    if not path.exists():
        raise ValueError(f"Logging config path {path} does not exist.")
    with open(path, "rt") as f:
        config = yaml.safe_load(f.read())
    if custom_yaml_path is None:
        config["handlers"]["info_file_handler"]["filename"] = "info.log"
        config["handlers"]["debug_file_handler"]["filename"] = "debug.log"
        config["handlers"]["critical_file_handler"]["filename"] = "critical.log"
    logging.config.dictConfig(config)


def debug_log(message):
    """Log a debug message (e.g. for background logs to assist debugging).

    Args:

        message: Message to log.

    """
    logger = logging.getLogger("Harmonic")
    logger.debug(message)


def warning_log(message):
    """Log a warning (e.g. for internal code warnings such as large dynamic
    ranges).

    Args:

        message: Warning to log.

    """
    logger = logging.getLogger("Harmonic")
    logger.warning(message)


def critical_log(message):
    """Log a critical message (e.g. core code failures etc).

    Args:

        message: Message to log.

    """
    logger = logging.getLogger("Harmonic")
    logger.critical(message)


def info_log(message):
    """Log an information message (e.g. evidence value printing, run completion
    etc).

    Args:

        message: Message to log.

    """
    logger = logging.getLogger("Harmonic")
    logger.info(message)
