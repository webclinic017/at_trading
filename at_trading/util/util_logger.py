import inspect
import logging
from logging.config import fileConfig
import os


def setup_logging(
        default_path='log.ini',
        default_level=logging.INFO,
        env_key='LOG_CFG'
):
    """

    :param default_path:
    :param default_level:
    :param env_key:
    """
    value = os.getenv(env_key, None)
    if value:
        _log_path = value
    else:
        _module_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        _log_path = '{}/{}'.format(_module_path, default_path)

    print('path {}'.format(_log_path))

    if os.path.exists(_log_path):
        print('reading from path {}'.format(_log_path))
        fileConfig(_log_path)
    else:
        print('Using basic config for logger, level[{}]'.format(str(default_level)))
        logging.basicConfig(level=default_level)
