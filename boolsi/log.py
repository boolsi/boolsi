import logging
import logging.config
from os import path, makedirs

from .constants import log_date_format


class DelegatingFormatter(logging.Formatter):
    """
    Formatter changing format based on the log level.
    """
    formats = {
        logging.WARNING: logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt=log_date_format),
        logging.ERROR: logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt=log_date_format),
        'DEFAULT': logging.Formatter('%(asctime)s %(message)s', datefmt=log_date_format)
    }

    def format(self, record):
        """
        Format log record.

        :param record: log record
        :return: formatted log record
        """
        f = self.formats.get(record.levelno, self.formats['DEFAULT'])
        return f.format(record)


def configure_logging(output_directory):
    """
    Configure logging.

    :param output_directory: output directory
    :return: None
    """
    makedirs(output_directory, exist_ok=True)

    config = {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'delegating': {
                '()': DelegatingFormatter
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'delegating',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'delegating',
                'filename': path.join(output_directory, 'boolsi.log'),
                'mode': 'w'
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
        }
    }

    logging.config.dictConfig(config)
