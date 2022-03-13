
import logging

from pathlib import Path

from .Macros import Macros

class Logger:

    def __init__(self, logger_file: Path, logger_name=None, _format=None):
        self.logger_file = logger_file
        self.logger_name = logger_name if logger_name is not None else __name__
        self.terminator = ''
        self._format=_format
        self.logger = self.get_logger()
        
    def get_logger(self):
        logging.basicConfig(filename=self.logger_file, filemode='w', format=_format)
        logger = logging.getLogger(self.logger_name)
        h = logging.StreamHandler(terminator=self.terminator)
        logger.addHandler(h)
        return logger
    
    def print(self, text, end='\n'):
        self.logger.info(text+end)

    def debug(self, text, end='\n'):
        self.logger.debug(text+end)
        
