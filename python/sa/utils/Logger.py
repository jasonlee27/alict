
import logging

from pathlib import Path

from .Macros import Macros

class Logger:

    def __init__(self, logger_file: Path, logger_name=None):
        self.logger_file = logger_file
        self.logger_name = logger_name if logger_name is not None else __name__
        self.logger = self.get_logger()
        
    def get_logger(self):
        logging.basicConfig(filename=self.logger_file, filemode='w')
        return logging.getLogger(self.logger_name)
    
    def print(self, text):
        self.logger.info(text)

    def debug(self, text):
        self.logger.debug(text)
        
