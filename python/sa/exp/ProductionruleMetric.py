import os
import nltk
import multiprocessing

from multiprocessing import Pool
# from functools import partial
from nltk.translate.bleu_score import SmoothingFunction

from ..utils.Macros import Macros
from ..utils.Utils import Utils


class ProductionruleMetric:

    
