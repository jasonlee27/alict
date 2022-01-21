# This script perturb and transform inputs in datasets that meet requirements

from typing import *

import re, os
import sys
import json
import spacy
import random
import string
import checklist
import numpy as np

# from checklist.editor import Editor
from checklist.expect import Expect
from pathlib import Path

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from .sentiwordnet.Sentiwordnet import Sentiwordnet
from .Synonyms import Synonyms


random.seed(27)

class TransformOperator:

    def __init__(self,
                 requirements,
                 editor=None
                 ):
        
        self.editor = editor # checklist.editor.Editor()
        self.capability = requirements['capability']
        self.description = requirements['description']
        # self.search_dataset = search_dataset
        self.transform_reqs = requirements['transform']
        # self.inv_replace_target_words = None
        # self.inv_replace_forbidden_words = None
        self.transform_func = self.transform_reqs.split()[0]
        self.transform_props = None
        if len(self.transform_reqs.split())>1:
            self.transform_props = self.transform_reqs.split()[1]
        # end if
