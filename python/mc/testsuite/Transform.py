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
                 editor,
                 req_capability,
                 req_description,
                 transform_reqs
                 ):
        
        self.editor = editor # checklist.editor.Editor()
        self.capability = req_capability
        self.description = req_description
        self.transform_reqs = transform_reqs

        if 'MFT' in transform_reqs.keys() or transform_reqs["MFT"] is not None:
            self.set_mft_env(transform_reqs['MFT'])
        # end if
    
    def set_mft_env(self, mft_transform_reqs):
        return

