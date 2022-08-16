# This script searches inputs in datasets that meet requirements

from typing import *
from pathlib import Path

import re, os
import sys
import json
import spacy
import random
import checklist
import numpy as np

from itertools import product
from checklist.test_types import MFT, INV, DIR

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..requirement.Requirements import Requirements
from ..testsuite.Search import ChecklistTestsuite, Sst

class Extractdata:

    @classmethod
    def get_sa_requirement(cls, task):
        return Requirements.get_requirements(task)

    @classmethod
    def get_our_seed(cls, task, data_file):
        
    
    @classmethod
    def get_checklist_sents(cls, task, testsuite_file):
        requirements = cls.get_sa_requirement(task)
        sents = dict()
        for req in requirements:
            _sents = ChecklistTestsuite.get_sents(testsuite_file,
                                                  req['description'])
            sents[req['description']] = _sents
        # end for
        return sents

    @classmethod
    def get_our_seed_sents(cls, task, dataset_name):
        if dataset_name==Macros.datasets[task][0]:
            sents = cls.get_our_seed(task,
                                     data_file=Macros.sst_datasent_file)
        elif dataset_name==Macros.datasets[task][1]:
            sents = cls.get_checklist_sents(task,
                                            testsuite_file=Macros.checklist_sa_dataset_file)
        # end if
        return sents
