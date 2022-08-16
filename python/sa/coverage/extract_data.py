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
        seed_dict = Utils.read_json(data_file)
        sents = dict()
        for seed in seed_dict:
            sents[seed['requirement']['description']] = seed['seeds']
        # end for
        return sents
    
    @classmethod
    def get_checklist_sents(cls, task, testsuite_file):
        # requirements = cls.get_sa_requirement(task)
        # sents = dict()
        # for req in requirements:
        #     _sents = ChecklistTestsuite.get_sents(testsuite_file,
        #                                           req['description'])
        #     sents[req['description']] = _sents
        # # end for
        sents = Utils.read_txt('./data/checklist_sents.txt')
        return sents

    @classmethod
    def get_our_seed_sents(cls, task, dataset_name):
        data_file = Macros.result_dir / f"seed_inputs_{task}_{dataset_name}.json"
        sents = cls.get_our_seed(task,
                                 data_file=data_file)
        return sents

    @classmethod
    def get_checklist_exp_sents(cls):
        data_file = Macros.result_dir / f"cfg_expanded_inputs_sa_checklist_random.json"
        sent_dict = Utils.read_json(data_file)
        sents = dict()
        for s in seed_dict:
            sents[s['requirement']['description']] = {
                'seed': list(s['inputs'].keys()),
                'exp': [
                    exp_obj[5]
                    for seed_key in s['inputs'].keys()
                    for exp_obj in s['inputs'][seed_key]['exp_inputs']
                ]
            }
        # end for
        return sents

    @classmethod
    def sample_sents(cls, sents, num_samples=12403):
        # TODO: how to sample?
        return sampled_sents

    @classmethod
    def write_target_sents(cls, task, dataset_name):
        bl_sents = cls.get_checklist_sents(task,
                                           testsuite_file=Macros.checklist_sa_dataset_file)
        our_sents = cls.get_our_seed_sents(task, dataset_name)
        our_sents = cls.sample_sents(our_sents, num_samples=len(bl_sents))
        sent_str = ''
        for s in our_sents:
            sent_str += f"{s}\n"
        # end for
        Utils.write_txt(sent_str, f'./data/seed_{task}_{dataset_name}_sents.txt')

        sents = cls.get_checklist_exp_sents()
        # TODO: sample seed+exp sentences
        Utils.write_txt(sent_str, f'./data/sa_checklist_sents.txt')
        return
