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

class Coveragedata:

    @classmethod
    def get_sa_requirement(cls, task):
        return Requirements.get_requirements(task)
    
    @classmethod
    def get_checklist_sents(cls, task, testsuite_file):
        requirements = cls.get_sa_requirement(task)
        sents = {
            req['description']: ChecklistTestsuite.get_sents(
                testsuite_file,
                req['description']
            )
            for req in requirements
        }
        # sents = Utils.read_txt('./data/checklist_sents.txt')
        return sents
    
    @classmethod
    def get_our_seed_sents(cls, task, dataset_name):
        data_file = Macros.result_dir / f"seed_inputs_{task}_{dataset_name}.json"
        seed_dict = Utils.read_json(data_file)
        sents = {
            seed['requirement']['description']: seed['seeds']
            for seed in seed_dict
        }
        return sents

    # @classmethod
    # def get_checklist_exp_sents(cls):
    #     data_file = Macros.result_dir / f"cfg_expanded_inputs_sa_checklist_random.json"
    #     sent_dict = Utils.read_json(data_file)
    #     sents = dict()
    #     for s in seed_dict:
    #         sents[s['requirement']['description']] = {
    #             'seed': list(s['inputs'].keys()),
    #             'exp': [
    #                 exp_obj[5]
    #                 for seed_key in s['inputs'].keys()
    #                 for exp_obj in s['inputs'][seed_key]['exp_inputs']
    #             ]
    #         }
    #     # end for
    #     return sents

    @classmethod
    def get_our_exp_sents(cls,
                          task,
                          dataset_name,
                          selection_method):
        data_file = Macros.result_dir / f"cfg_expanded_inputs_{task}_{dataset_name}_{selection_method}.json"
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
    def sample_seed_sents(cls, bl_sents, our_sents, num_samples=12403):
        for lc in bl_sents.keys():
            _bl_sents = bl_sents[lc]
            _our_sents = our_sents[lc]
            num_samples = min(len(_bl_sents), len(_our_sents))
            if len(_bl_sents)>len(_our_sents):
                ind_list = list(range(len(_bl_sents)))
                random.shuffle(ind_list)
                _bl_sents = [_bl_sents[ind] for ind in ind_list[:num_samples]]
            elif len(_bl_sents)<len(_our_sents):
                ind_list = list(range(len(_our_sents)))
                random.shuffle(ind_list)
                _our_sents = [_our_sents[ind] for ind in ind_list[:num_samples]]
            # end if
            
        # end for
        return sampled_sents

    @classmethod
    def write_target_seed_sents(cls, task, dataset_name, num_samples=-1):
        bl_sents = cls.get_checklist_sents(task,
                                           testsuite_file=Macros.checklist_sa_dataset_file)
        our_sents = cls.get_our_seed_sents(task, dataset_name)
        if num_samples>0:
            bl_sents, our_sents = cls.sample_sents(bl_sents,
                                                   our_sents,
                                                   num_samples=num_samples)
        # end if

        # write testcase sentences
        save_dir = Macros.python_dir / task /'coverage' / 'data'
        for key in our_sents.keys():
            cksum_val = Utils.get_cksum(key)
            sent_str = '\n'.join([s[1] for s in our_sents[key]])
            Utils.write_txt(sent_str, save_dir / f'seed_{task}_{dataset_name}_{cksum_val}.txt')
        # end for

        for key in bl_sents.keys():
            cksum_val = Utils.get_cksum(key)
            sent_str = '\n'.join([s[1] for s in bl_sents[key]])
            Utils.write_txt(sent_str, save_dir / f'checklist_{task}_{dataset_name}_{cksum_val}.txt')
        # end for
        return

    @classmethod
    def write_target_exp_sents(cls,
                               task,
                               dataset_name,
                               selection_method,
                               num_samples=-1):
        bl_sents = cls.get_checklist_sents(task,
                                           testsuite_file=Macros.checklist_sa_dataset_file)
        our_sents = cls.get_our_exp_sents(task,
                                          dataset_name,
                                          selection_method)
        if num_samples>0:
            bl_sents, our_sents = cls.sample_sents(bl_sents,
                                                   our_sents,
                                                   num_samples=num_samples)
        # end if

        # write testcase sentences
        save_dir = Macros.python_dir / task / 'coverage' / 'data'
        for key in our_sents.keys():
            cksum_val = Utils.get_cksum(key)
            sent_str = '\n'.join(our_sents[key])
            Utils.write_txt(sent_str, save_dir / f'seed_{task}_{dataset_name}_{cksum_val}.txt')
        # end for

        for key in bl_sents.keys():
            cksum_val = Utils.get_cksum(key)
            sent_str = '\n'.join(bl_sents[key])
            Utils.write_txt(sent_str, save_dir / f'checklist_{task}_{dataset_name}_{cksum_val}.txt')
        # end for
        return
