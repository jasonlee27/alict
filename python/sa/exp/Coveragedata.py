
from typing import *

import re, os
import nltk
import copy
import random
import numpy
import spacy

from pathlib import Path
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger


class Coveragedata:

    @classmethod
    def read_sst_testcase(cls,
                          task,
                          dataset_name,
                          selection_method,
                          num_seeds,
                          num_trial):
        _num_trial = '' if num_trial==1 else str(num_trial)
        test_results_dir = Macros.result_dir / f"templates{_num_trial}_{task}_{dataset_name}_{selection_method}_{num_seeds}seeds"
        cksum_vals = [
            (l.strip().split('\t')[0], l.strip().split('\t')[1])
            for l in Utils.read_txt(test_results_dir / 'cksum_map.txt')
        ]
        
        test_data = dict()
        for lc_desc, cksum_val in cksum_vals:
            seed_file = test_results_dir / f"seeds_{cksum_val}.json"
            exp_file = test_results_dir / f"exps_{cksum_val}.json"
            seeds = [s['input'] for s in Utils.read_json(seed_file)]
            exps = [e['input'] for e in Utils.read_json(exp_file)]
            test_data[lc_desc]= {
                'sents': seeds+exps,
                'cksum': cksum_val
            }
        # end for
        return test_data

    @classmethod
    def read_checklist_testcase(cls, input_file):
        tsuite, tsuite_dict = Utils.read_testsuite(input_file)
        test_names = list(set(tsuite_dict['test_name']))
        test_data = dict()
        num_data = 0
        for test_name in test_names:
            if test_name in Macros.CHECKLIST_LC_LIST:
                lc_desc = Macros.LC_MAP[test_name]
                cksum_val = Utils.get_cksum(lc_desc)
                sents = tsuite.tests[test_name].data
                test_data[lc_desc] = {
                    'sents': tsuite.tests[test_name].data,
                    'cksum': cksum_val
                    # 'labels': tsuite.tests[test_name].labels
                }
            # end if
        # end for
        return test_data
        
    @classmethod
    def write_sst_testcase(cls, num_seeds, num_trial, sst_testcases):
        _num_trial = '' if num_trial==1 else str(num_trial)
        cov_dir = Macros.result_dir / 'coverage' / 'txt_files'
        res_dir = cov_dir / f"ours_trial{_num_trial}_{num_seeds}seeds"
        res_dir.mkdir(parents=True, exist_ok=True)
        # sst_testcases = cls.read_sst_testcase(task,
        #                                       dataset_name,
        #                                       selection_method,
        #                                       num_seeds,
        #                                       num_trial)
        # checklist_testcases = cls.read_checklist_testcase(Macros.checklist_sa_dataset_file)
        
        cksum_map_text = ''
        for lc_desc in sst_testcases.keys():
            sst_sents = sst_testcases[lc_desc]['sents']
            sst_cksum = sst_testcases[lc_desc]['cksum']
            res_sst_text = ''
            # num_sents = 0
            # if len(sst_sents)>len(checklist_sents):
            #     num_sents = len(checklist_sents)
            #     sent_ids = list(range(num_sents))
            #     random.shuffle(sent_ids)
            #     sst_sents = [sst_sents[s_i] for s_i in sent_ids]
            # elif len(sst_sents)<len(checklist_sents):
            #     num_sents = len(sst_sents)
            #     sent_ids = list(range(num_sents))
            #     random.shuffle(sent_ids)
            #     checklist_sents = [checklist_sents[s_i] for s_i in sent_ids]
            # # end if
            cksum_map_text += f"{lc_desc}\t{sst_cksum}\n"
            for d_i in range(len(sst_sents)):
                res_sst_text += f"{sst_sents[d_i]}\n"
            # end for
            sst_save_file = res_dir / f"our_sents_{sst_cksum}.txt"
            Utils.write_txt(res_sst_text, sst_save_file)
        # end for
        Utils.write_txt(cksum_map_text, res_dir / 'cksum_map.txt')
        return

    @classmethod
    def write_checklist_testcase(cls, checklist_testcases):
        cov_dir = Macros.result_dir / 'coverage' / 'txt_files'
        res_dir = cov_dir / "checklist"
        res_dir.mkdir(parents=True, exist_ok=True)
        cksum_map_text = ''
        for lc_desc in checklist_testcases.keys():
            checklist_sents = checklist_testcases[lc_desc]['sents']
            checklist_cksum = checklist_testcases[lc_desc]['cksum']
            res_checklist_text = ''
            
            # num_sents = 0
            # if len(sst_sents)>len(checklist_sents):
            #     num_sents = len(checklist_sents)
            #     sent_ids = list(range(num_sents))
            #     random.shuffle(sent_ids)
            #     sst_sents = [sst_sents[s_i] for s_i in sent_ids]
            # elif len(sst_sents)<len(checklist_sents):
            #     num_sents = len(sst_sents)
            #     sent_ids = list(range(num_sents))
            #     random.shuffle(sent_ids)
            #     checklist_sents = [checklist_sents[s_i] for s_i in sent_ids]
            # # end if
            cksum_map_text += f"{lc_desc}\t{checklist_cksum}\n"
            for d_i in range(len(checklist_sents)):
                res_checklist_text += f"{checklist_sents[d_i]}\n"
            # end for
            checklist_save_file = res_dir / f"checklist_sents_{checklist_cksum}.txt"
            Utils.write_txt(res_checklist_text, checklist_save_file)
        # end for
        Utils.write_txt(cksum_map_text, res_dir / 'cksum_map.txt')
        return

    
def main_write(task,
               search_dataset_name,
               selection_method,
               num_seeds,
               all_num_trials):
    checklist_testcases = Coveragedata.read_checklist_testcase(Macros.checklist_sa_dataset_file)
    Coveragedata.write_checklist_testcase(checklist_testcases)
    for num_trial in range(1, all_num_trials+1):
        sst_testcases = Coveragedata.read_sst_testcase(task,
                                                       search_dataset_name,
                                                       selection_method,
                                                       num_seeds,
                                                       num_trial)
        Coveragedata.write_sst_testcase(num_seeds, num_trial, sst_testcases)
    # end for
    return
    
