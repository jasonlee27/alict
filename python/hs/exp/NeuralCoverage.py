# experiment for neural coverage

import re
import os
import math
import nltk
import time
import random


from ..seed.Search import Hatecheck
from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger


class NeuralCoverage:

    RES_DIR = Macros.result_dir / 'neural_coverage'
    
    @classmethod
    def read_our_testcases(cls,
                          task,
                          search_dataset_name,
                          selection_method):
        test_results_dir = Macros.result_dir / f"templates_{task}_{search_dataset_name}_{selection_method}"
        cksum_vals = [
            (l.strip().split('\t')[0], l.strip().split('\t')[1])
            for l in Utils.read_txt(test_results_dir / 'cksum_map.txt')
        ]
        test_data = dict()
        for lc_desc, cksum_val in cksum_vals:
            seed_file = test_results_dir / f"seeds_{cksum_val}.json"
            exp_file = test_results_dir / f"exps_{cksum_val}.json"
            if os.path.exists(seed_file) and \
               os.path.exists(exp_file):
                seeds = [s['input'] for s in Utils.read_json(seed_file)]
                exps = [e['input'] for e in Utils.read_json(exp_file)]
                test_data[lc_desc]= {
                    'sents': seeds+exps,
                    'cksum': cksum_val
                }
            # end if
        # end for
        return test_data
    
    @classmethod
    def write_our_testcases(cls,
                            task,
                            search_dataset_name,
                            selection_method,
                            testcases):
        # texts_all = list()
        res_dir = cls.RES_DIR / f"{task}_{search_dataset_name}_{selection_method}" / 'ours'
        res_dir.mkdir(parents=True, exist_ok=True)

        cksum_map_text = ''
        for lc_desc in testcases.keys():
            test_cases = list()
            sents = testcases[lc_desc]['sents']
            cksum = testcases[lc_desc]['cksum']
            # res_sst_text = ''
            cksum_map_text += f"{lc_desc}\t{cksum}\n"
            for d_i in range(len(sents)):
                # res_sst_text += f"{sst_sents[d_i]}\n"
                test_cases.append(sents[d_i])
            # end for
            res_file = res_dir / f"sents_{cksum}.json"
            Utils.write_json(test_cases, res_file, pretty_format=False)
        # end for
        Utils.write_txt(cksum_map_text, res_dir / 'cksum_map.txt')
        return

    @classmethod
    def read_hatecheck_testcases(cls,
                                 task,
                                 search_dataset_name,
                                 selection_method):
        seed_res_dir_name = f"templates_{task}_{search_dataset_name}_{selection_method}"
        seed_dir = Macros.result_dir / seed_res_dir_name
        cksums = Utils.read_txt(seed_dir / 'cksum_map.txt')
        test_data = dict()
        sents = Hatecheck.get_sents(
            Macros.hatecheck_data_file,
        )
        for l in cksums:
            lc, cksum_val = l.split('\t')[0].strip(), l.split('\t')[1].strip()
            func_name = [
                Hatecheck.FUNCTIONALITY_MAP[key]
                for key in Hatecheck.FUNCTIONALITY_MAP.keys()
                if key.split('::')[-1]==lc
            ][0]
            _sents = [s['sent'] for s in sents if s['func']==func_name or s['func'] in func_name]
            test_data[lc]= {
                'sents': _sents,
                'cksum': cksum_val
            }
        # end for
        return test_data

    @classmethod
    def write_hatecheck_testcases(cls,
                                  task,
                                  search_dataset_name,
                                  selection_method,
                                  testcases):
        # texts_all = list()
        res_dir = cls.RES_DIR / f"{task}_{search_dataset_name}_{selection_method}" / 'hatecheck'
        res_dir.mkdir(parents=True, exist_ok=True)
        
        cksum_map_text = ''
        for lc_desc in testcases.keys():
            test_cases = list()
            sents = testcases[lc_desc]['sents']
            cksum = testcases[lc_desc]['cksum']
            # res_sst_text = ''
            cksum_map_text += f"{lc_desc}\t{cksum}\n"
            for d_i in range(len(sents)):
                # res_sst_text += f"{sst_sents[d_i]}\n"
                test_cases.append(sents[d_i])
            # end for
            res_file = res_dir / f"sents_{cksum}.json"
            Utils.write_json(test_cases, res_file, pretty_format=False)
        # end for
        Utils.write_txt(cksum_map_text, res_dir / 'cksum_map.txt')
        return


def main(task,
         search_dataset_name,
         selection_method):
    # get all sentences into json file
    hatecheck_testcases = NeuralCoverage.read_hatecheck_testcases(task,
                                                                  search_dataset_name,
                                                                  selection_method)
    NeuralCoverage.write_hatecheck_testcases(task,
                                             search_dataset_name,
                                             selection_method,
                                             hatecheck_testcases)
    
    testcases = NeuralCoverage.read_our_testcases(task,
                                                  search_dataset_name,
                                                  selection_method)
    NeuralCoverage.write_our_testcases(task,
                                       search_dataset_name,
                                       selection_method,
                                       testcases)
    return
    
