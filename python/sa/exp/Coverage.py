
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

random.seed(27)

class Coverage:

    @classmethod
    def write_sst_testcase(cls, task, dataset_name, selection_method, save_file=None):
        test_results_dir = Macros.result_dir / f"test_results_{task}_{dataset_name}_{selection_method}"
        cksum_vals = [
            os.path.basename(testsuite_file).split("_")[-1].split(".")[0]
            for testsuite_file in os.listdir(test_results_dir)
            if testsuite_file.startswith(f"{task}_testsuite_seeds_") and testsuite_file.endswith(".pkl")
        ]
        
        test_data = dict()
        for cksum_val in cksum_vals:
            testsuite_files = [
                f"{task}_testsuite_seeds_{cksum_val}.pkl",
                f"{task}_testsuite_exps_{cksum_val}.pkl"
            ]
            test_name = ''
            testsuite_files = [tf for tf in testsuite_files if os.path.exists(test_results_dir / tf)]
            for f_i, testsuite_file in enumerate(testsuite_files):
                tsuite, tsuite_dict = Utils.read_testsuite(test_results_dir / testsuite_file)
                for tn in list(set(tsuite_dict['test_name'])):
                    test_name = tn.split('::')[-1]
                    # if tsuite.tests[tn].labels is not None and test_name not in cls.LC_NOT_INCLUDED_LIST:
                    if test_name in Macros.OUR_LC_LIST:
                        target_test_name = test_name
                        texts, labels, _types = list(), list(), list()
                        if test_name in Macros.OUR_LC_LIST[9:]:
                            target_test_name = str(Macros.OUR_LC_LIST[9:])
                            if target_test_name not in test_data.keys():
                                test_data[target_test_name] = {
                                    'sents': tsuite.tests[tn].data,
                                    # 'labels': tsuite.tests[tn].labels
                                }
                            else:
                                test_data[target_test_name]['sents'].extend(tsuite.tests[tn].data)
                                # test_data[test_name]['labels'].extend(tsuite.tests[tn].labels)
                            # end if                            
                        else:
                            if test_name not in test_data.keys():
                                test_data[test_name] = {
                                    'sents': tsuite.tests[tn].data,
                                    # 'labels': tsuite.tests[tn].labels
                                }
                            else:
                                test_data[test_name]['sents'].extend(tsuite.tests[tn].data)
                                # test_data[test_name]['labels'].extend(tsuite.tests[tn].labels)
                            # end if
                        # end if
                    # end if
                # end for
            # end for
            # if test_name in test_data.keys():
            #     # set data labels in a range between 0. and 1. from 0,1,2
            #     test_data[test_name]['labels'] = [0.5*float(l) for l in test_data[test_name]['labels']]
            # # end if
        # end for

        # if save_file is not None:
        #     res_text = ''
        #     for test_name in test_data.keys():
        #         test_sents = test_data[test_name]['sents']
        #         # labels = test_data[test_name]['labels']
        #         for d_i in range(len(test_sents)):
        #             res_text += f"{test_sents[d_i]}\n"
        #         # end for
        #     # end for
        #     Utils.write_txt(res_text, save_file)
        # # end if
        return test_data

    @classmethod
    def write_testcase_to_txt(cls,
                              task,
                              dataset_name,
                              selection_method):
        tsuite, tsuite_dict = Utils.read_testsuite(Macros.checklist_sa_dataset_file)
        sst_testcases = cls.write_sst_testcase(task, dataset_name, selection_method)
        test_names = list(set(tsuite_dict['test_name']))
        test_data = dict()
        num_data = 0
        for test_name in test_names:
            if test_name in Macros.CHECKLIST_LC_LIST:
                if test_name in Macros.CHECKLIST_LC_LIST[8:10]:
                    target_test_name = str(Macros.CHECKLIST_LC_LIST[8:10])
                    if target_test_name not in test_data.keys():
                        test_data[target_test_name] = {
                            'sents': tsuite.tests[test_name].data,
                            # 'labels': tsuite.tests[test_name].labels
                        }
                    else:
                        test_data[target_test_name]['sents'].extend(tsuite.tests[test_name].data)
                    # end if
                else:
                    sents = tsuite.tests[test_name].data
                    test_data[test_name] = {
                        'sents': tsuite.tests[test_name].data,
                        # 'labels': tsuite.tests[test_name].labels
                    }
                # end if
            # end if
        # end for
        res_sst_text = ''
        res_checklist_text = ''
        sst_testcases = cls.write_sst_testcase(task, dataset_name, selection_method)
        for test_name in test_data.keys():
            sst_sents = sst_testcases[Macros.LC_MAP[test_name]]['sents']
            checklist_sents = test_data[test_name]['sents']
            num_sents = 0
            if len(sst_sents)>len(checklist_sents):
                num_sents = len(checklist_sents)
                sent_ids = list(range(num_sents))
                random.shuffle(sent_ids)
                sst_sents = [sst_sents[s_i] for s_i in sent_ids]
            elif len(sst_sents)<len(checklist_sents):
                num_sents = len(sst_sents)
                sent_ids = list(range(num_sents))
                random.shuffle(sent_ids)
                checklist_sents = [checklist_sents[s_i] for s_i in sent_ids]
            # end if
            for d_i in range(len(sst_sents)):
                res_sst_text += f"{sst_sents[d_i]}\n"
            # end for
            for d_i in range(len(checklist_sents)):
                res_checklist_text += f"{checklist_sents[d_i]}\n"
            # end for
        # end for
        save_dir = Macros.result_dir / 'coverage' / 'txt_files'
        save_dir.mkdir(parents=True, exist_ok=True)
        sst_save_file = save_dir / 'our_sents.txt'
        checklist_save_file = save_dir / 'checklist_sents.txt'
        Utils.write_txt(res_sst_text, sst_save_file)
        Utils.write_txt(res_checklist_text, checklist_save_file)
        return
    
