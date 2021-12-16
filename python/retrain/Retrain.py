# This script is to re-test models using failed cases
# found from testsuite results

from typing import *
from pathlib import Path

from checklist.test_suite import TestSuite as suite

from ..utils.Macros import Macros
from ..utils.Utils import Utils
# from Testsuite import Testsuite
from ..model.Model import Model

import os


class ChecklistTestcases:

    @classmethod
    def write_checklist_testcase(cls, save_file):
        def example_to_dict_fn(data):
            return { "test_sent": data }
        tsuite = suite().from_file(Macros.checklist_sst_dataset_file)
        tsuite_dict = tsuite.to_dict(example_to_dict_fn=example_to_dict_fn)
        test_names = list(set(tsuite_dict['test_name']))
        test_data = dict()
        results = list()
        for test_name in test_names:
            test_data[test_name] = {
                "sents": tsuite.tests[test_name].data,
                "labels": tsuite.tests[test_name].labels
            }
            if type(test_data[test_name]['labels'])!=list:
                test_data[test_name]['labels'] = [test_data[test_name]['labels']]*len(test_data[test_name]['sents'])
            # end if
            # print(test_name, len(test_data[test_name]['sents']), len(test_data[test_name]['labels']))
        # end for
        
        for idx in range(len(tsuite_dict['test_sent'])):
            test_sent = tsuite_dict['test_sent'][idx]
            test_name = tsuite_dict['test_name'][idx]
            test_case = tsuite_dict['test_case'][idx]
            label = test_data[test_name]['labels'][
                test_data[test_name]['sents'].index(test_sent)
            ]
            results.append({
                "sent": test_sent,
                "label": label,
                "test_case": test_case,
                "test_name": test_name
            })
        # end for
        Utils.write_json(results, save_file, pretty_format=True)
        return results
    
    @classmethod
    def get_checklist_testcase(cls):
        if not os.path.exists(Macros.checklist_sst_testcase_file):
            Macros.checklist_result_dir.mkdir(parents=True, exist_ok=True)
            return cls.write_checklist_testcase(
                Macros.checklist_sst_testcase_file
            )
        # end if
        return Utils.read_json(
            Macros.checklist_sst_testcase_file
        )
    

class Retrain:

    @classmethod
    def get_checklist_testcase(cls):
        return ChecklistTestcases.get_checklist_testcase()
    
    @classmethod
    def get_failed_cases_from_test_results(cls):
        pass

    @classmethod
    def load_model(cls):
        pass

    @classmethod
    def train(cls):
        pass

    @classmethod
    def evaluate(cls):
        pass

    @classmethod
    def test(cls):
        pass
    


