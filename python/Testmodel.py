# This script is to test models
# using written testsuites

from typing import *
from pathlib import Path

from checklist.test_suite import TestSuite as suite

from Macros import Macros
from Utils import Utils
from Testsuite import Testsuite
from Model import Model


class Testmodel:

    model_func_map = {
        "sentiment_analysis": Model.sentiment_pred_and_conf
    }

    @classmethod
    def load_testsuite(cls, testsuite_file: Path):
        return suite().from_file(testsuite_file)

    @classmethod
    def run(cls, task):
        testsuite_file = Macros.result_dir / "test_results" / f'{task}_testsuite.pkl'
        testsuite = cls.load_testsuite(testsuite_file)
        for mname, model in Model.load_models(task):
            print(f">>>>> MODEL: {mname}")
            Model.run(testsuite, model, cls.model_func_map[task])
            print(f"<<<<< MODEL: {mname}")
        # end for
        return

        
if __name__=="__main__":
    for task in Macros.datasets.keys():
        print(f"***** TASK: {task} *****")
        Testmodel.run(task)
        print("**********")
        
