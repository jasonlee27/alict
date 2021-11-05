# This script is to test models
# using written testsuites

from typing import *
from pathlib import Path

from checklist.test_suite import TestSuite as suite

from Macros import Macros
from Utils import Utils
from Testsuite import Testsuite
from Model import Model

import os

class Testmodel:

    model_func_map = {
        "sentiment_analysis": Model.sentiment_pred_and_conf
    }

    @classmethod
    def load_testsuite(cls, testsuite_file: Path):
        return suite().from_file(testsuite_file)

    @classmethod
    def run(cls, task):
        for test_file in os.listdir(Macros.result_dir / "test_results"):
            if test_file.startswith(f"{task}_testsuite_") and test_file.endswith(".pkl"):
                testsuite_file = Macros.result_dir / "test_results" / test_file
                testsuite = cls.load_testsuite(testsuite_file)
                test_info = testsuite.info[task]["capability"]+"::"+testsuite.info[task]["description"]
                print(f">>>>> TEST: {test_info}")
                for mname, model in Model.load_models(task):
                    print(f">>>>> MODEL: {mname}")
                    Model.run(testsuite, model, cls.model_func_map[task])
                    print(f"<<<<< MODEL: {mname}")
                # end for
                print(f"<<<<< TEST")
            # end if
        # end for
        return

        
if __name__=="__main__":
    for task in Macros.datasets.keys():
        print(f"***** TASK: {task} *****")
        Testmodel.run(task)
        print("**********")
        
