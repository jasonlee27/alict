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
        print(f"***** TASK: {task} *****")
        cksum_vals = [
            os.path.basename(test_file).split("_")[-1].split(".")[0]
            for test_file in os.listdir(Macros.result_dir / "test_results")
            if test_file.startswith(f"{task}_testsuite_seeds_") and test_file.endswith(".pkl")
        ]

        for cksum_val in cksum_vals:
            test_files = [
                f"{task}_testsuite_seeds_{cksum_val}.pkl",
                f"{task}_testsuite_seed_templates_{cksum_val}.pkl",
                f"{task}_testsuite_exp_templates_{cksum_val}.pkl"
            ]
            for test_file in test_files:
                print(test_file)
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
            # end for
        # end for
        print("**********")
        return

        
if __name__=="__main__":
    for task in Macros.datasets.keys():
        Testmodel.run(task)
    # end for
        
