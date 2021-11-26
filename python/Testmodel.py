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
import sys


def argparse():
    arg_dict = dict()
    if len(sys.argv)==0:
        return
    else:
        arg_key_ids = list(range(len(sys.argv)))
        for kid in range(len(sys.argv)):
            key, val = sys.argv[kid].split("=")[0], sys.argv[kid].split("=")[1]
            arg_dict[key] = val
        # end for
        return arg_dict
    # end if


class Testmodel:

    model_func_map = {
        "sentiment_analysis": Model.sentiment_pred_and_conf
    }

    @classmethod
    def load_testsuite(cls, testsuite_file: Path):
        return suite().from_file(testsuite_file)

    @classmethod
    def _run(cls, task: str):
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
    
    @classmethod
    def _run_bl(cls, task, bl_name):
        print(f"***** TASK: {task} *****")
        print(f"***** Baseline: {bl_name} *****")
        testsuite = cls.load_testsuite(Macros.BASELINES[bs_name]["testsuite_file"])
        for mname, model in Model.load_models(task):
            print(f">>>>> MODEL: {mname}")
            Model.run(testsuite, model, cls.model_func_map[task])
            print(f"<<<<< MODEL: {mname}")
        # end for
        print("**********")
        print("**********")
        return

    @classmethod
    def run(cls, task):
        args = argparse()
        bl_name = None
        if "baseline" in args.keys():
            bl_name = args["baseline"]
        # end if
        for task in Macros.datasets.keys():
            if bl_name:
                cls._run_baseline(task, bl_name)
            else:
                cls._run(task)
            # end if
        # end for
        return
    
    
if __name__=="__main__":
    args = argparse()
    for task in Macros.datasets.keys():
        Testmodel.run(task, args)
    # end for
    
