# This script is to test models
# using written testsuites

from typing import *
from pathlib import Path

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
        suite = Testsuite()
        return suite.from_file(testsuite_file)

    @classmethod
    def run(cls, task):
        testsuite_file = Macros.result_dir / "test_results" / f'{task}_testsuite.pkl'
        testsuite = cls.load_testsuite(testsuite_file)
        for model in load_models(task):
            testsuite.run(cls.model_func_map[task], model=model)
            testsuite.summary()
        # end for
        return

        
if __name__=="__main__":
    for task in Macros.datasets.keys():
        Testmodel.run(task)
        
