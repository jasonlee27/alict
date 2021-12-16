from typing import *

import re, os
import sys
import json
import random
import argparse

from pathlib import Path

from .utils.Macros import Macros
from .utils.Utils import Utils

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--task', type=str, nargs='+',
                    help='task to be run')

args = parser.parse_args()

def run_requirements():
    from .requirement.Requirements import Requirements
    Requirements.convert_test_type_txt_to_json()
    return

def run_templates():
    from .testsuite.Template import Template
    Template.get_templates(num_seeds=10)
    return

def run_testsuites():
    from .testsuite.Testsuite import Testsuite
    Testsuite.write_testsuites()
    return

def run_testmodel():
    from .model.Testmodel import Testmodel
    Testmodel.main()
    return

def run_retrain():
    from .retrain.Retrain import Retrain
    Retrain.get_checklist_testcase()
    

func_map = {
    "requirement": run_requirements,
    "template": run_templates,
    "testsuite": run_testsuites,
    "testmodel": run_testmodel,
    "retrain": run_retrain
}

if __name__=="__main__":
    task_name = args.task[0]
    func_map[task_name]()
