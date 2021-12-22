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
parser.add_argument('--run', type=str, required=True,
                    choices=['requirement', 'template', 'testsuite', 'testmodel', 'retrain'],
                    help='task to be run')
parser.add_argument('--nlp_task', type=str, default="sa",
                    choices=['sa', 'qqp', 'mc'],
                    help='nlp task of focus')
parser.add_argument('--search_dataset', type=str, default=None,
                    help='name of dataset for searching testcases that meets the requirement')
parser.add_argument('--num_seeds', type=int, default=10,
                    help='number of seed inputs found in search dataset')
parser.add_argument('--model_name', type=str, default=None,
                    help='name of model to be evaluated or retrained')
parser.add_argument('--num_labels', type=int, default=2,
                    help='number of labels for the model to be evaluated or retrained')

args = parser.parse_args()

def run_requirements():
    from .requirement.Requirements import Requirements
    Requirements.convert_test_type_txt_to_json()
    return

def run_templates():
    from .testsuite.Template import Template
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    num_seeds = args.num_seeds
    Template.get_templates(
        nlp_task=nlp_task,
        dataset=search_dataset_name,
        num_seeds=num_seeds
    )
    return

def run_testsuites():
    from .testsuite.Testsuite import Testsuite
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    num_seeds = args.num_seeds
    Testsuite.write_testsuites(
        nlp_task=nlp_task,
        dataset=search_dataset_name,
        num_seeds=num_seeds
    )
    return

def run_testmodel():
    from .model.Testmodel import Testmodel
    Testmodel.main()
    return

def run_retrain():
    from.retrain.Retrain import retrain
    if not os.path.exists(Macros.checklist_sa_testcase_file):
        from .retrain.Retrain import Retrain
        Retrain.get_checklist_testcase()
    # end if
    model_name = args.model_name
    num_labels = args.num_labels
    _, _ = retrain(
        model_name=model_name,
        num_labels=num_labels,
        dataset_file=Macros.checklist_sa_testcase_file
    )
    return
    

func_map = {
    "sa": {
        "requirement": run_requirements,
        "template": run_templates,
        "testsuite": run_testsuites,
        "testmodel": run_testmodel,
        "retrain": run_retrain
    }
}

if __name__=="__main__":
    func_map[args.nlp_task][args.run]()