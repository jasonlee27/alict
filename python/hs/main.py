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
                    choices=[
                        'requirement', 'template', 'testsuite', 'testmodel'
                    ], help='task to be run')
parser.add_argument('--nlp_task', type=str, default='hs',
                    choices=['hs'],
                    help='nlp task of focus')
parser.add_argument('--search_dataset', type=str, default='hatexplain',
                    help='name of dataset for searching testcases that meets the requirement')
parser.add_argument('--num_seeds', type=int, default=Macros.max_num_seeds,
                    help='number of seed inputs found in search dataset')
parser.add_argument('--syntax_selection', type=str, default='random',
                    choices=['prob', 'random', 'bertscore', 'noselect'],
                    help='method for selection of syntax suggestions')
parser.add_argument('--model_name', type=str, default=None,
                    help='name of model to be evaluated or retrained')


args = parser.parse_args()
def run_requirements():
    from .requirement.Requirements import Requirements
    nlp_task = args.nlp_task
    Requirements.convert_test_type_txt_to_json()
    Requirements.get_requirements(nlp_task)
    return

def run_templates():
    from .testsuite.Template import Template
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    num_seeds = args.num_seeds
    selection_method = args.syntax_selection
    log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "template_generation.log"
    Template.get_templates(
        num_seeds=num_seeds,
        nlp_task=nlp_task,
        dataset_name=search_dataset_name,
        selection_method=selection_method,
        log_file=log_file
    )
    return

def run_testsuites():
    from .testsuite.Testsuite import Testsuite
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    num_seeds = args.num_seeds
    selection_method = args.syntax_selection
    log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "testsuite_generation.log"
    Testsuite.write_testsuites(
        nlp_task=nlp_task,
        dataset=search_dataset_name,
        selection_method=selection_method,
        num_seeds=num_seeds,
        log_file=log_file
    )
    return

def run_testmodel():
    from .model.Testmodel import main as Testmodel_main
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    test_baseline = args.test_baseline
    # if test_baseline:
    #     selection_method = 'checklist'
    # # end if
    test_type = args.test_type
    local_model_name = args.local_model_name
    log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "test_orig_model.log"
    Testmodel_main(
        nlp_task,
        search_dataset_name,
        selection_method,
        test_baseline,
        test_type,
        log_file,
        local_model_name=local_model_name
    )
    return
    

func_map = {
    "sa": {
        'requirement': run_requirements,
        'template': run_templates,
        'testsuite': run_testsuites,
        'testmodel': run_testmodel
    }
}

if __name__=="__main__":
    func_map[args.nlp_task][args.run]()
