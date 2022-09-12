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
                        'requirement', 'template', 'testsuite', 'testmodel', 'analyze'
                    ], help='task to be run')
parser.add_argument('--nlp_task', type=str, default='hs',
                    choices=['hs'],
                    help='nlp task of focus')
parser.add_argument('--search_dataset', type=str, default='hatexplain',
                    help='name of dataset for searching testcases that meets the requirement')
parser.add_argument('--num_seeds', type=int, default=Macros.max_num_seeds,
                    help='number of seed inputs found in search dataset')
parser.add_argument('--num_trials', type=int, default=1,
                    help='number of trials for the experiment')
parser.add_argument('--syntax_selection', type=str, default='random',
                    choices=['prob', 'random', 'bertscore', 'noselect'],
                    help='method for selection of syntax suggestions')
parser.add_argument('--model_name', type=str, default=None,
                    help='name of model to be evaluated or retrained')
parser.add_argument('--test_baseline', action='store_true',
                    help='test models on running baseline (hatecheck) test cases')

args = parser.parse_args()
rand_seed_num = Macros.RAND_SEED[args.num_trials]
random.seed(rand_seed_num)

def run_requirements():
    from .requirement.Requirements import Requirements
    nlp_task = args.nlp_task
    Requirements.convert_test_type_txt_to_json()
    Requirements.get_requirements(nlp_task)
    return

def run_templates():
    from .testsuite.Template import Template
    from torch.multiprocessing import Pool, Process, set_start_method
    set_start_method('spawn')
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    num_seeds = args.num_seeds
    num_trials = '' if args.num_trials==1 else str(args.num_trials)
    _num_trials = '' if num_trials==1 else str(num_trials)
    if num_seeds<0:
        log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}"
    else:
        log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds"
    # end if
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"template{num_trials}_generation.log"
    Template.get_templates(
        nlp_task=nlp_task,
        dataset_name=search_dataset_name,
        selection_method=selection_method,
        num_seeds=num_seeds,
        num_trials=num_trials,
        log_file=log_file
    )
    return

def run_testsuites():
    from .testsuite.Testsuite import Testsuite
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    num_seeds = args.num_seeds
    num_trials = '' if args.num_trials==1 else str(args.num_trials)
    if num_seeds<0:
        log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}"
    else:
        log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds"
    # end if
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"testsuite{num_trials}_generation.log"
    Testsuite.write_testsuites(
        nlp_task=nlp_task,
        dataset=search_dataset_name,
        selection_method=selection_method,
        num_seeds=num_seeds,
        num_trials=num_trials,
        log_file=log_file
    )
    return

def run_testmodel():
    from .model.Testmodel import main as Testmodel_main
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    num_seeds = args.num_seeds
    test_baseline = args.test_baseline
    num_trials = '' if args.num_trials==1 else str(args.num_trials)
    if num_seeds<0:
        log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}"
    else:
        log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds"
    # end if
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "test_orig_model.log"
    Testmodel_main(
        nlp_task,
        search_dataset_name,
        selection_method,
        num_seeds,
        num_trials,
        test_baseline,
        log_file
    )
    return

def run_analyze():
    from .model.Result import Result
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    test_baseline = args.test_baseline
    num_seeds = args.num_seeds
    num_trials = '' if args.num_trials==1 else str(args.num_trials)
    if test_baseline:
        if num_seeds<0:
            result_dir = Macros.result_dir / f"test_results{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}"
        else:
            result_dir = Macros.result_dir / f"test_results{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds"
        # end if
        result_file = result_dir / 'test_results_hatecheck.txt'
        save_to = result_dir / 'test_result_hatecheck_analysis.json'
        Result.analyze_hatecheck(
            result_file,
            Macros.hs_models_file,
            save_to
        )
    else:
        if num_seeds<0:
            result_dir = Macros.result_dir / f"test_results{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}"
            template_file = Macros.result_dir / f"cfg_expanded_inputs{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}.json"
        else:
            result_dir = Macros.result_dir / f"test_results{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds"
            template_file = Macros.result_dir / f"cfg_expanded_inputs{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds.json"
        # end if
        result_file = result_dir / 'test_results.txt'
        save_to = result_dir / 'test_result_analysis.json'
        Result.analyze(
            result_file,
            Macros.hs_models_file,
            template_file,
            save_to
        )
    # end if
    return


func_map = {
    "hs": {
        'requirement': run_requirements,
        'template': run_templates,
        'testsuite': run_testsuites,
        'testmodel': run_testmodel,
        'analyze': run_analyze
    }
}

if __name__=="__main__":
    func_map[args.nlp_task][args.run]()
