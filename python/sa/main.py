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
                    choices=['requirement', 'template', 'testsuite', 'testmodel', 'retrain', 'analyze'],
                    help='task to be run')
parser.add_argument('--nlp_task', type=str, default="sa",
                    choices=['sa'],
                    help='nlp task of focus')
parser.add_argument('--search_dataset', type=str, default="sst",
                    help='name of dataset for searching testcases that meets the requirement')
parser.add_argument('--num_seeds', type=int, default=Macros.max_num_seeds,
                    help='number of seed inputs found in search dataset')
parser.add_argument('--syntax_selection', type=str, default='prob',
                    choices=['prob', 'random'],
                    help='method for selection of syntax suggestions')
parser.add_argument('--model_name', type=str, default=None,
                    help='name of model to be evaluated or retrained')

# arguments for testing model
parser.add_argument('--local_model_name', type=str, default=None,
                    help='name of retrained model to be evaluated')
parser.add_argument('--test_type', type=str, default="testsuite",
                    help='test dataset type (testsuite file or different dataset format)')
parser.add_argument('--test_baseline', action='store_true',
                    help='test models on running baseline (checklist) test cases')

# arguments for retraining
parser.add_argument('--label_vec_len', type=int, default=2,
                    help='label vector length for the model to be evaluated or retrained')

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
    is_random_select = False
    if args.syntax_selection=='random':
        is_random_select = True
    # end if
    Template.get_templates(
        num_seeds=num_seeds,
        nlp_task=nlp_task,
        dataset_name=search_dataset_name,
        is_random_select=is_random_select
    )
    return

def run_testsuites():
    from .testsuite.Testsuite import Testsuite
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    num_seeds = args.num_seeds
    is_random_select = False
    if args.syntax_selection=='random':
        is_random_select = True
    # end if
    Testsuite.write_testsuites(
        nlp_task=nlp_task,
        dataset=search_dataset_name,
        is_random_select=is_random_select,
        num_seeds=num_seeds
    )
    return

def run_testmodel():
    from .model.Testmodel import main as Testmodel_main
    nlp_task = args.nlp_task
    is_random_select = False
    if args.syntax_selection=='random':
        is_random_select = True
    # end if
    test_baseline = args.test_baseline
    test_type = args.test_type
    search_dataset_name = args.search_dataset
    local_model_name = args.local_model_name
    Testmodel_main(
        nlp_task,
        search_dataset_name,
        is_random_select,
        test_baseline,
        test_type,
        local_model_name=local_model_name
    )
    return

def run_retrain():
    from.retrain.Retrain import retrain
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    model_name = args.model_name
    label_vec_len = args.label_vec_len
    testcase_file = None
    if search_dataset_name==Macros.datasets[nlp_task][0]:
        testcase_file = Macros.sst_sa_testcase_file
        if not os.path.exists(str(testcase_file)):
            from .retrain.Retrain import Retrain
            Retrain.get_sst_testcase_for_retrain(nlp_task)
        # end if
    elif search_dataset_name==Macros.datasets[nlp_task][1]:
        testcase_file = Macros.checklist_sa_testcase_file
        if not os.path.exists(str(testcase_file)):
            from .retrain.Retrain import Retrain
            Retrain.get_checklist_testcase_for_retrain(nlp_task)
        # end if
    # end if
    _ = retrain(
        task=nlp_task,
        model_name=model_name,
        label_vec_len=label_vec_len,
        dataset_file=testcase_file,
        test_by_types=True
    )
    return

def run_analyze():
    from .model.Result import Result
    nlp_task = args.nlp_task
    selection_method = 'RANDOM'
    if args.syntax_selection=='random':
        selection_method = 'PROB'
    # end if
    search_dataset_name = args.search_dataset
    result_file = Macros.result_dir / f"test_results_{nlp_task}_{search_dataset_name}_{selection_method}" / 'test_results.txt'
    template_file = Macros.result_dir / f"cfg_expanded_inputs_{nlp_task}_{search_dataset_name}_{selection_method}.json"
    save_to = Macros.result_dir / f"test_results_{nlp_task}_{search_dataset_name}_{selection_method}" / 'test_result_analysis.json'
    Result.analyze(
        result_file,
        Macros.sa_models_file,
        template_file,
        save_to
    )
    return

func_map = {
    "sa": {
        "requirement": run_requirements,
        "template": run_templates,
        "testsuite": run_testsuites,
        "testmodel": run_testmodel,
        "retrain": run_retrain,
        "analyze": run_analyze
    }
}

if __name__=="__main__":
    func_map[args.nlp_task][args.run]()
