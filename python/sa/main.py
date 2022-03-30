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
                        'requirement', 'template', 'testsuite',
                        'testmodel', 'retrain', 'analyze',
                        'retrain_analyze', 'explain_nlp', 'selfbleu',
                        'tables'
                    ], help='task to be run')
parser.add_argument('--nlp_task', type=str, default='sa',
                    choices=['sa'],
                    help='nlp task of focus')
parser.add_argument('--search_dataset', type=str, default='sst',
                    help='name of dataset for searching testcases that meets the requirement')
parser.add_argument('--num_seeds', type=int, default=Macros.max_num_seeds,
                    help='number of seed inputs found in search dataset')
parser.add_argument('--syntax_selection', type=str, default='random',
                    choices=['prob', 'random', 'bertscore', 'noselect'],
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
parser.add_argument('--lcs', action='store_true',
                    help='flag for indicating retraining model over each lc dataset')
parser.add_argument('--label_vec_len', type=int, default=2,
                    help='label vector length for the model to be evaluated or retrained')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs for retraining')
parser.add_argument('--testing_on_trainset', action='store_true',
                    help='flag for testing the model with train set')

# arguments for tables and plots
parser.add_argument('--which', type=str, default=None, nargs='+',
                    help='tables/plots that you are interested in making')

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

def run_retrain():
    from.retrain.Retrain import main_retrain
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    model_name = args.model_name
    label_vec_len = args.label_vec_len
    testing_on_trainset = args.testing_on_trainset
    retrain_by_lcs = args.lcs
    epochs = args.epochs
    main_retrain(nlp_task, 
                 search_dataset_name, 
                 selection_method, 
                 model_name, 
                 label_vec_len, 
                 retrain_by_lcs,
                 testing_on_trainset,
                 epochs,
                 Macros.log_dir)
    return

def run_analyze():
    from .model.Result import Result
    nlp_task = args.nlp_task
    selection_method = args.syntax_selection
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

def run_retrain_analyze():
    from .retrain.RetrainResult import RetrainResult
    nlp_task = args.nlp_task
    selection_method = args.syntax_selection
    search_dataset_name = args.search_dataset
    model_name = args.model_name
    RetrainResult.analyze(
        nlp_task,
        search_dataset_name,
        selection_method,
        model_name,
        is_retrained_by_lcs=True
    )
    return

def run_explainNLP():
    from .explainNLP.main import explain_nlp_main
    nlp_task = args.nlp_task
    selection_method = args.syntax_selection
    # selection_method = 'RANDOM'
    # if args.syntax_selection=='prob':
    #     selection_method = 'PROB'
    # # end if
    search_dataset_name = args.search_dataset
    model_name = args.model_name
    explain_nlp_main(
        nlp_task,
        search_dataset_name,
        selection_method,
        model_name
    )
    return

def run_selfbleu():
    from .utils.SelfBleu import main as selfbleu_main
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    model_name = args.model_name
    testcase_file = Macros.retrain_dataset_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}_testcase.json"
    eval_testcase_file = Macros.checklist_sa_testcase_file
    if not os.path.exists(str(testcase_file)):
        from .retrain.Retrain import Retrain
        Retrain.get_sst_testcase_for_retrain(nlp_task, selection_method)
    # end if
    if not os.path.exists(str(eval_testcase_file)):
        from .retrain.Retrain import Retrain
        Retrain.get_checklist_testcase_for_retrain(nlp_task)
    # end if
    selfbleu_main(nlp_task, search_dataset_name, selection_method)
    return

# ==========
# Tables & Plots

def run_make_tables():
    from .paper.Tables import Tables
    options = {
        'which': args.which,
        'task': args.nlp_task,
        'search_dataset_name': args.search_dataset,
        'selection_method': args.syntax_selection
    }
    Tables.make_tables(**options)
    return
    

func_map = {
    "sa": {
        'requirement': run_requirements,
        'template': run_templates,
        'testsuite': run_testsuites,
        'testmodel': run_testmodel,
        'retrain': run_retrain,
        'analyze': run_analyze,
        'retrain_analyze': run_retrain_analyze,
        'explain_nlp': run_explainNLP,
        'selfbleu': run_selfbleu,
        'tables': run_make_tables
    }
}

if __name__=="__main__":
    func_map[args.nlp_task][args.run]()
