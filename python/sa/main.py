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
                        'requirement', 'template', 'testsuite', 'seedgen',
                        'testmodel', 'testmodel_seed', 'retrain', 'analyze',
                        'analyze_seed', 'retrain_analyze', 'explain_nlp', 'selfbleu',
                        'pdrule_cov', 'humanstudy', 'humanstudy_results', 'coverage_data',
                        'tables', 'plots'
                    ], help='task to be run')
parser.add_argument('--nlp_task', type=str, default='sa',
                    choices=['sa'],
                    help='nlp task of focus')
parser.add_argument('--search_dataset', type=str, default='sst',
                    help='name of dataset for searching testcases that meets the requirement')
parser.add_argument('--num_seeds', type=int, default=-1,
                    help='number of seed inputs found in search dataset. It uses all seeds if negative value')
parser.add_argument('--num_trials', type=int, default=1,
                    help='number of trials for the experiment')
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
    num_seeds = args.num_seeds
    num_trials = '' if args.num_trials==1 else str(args.num_trials)
    selection_method = args.syntax_selection
    _num_trials = '' if num_trials==1 else str(num_trials)
    if num_seeds<0:
        log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}"
    else:
        log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds"
    # end if
    log_dir.mkdir(parents=True, exist_ok=True)
    # log_file = log_dir / "template2_generation.log"
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

def run_seedgen():
    from .testsuite.Seedgen import Seedgen
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    num_seeds = args.num_seeds # -1 means acceptance of every seeds
    num_trials = args.num_trials
    # selection_method = args.syntax_selection
    if num_seeds<0:
        log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}"
    else:
        log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}_{num_seeds}seeds"
    # end if
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "seed_generation.log"
    Seedgen.get_seeds(
        nlp_task=nlp_task,
        dataset_name=search_dataset_name,
        num_seeds=num_seeds,
        log_file=log_file
    )
    return

def run_testsuites():
    from .testsuite.Testsuite import Testsuite
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    num_seeds = args.num_seeds
    num_trials = '' if args.num_trials==1 else str(args.num_trials)
    selection_method = args.syntax_selection
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
    num_trials = args.num_trials
    test_baseline = args.test_baseline
    # if test_baseline:
    #     selection_method = 'checklist'
    # # end if
    test_type = args.test_type
    local_model_name = args.local_model_name
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
        test_type,
        log_file,
        local_model_name=local_model_name
    )
    return

def run_seed_testmodel():
    from .model.Testmodel import main_seed
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    num_seeds = args.num_seeds
    num_trials = args.num_trials
    test_baseline = args.test_baseline
    test_type = args.test_type
    local_model_name = args.local_model_name
    if num_seeds<0:
        log_dir = Macros.log_dir / f"seeds_{nlp_task}_{search_dataset_name}"
    else:
        log_dir = Macros.log_dir / f"seeds_{nlp_task}_{search_dataset_name}_{num_seeds}seeds"
    # end if
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "test_orig_model.log"
    main_seed(
        nlp_task,
        search_dataset_name,
        selection_method,
        num_seeds,
        num_trials,
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
        result_file = result_dir / 'test_results_checklist.txt'
        save_to = result_dir / 'test_result_checklist_analysis.json'
        Result.analyze_checklist(
            result_file,
            Macros.sa_models_file,
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
            Macros.sa_models_file,
            template_file,
            save_to
        )
    # end if
    return

def run_analyze_seed():
    from .model.Result import Result
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    test_baseline = args.test_baseline
    num_seeds = args.num_seeds
    num_trials = '' if args.num_trials==1 else str(args.num_trials)
    if num_seeds<0:
        res_dir = Macros.result_dir / f"seeds{num_trials}_{nlp_task}_{search_dataset_name}"
    else:
        res_dir = Macros.result_dir / f"seeds{num_trials}_{nlp_task}_{search_dataset_name}_{num_seeds}seeds"
    # end if
    result_file = res_dir / 'test_results.txt'
    bl_result_file = res_dir / 'test_results_checklist.txt'
    save_to = res_dir / 'test_result_analysis.json'
    Result.analyze_seed_performance(
        result_file,
        bl_result_file,
        model_name_file=Macros.sa_models_file,
        saveto=save_to
    )
    return

def run_retrain_analyze():
    from .retrain.RetrainResult import RetrainResult
    nlp_task = args.nlp_task
    selection_method = args.syntax_selection
    search_dataset_name = args.search_dataset
    model_name = args.model_name
    epochs = args.epochs
    RetrainResult.analyze(
        nlp_task,
        search_dataset_name,
        selection_method,
        model_name,
        epochs,
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
    from .exp.SelfBleu import main_seed
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    num_seeds = args.num_seeds
    num_trials = args.num_trials
    main_seed(nlp_task,
              search_dataset_name,
              selection_method,
              num_seeds,
              num_trials)
    return

def run_pdrule_cov():
    from .exp.ProductionruleCoverage import main_seed, main_seed_sample
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    num_seeds = args.num_seeds
    num_trials = args.num_trials
    main_seed(nlp_task,
              search_dataset_name,
              selection_method,
              num_seeds,
              num_trials)
    main_seed_sample(nlp_task,
                     search_dataset_name,
                     selection_method,
                     num_seeds,
                     num_trials)
    return

# ==========
# Human study

def run_humanstudy():
    from .exp.Humanstudy import Humanstudy
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    Humanstudy.main_sample(nlp_task,
                           search_dataset_name,
                           selection_method)
    return

def run_humanstudy_result():
    from .exp.Humanstudy import Humanstudy
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    model_name = args.model_name
    num_samples = 20
    Humanstudy.main_result(nlp_task,
                           search_dataset_name,
                           selection_method,
                           model_name,
                           num_samples)
    return

# ==========
# Coverage Exp

def run_coverage_data():
    # from .coverage.extract_data import Coveragedata
    from .exp.Coveragedata import main_write
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    num_seeds = args.num_seeds
    num_trials = args.num_trials
    # Coveragedata.write_target_seed_sents(nlp_task,
    #                                      search_dataset_name)
    # # Coveragedata.write_target_exp_sents(nlp_task,
    # #                                     search_dataest_name,
    # #                                     selection_method)
    main_write(nlp_task,
               search_dataset_name,
               selection_method,
               num_seeds,
               num_trials)
    return

# ==========
# Tables & Plots

def run_make_tables():
    from .paper.Tables import Tables
    options = {
        'which': args.which,
        'task': args.nlp_task,
        'search_dataset_name': args.search_dataset,
        'selection_method': args.syntax_selection,
        'epochs': args.epochs,
        'model_name': args.model_name,
        'num_seeds': args.num_seeds,
        'num_trials': args.num_trials,
    }
    Tables.make_tables(**options)
    return

def run_make_plots():
    from .paper.Plots import Plots
    options = {
        'which': args.which,
        'task': args.nlp_task,
        'search_dataset_name': args.search_dataset,
        'selection_method': args.syntax_selection,
        'epochs': args.epochs,
        'model_name': args.model_name,
        'num_seeds': args.num_seeds,
        'num_trials': args.num_trials,
    }
    Plots.make_plots(**options)
    return
    

func_map = {
    "sa": {
        'requirement': run_requirements,
        'template': run_templates,
        'seedgen': run_seedgen,
        'testsuite': run_testsuites,
        'testmodel': run_testmodel,
        'testmodel_seed': run_seed_testmodel,
        'retrain': run_retrain,
        'analyze': run_analyze,
        'analyze_seed': run_analyze_seed,
        'retrain_analyze': run_retrain_analyze,
        'explain_nlp': run_explainNLP,
        'selfbleu': run_selfbleu,
        'pdrule_cov': run_pdrule_cov,
        'humanstudy': run_humanstudy,
        'humanstudy_results': run_humanstudy_result,
        'coverage_data': run_coverage_data,
        'tables': run_make_tables,
        'plots': run_make_plots
    }
}

if __name__=="__main__":
    func_map[args.nlp_task][args.run]()
