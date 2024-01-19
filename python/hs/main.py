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
                        'requirement', 'template', 
                        'testsuite', 'testsuite_tosem',
                        'testmodel', 'testmodel_tosem', 
                        'analyze', 'analyze_tosem',
                        'failcase',
                        'selfbleu', 'selfbleu_mtnlp', 'selfbleu_hatecheck',
                        'pdrule_cov', 'pdrule_cov_mtnlp', 'pdrule_cov_hatecheck',
                        'humanstudy', 
                        'humanstudy_tosem',
                        'humanstudy_results',
                        'humanstudy_results_tosem',
                        'neural_coverage_data',
                        'tables', 'plots'
                    ], help='task to be run')
parser.add_argument('--nlp_task', type=str, default='hs',
                    choices=['hs'],
                    help='nlp task of focus')
parser.add_argument('--search_dataset', type=str, default='hatexplain',
                    help='name of dataset for searching testcases that meets the requirement')
parser.add_argument('--num_seeds', type=int, default=-1,
                    help='number of seed inputs found in search dataset')
parser.add_argument('--num_trials', type=int, default=1,
                    help='number of trials for the experiment')
parser.add_argument('--syntax_selection', type=str, default='random',
                    choices=['prob', 'random', 'bertscore', 'noselect'],
                    help='method for selection of syntax suggestions')
parser.add_argument('--gpu_ids', type=int, nargs="+", default=None,
                    help='method for selection of syntax suggestions')
parser.add_argument('--model_name', type=str, default=None,
                    help='name of model to be evaluated or retrained')
parser.add_argument('--test_baseline', action='store_true',
                    help='test models on running baseline (hatecheck) test cases')

# arguments for tables and plots
parser.add_argument('--which', type=str, default=None, nargs='+',
                    help='tables/plots that you are interested in making')

args = parser.parse_args()
rand_seed_num = Macros.RAND_SEED[args.num_trials] if args.num_trials>0 else Macros.RAND_SEED[1]
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
    num_trials = '' if args.num_trials<2 else str(args.num_trials)
    _num_trials = '' if args.num_trials<2 else str(num_trials)
    gpu_ids = args.gpu_ids
    if gpu_ids is not None:
        assert len(gpu_ids)==Macros.num_processes
    # end if
    
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
        gpu_ids=gpu_ids,
        log_file=log_file
    )
    return

def run_testsuites():
    from .testsuite.Testsuite import Testsuite
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    num_seeds = args.num_seeds
    num_trials = '' if args.num_trials<2 else str(args.num_trials)
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

def run_testsuites_tosem():
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
    print(log_file)
    Testsuite.write_testsuites_tosem(
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
    num_trials = args.num_trials # '' if args.num_trials==1 else str(args.num_trials)
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

def run_testmodel_tosem():
    from .model.Testmodel import main_tosem as Testmodel_main_tosem
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    num_seeds = args.num_seeds
    num_trials = args.num_trials
    test_baseline = args.test_baseline
    # if test_baseline:
    #     selection_method = 'checklist'
    # # end if
    local_model_name = Macros.openai_chatgpt_engine_name
    if num_seeds<0:
        log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}"
    else:
        log_dir = Macros.log_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds"
    # end if
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'test_orig_model_tosem.log'
    Testmodel_main_tosem(
        nlp_task,
        search_dataset_name,
        selection_method,
        num_seeds,
        num_trials,
        test_baseline,
        log_file,
        local_model_name=local_model_name
    )
    return

def run_analyze():
    from .model.Result import Result
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    test_baseline = args.test_baseline
    num_seeds = args.num_seeds
    num_trials = '' if args.num_trials<2 else str(args.num_trials)
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
            template_result_dir = Macros.result_dir / f"templates{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}"
            result_dir = Macros.result_dir / f"test_results{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}"
        else:
            template_result_dir = Macros.result_dir / f"templates{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds"
            result_dir = Macros.result_dir / f"test_results{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds"
        # end if
        # result_file = result_dir / 'test_results.txt'
        save_to = result_dir / 'test_result_analysis.json'
        Result.analyze(
            nlp_task,
            template_result_dir,
            result_dir,
            Macros.hs_models_file,
            save_to
        )
    # end if
    return

def run_analyze_tosem():
    from .model.Result import Result
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    test_baseline = args.test_baseline
    num_seeds = args.num_seeds
    num_trials = '' if args.num_trials==1 else str(args.num_trials)
    tosem_model_names = [
        Macros.openai_chatgpt_engine_name,
        Macros.openai_chatgpt4_engine_name
    ]
    if test_baseline:
        if num_seeds<0:
            result_dir = Macros.result_dir / f"test_results{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}"
        else:
            result_dir = Macros.result_dir / f"test_results{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds"
        # end if
        result_file = result_dir / 'test_results_tosem_hatecheck.txt'
        save_to = result_dir / 'test_result_tosem_hatecheck_analysis.json'
        Result.analyze_hatecheck_tosem(
            result_file,
            tosem_model_names,
            save_to
        )
    else:
        if num_seeds<0:
            template_result_dir = Macros.result_dir / f"templates{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}"
            result_dir = Macros.result_dir / f"test_results{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}"
        else:
            template_result_dir = Macros.result_dir / f"templates{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds"
            result_dir = Macros.result_dir / f"test_results{num_trials}_{nlp_task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds"
        # end if
        # result_file = result_dir / 'test_results.txt'
        save_to = result_dir / 'test_result_tosem_analysis.json'
        Result.analyze_tosem(
            nlp_task,
            template_result_dir,
            result_dir,
            tosem_model_names,
            save_to
       )
    # end if
    return

# ==========
# Exp

def run_failcase():
    from .exp.Failcase import main_fail, main_p2f_f2p
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    main_fail(nlp_task,
              search_dataset_name,
              selection_method)
    main_p2f_f2p(nlp_task,
                 search_dataset_name,
                 selection_method)
    return

def run_selfbleu():
    print('Run run_selfbleu..')
    from .exp.SelfBleu import main_sample
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    main_sample(nlp_task,
                search_dataset_name,
                selection_method)
    return

def run_selfbleu_mtnlp():
    print('Run run_selfbleu_mtnlp..')
    from .exp.SelfBleu import main_mtnlp
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    main_mtnlp(nlp_task,
               search_dataset_name,
               selection_method)
    return

def run_selfbleu_hatecheck():
    print('Run run_selfbleu_hatecheck..')
    from .exp.SelfBleu import main_hatecheck
    nlp_task = args.nlp_task
    selection_method = args.syntax_selection
    main_hatecheck(nlp_task,
                   selection_method)
    return

def run_pdrule_cov():
    from .exp.ProductionruleCoverage import main_sample, main_all
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    main_sample(nlp_task,
                search_dataset_name,
                selection_method)
    # main_all(nlp_task,
    #          search_dataset_name,
    #          selection_method)
    return


def run_pdrule_cov_hatecheck():
    from .exp.ProductionruleCoverage import main_hatecheck
    nlp_task = args.nlp_task
    selection_method = args.syntax_selection
    main_hatecheck(nlp_task,
                   selection_method)
    return

def run_pdrule_cov_mtnlp():
    from .exp.ProductionruleCoverage import main_mtnlp
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    main_mtnlp(nlp_task,
               search_dataset_name,
               selection_method)
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

def run_humanstudy_tosem():
    from .exp.Humanstudy import Humanstudy
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    Humanstudy.main_sample_tosem(
        nlp_task,
        search_dataset_name,
        selection_method
    )
    return

def run_humanstudy_result_tosem():
    from .exp.Humanstudy import Humanstudy
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    print("Run run_humanstudy_result_tosem..")
    Humanstudy.main_result_tosem(
        nlp_task,
        search_dataset_name,
        selection_method
    )
    return

def run_humanstudy_result():
    from .exp.Humanstudy import Humanstudy
    nlp_task = args.nlp_task
    search_dataset_name = args.search_dataset
    selection_method = args.syntax_selection
    model_name = args.model_name
    num_samples = 20
    print('Run run_humanstudy_result..')
    Humanstudy.main_result(nlp_task,
                           search_dataset_name,
                           selection_method,
                           model_name,
                           num_samples)
    return


def run_neural_coverage_data():
    # from .coverage.extract_data import Coveragedata
    from .exp.NeuralCoverage import main
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
    print("Run run_neural_coverage_data..")
    main(nlp_task,
         search_dataset_name,
         selection_method)
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
        'model_name': args.model_name,
    }
    Plots.make_plots(**options)
    return


func_map = {
    "hs": {
        'requirement': run_requirements,
        'template': run_templates,
        'testsuite': run_testsuites,
        'testsuite_tosem': run_testsuites_tosem,
        'testmodel': run_testmodel,
        'testmodel_tosem': run_testmodel_tosem,
        'analyze': run_analyze,
        'analyze_tosem': run_analyze_tosem,
        'failcase': run_failcase,
        'selfbleu': run_selfbleu,
        'selfbleu_mtnlp': run_selfbleu_mtnlp,
        'selfbleu_hatecheck': run_selfbleu_hatecheck,
        'pdrule_cov': run_pdrule_cov,
        'pdrule_cov_mtnlp': run_pdrule_cov_mtnlp,
        'pdrule_cov_hatecheck': run_pdrule_cov_hatecheck,
        'humanstudy': run_humanstudy,
        'humanstudy_tosem': run_humanstudy_tosem,
        'humanstudy_results_tosem': run_humanstudy_result_tosem,
        'humanstudy_results': run_humanstudy_result,
        'neural_coverage_data': run_neural_coverage_data,
        'tables': run_make_tables,
        'plots': run_make_plots
    }
}

if __name__=="__main__":
    func_map[args.nlp_task][args.run]()
