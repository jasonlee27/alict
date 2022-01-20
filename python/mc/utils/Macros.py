# This script is for defining all macros used in scripts in slpproject/python/hdlp/

from typing import *
from pathlib import Path

import os

class Macros:

    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__))) # nlptest/python/mc/utils
    root_dir: Path = this_dir.parent.parent.parent # nlptest/
    
    result_dir: Path = root_dir / "_results" # nlptest/_results
    download_dir: Path = root_dir / "_downloads" # nlptest/_downloads
    paper_dir = result_dir / "papers"
    dataset_dir = download_dir / "datasets"

    FMT_INT = "{:,d}"
    FMT_PER = "{:.1%}"
    FMT_FLOAT = "{:,.2f}"
    
    SEED = 27
    MASK = "{mask}"
    ADJ_MASK = "{a:mask}"

    mc_task = 'mc'
    
    # nlp_tasks = [sa_task, mc_task, mc_task]
    mc_label_map = {'different': 0, 'same': 1}

    # SQuAD: https://rajpurkar.github.io/SQuAD-explorer/
    datasets = ['squad', 'checklist']

    # Testsuite
    num_seeds = 20
    num_cfg_exp_elem = 20
    num_suggestions_on_exp_grammer_elem = 5

    num_synonyms_for_replace = 3 # number of synonyms used for replace_synosyms in Qgenerator
    
    nsamples = 500
    max_num_seeds = 100 # maximum number of selected sentences
    max_num_sents = 100000 # number of testcase sentences
    max_num_sents_for_perturb = 1000 # number of sentences for perturbation
    num_synonym_placeholders = 10
    max_num_synonyms = 5 # number of synonyms to be used when there are too many placeholders in one sentence

    # MC
    squad_dataset_dir: Path = dataset_dir / "squad"
    squad_train_file: Path = squad_dataset_dir / "train-v2.0.json"
    squad_valid_file: Path = squad_dataset_dir / "dev-v2.0.json"
    squad_test_file: Path = squad_dataset_dir / "test.tsv"
    
    # SentiWordNet
    swn_data_file: Path = download_dir / "SentiWordNet" / "data" / "SentiWordNet_3.0.0.txt"
    
    # Checklist Testsuite
    checklist_dir: Path = download_dir / "checklist"
    checklist_data_dir: Path = checklist_dir / "release_suites"
    checklist_mc_dataset_file: Path = checklist_data_dir / "squad_suite.pkl"
    
    mc_models_file = download_dir / "models" / "mc_models.txt"
    
    
    BASELINES = {
        "checklist": {
            "testsuite_file": checklist_mc_dataset_file
        }
    }

    # Retrain
    TRAIN_RATIO = 0.8
    retrain_output_dir: Path = result_dir / "retrain"
    retrain_model_dir: Path = retrain_output_dir / "models"
    retrain_dataset_dir: Path = retrain_output_dir / "datasets"
    checklist_sa_testcase_file: Path = retrain_dataset_dir / "checklist_mc_testcase.json"
    squad_mc_testcase_file: Path = retrain_dataset_dir / "squad_mc_testcase.json"
