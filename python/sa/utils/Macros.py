# This script is for defining all macros used in scripts in slpproject/python/hdlp/

from typing import *
from pathlib import Path

import os

class Macros:

    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__))) # nlptest/python/utils
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

    ## NLP dataset
    sa_task = 'sa'
    mc_task = 'mc'
    qqp_task = 'qqp'
    nlp_tasks = [sa_task, mc_task, qqp_task]
    sa_label_map = {'negative': 0, 'positive': 2, 'neutral': 1}

    # stanfordSentimentTreebank:
    # dynasent paper: https://arxiv.org/pdf/2012.15349.pdf
    datasets = {
        sa_task: ['sst', 'checklist', 'airlinetweets'],
        mc_task: [],
        qqp_task: []
    }

    # Testsuite
    num_seeds = 20
    num_cfg_exp_elem = 20
    num_suggestions_on_exp_grammer_elem = 5
    nsamples = 500
    max_num_seeds = 100 # maximum number of selected sentences
    max_num_sents = 100000 # number of testcase sentences
    max_num_sents_for_perturb = 1000 # number of sentences for perturbation
    num_synonym_placeholders = 10
    max_num_synonyms = 2 # number of synonyms to be used when there are too many placeholders in one sentence

    # SST
    sst_datasent_file: Path = dataset_dir / "stanfordSentimentTreebank" / "datasetSentences.txt"
    sst_dict_file: Path = dataset_dir / "stanfordSentimentTreebank" / "dictionary.txt"
    sst_label_file: Path = dataset_dir / "stanfordSentimentTreebank" / "sentiment_labels.txt"
    # Tweets
    tweet_file: Path = dataset_dir / "airplanetweets" / "Tweets.csv"

    # Dynasent
    dyna_r1_test_src_file: Path = dataset_dir / "dynasent" / "dynasent-v1.1" / "dynasent-v1.1-round01-yelp-test.jsonl"

    # SentiWordNet
    swn_data_file: Path = download_dir / "SentiWordNet" / "data" / "SentiWordNet_3.0.0.txt"
    
    # Checklist Testsuite
    checklist_dir: Path = download_dir / "checklist"
    checklist_data_dir: Path = checklist_dir / "release_suites"
    checklist_sa_dataset_file: Path = checklist_data_dir / "sentiment_suite.pkl"
    
    sa_models_file = download_dir / "models" / "sentiment_models.txt"
    
    
    BASELINES = {
        "checklist": {
            "testsuite_file": checklist_sa_dataset_file
        }
    }

    # Retrain
    TRAIN_RATIO = 0.8
    retrain_output_dir: Path = result_dir / "retrain"
    retrain_model_dir: Path = retrain_output_dir / "models"
    retrain_dataset_dir: Path = retrain_output_dir / "datasets"
    checklist_sa_testcase_file: Path = retrain_dataset_dir / "checklist_sa_testcase.json"
    sst_sa_testcase_file: Path = retrain_dataset_dir / "sst_sa_testcase.json"
