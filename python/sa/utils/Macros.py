# This script is for defining all macros used in scripts in slpproject/python/hdlp/

from typing import *
from pathlib import Path

import os

class Macros:

    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    root_dir: Path = this_dir.parent.parent.parent 
    python_dir: Path = this_dir.parent.parent
    storage_dir: Path = Path('/nas1-nfs1/data/jxl115330/s2lct')
    # storage_dir: Path = root_dir
    result_dir: Path = storage_dir / "_results"
    
    download_dir: Path = storage_dir / "_downloads"
    log_dir: Path = storage_dir / "_logs"
    paper_dir = root_dir / "paper" / "fse23"
    dataset_dir = download_dir / "datasets"

    FMT_INT = "{:,d}"
    FMT_PER = "{:.1%}"
    FMT_FLOAT = "{:,.2f}"
    
    RAND_SEED = {1: 27, 2: 26, 3: 28} #26(trial2), 27(trial1), 28(trial3)
    MASK = "{mask}"

    num_processes = 4

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
    num_cfg_exp_elem = -1 # number of syntax suggestions used in Generator
    num_suggestions_on_exp_grammer_elem = 20 # number of word suggestions used in Suggest
    nsamples = 11000 # 500
    max_num_seeds = 50 # maximum number of selected sentences
    max_num_sents = 1000000 # number of testcase sentences
    max_num_sents_for_perturb = 1000 # number of sentences for perturbation
    num_synonym_placeholders = 5
    max_num_synonyms = 10 # number of synonyms to be used when there are too many placeholders in one sentence

    # SST
    sst_datasent_file: Path = dataset_dir / "stanfordSentimentTreebank" / "datasetSentences.txt"
    sst_dict_file: Path = dataset_dir / "stanfordSentimentTreebank" / "dictionary.txt"
    sst_label_file: Path = dataset_dir / "stanfordSentimentTreebank" / "sentiment_labels.txt"
    
    # Tweets
    tweet_file: Path = dataset_dir / "airplanetweets" / "Tweets.csv"

    # Dynasent
    dyna_r1_test_src_file: Path = dataset_dir / "dynasent" / "dynasent-v1.1" / "dynasent-v1.1-round01-yelp-test.jsonl"

    # SentiWordNet
    swn_data_file: Path = dataset_dir / "SentiWordNet" / "data" / "SentiWordNet_3.0.0.txt"
    
    # Checklist Testsuite
    checklist_dir: Path = download_dir / "checklist"
    checklist_data_dir: Path = checklist_dir / "release_suites"
    checklist_sa_dataset_file: Path = checklist_data_dir / "sentiment_suite.pkl"
    checklist_sa_fairness_dataset_file: Path = checklist_data_dir / "sentiment_suite_for_fairness.pkl"
    
    sa_models_file = download_dir / "models" / "sentiment_models.txt"
    
    
    BASELINES = {
        "checklist": {
            "testsuite_file": checklist_sa_dataset_file
        },
        "checklist_fairness": {
            "testsuite_file": checklist_sa_fairness_dataset_file
        }
    }

    # Retrain
    TRAIN_RATIO = 1.0
    retrain_output_dir: Path = result_dir / "retrain"
    retrain_model_dir: Path = retrain_output_dir / "models"
    retrain_dataset_dir: Path = retrain_output_dir / "datasets"
    checklist_sa_testcase_file: Path = retrain_dataset_dir / "sa_checklist_testcase.json"
    sst_sa_testcase_file: Path = retrain_dataset_dir / "sst_sa_testcase.json"

    # Self-Bleu
    selfbleu_result_dir: Path = result_dir / 'selfbleu'
    
    # Production Rule Coverage
    pdr_cov_result_dir: Path = result_dir / 'pdr_cov'
    
    # SST2
    sst2_dir: Path = dataset_dir / "sst2"
    sst2_sa_trainset_file: Path = retrain_dataset_dir / "sa_sst2_trainset.json"


    CHECKLIST_LC_LIST = [
        'Sentiment-laden words in context',
        'neutral words in context',
        'used to, but now',
        'simple negations: not negative',
        'simple negations: not neutral is still neutral',
        'Hard: Negation of positive with neutral stuff in the middle (should be negative)',
        'simple negations: I thought x was negative, but it was not (should be neutral or positive)',
        'my opinion is what matters',
        'Q & A: yes', #'Q & A: yes (neutral)',
        'Q & A: no'
    ] # length=10

    OUR_LC_LIST = [
        'Short sentences with sentiment-laden adjectives',
        'Short sentences with neutral adjectives and nouns',
        'Sentiment change over time, present should prevail',
        'Negated negative should be positive or neutral',
        'Negated neutral should still be neutral',
        'Negated positive with neutral content in the middle',
        'Negation of negative at the end, should be positive or neutral',
        'Author sentiment is more important than of others',
        'parsing sentiment in (question, yes) form',
        'Parsing sentiment in (question, no) form',
    ] # length=10

    LC_MAP = {
        CHECKLIST_LC_LIST[0]: OUR_LC_LIST[0],
        CHECKLIST_LC_LIST[1]: OUR_LC_LIST[1],
        CHECKLIST_LC_LIST[2]: OUR_LC_LIST[2],
        CHECKLIST_LC_LIST[3]: OUR_LC_LIST[3],
        CHECKLIST_LC_LIST[4]: OUR_LC_LIST[4],
        CHECKLIST_LC_LIST[5]: OUR_LC_LIST[5],
        CHECKLIST_LC_LIST[6]: OUR_LC_LIST[6],
        CHECKLIST_LC_LIST[7]: OUR_LC_LIST[7],
        CHECKLIST_LC_LIST[8]: OUR_LC_LIST[8],
        CHECKLIST_LC_LIST[9]: OUR_LC_LIST[9]
    }

    # ChatGPT
    # ==========
    openai_chatgpt_engine_name = 'gpt-3.5-turbo-instruct'
    openai_chatgpt4_engine_name = 'gpt-4'
    openai_chatgpt_sa_prompt = "write the sentiment of the following sentence between positive, neutral and negative and respond with 'the sentiment is {sentiment}' and nothing else:"
    resp_temp = 0.
    llm_resp_max_len = 100