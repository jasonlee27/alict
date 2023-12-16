# This script is for defining all macros used in scripts in slpproject/python/hdlp/

from typing import *
from pathlib import Path

import os

class Macros:

    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__))) # s2lct/python/utils
    root_dir: Path = this_dir.parent.parent.parent # s2lct/
    storage_dir: Path = Path('/nas1-nfs1/data/jxl115330/s2lct')
    # storage_dir: Path = root_dir
    result_dir: Path = storage_dir / '_results' 
    # result_dir: Path = root_dir / "_results" # s2lct/_results
    
    download_dir: Path = storage_dir / '_downloads' # /glusterfs/data/jxl115330/s2lct/_downloads
    log_dir: Path = storage_dir / '_logs' # /glusterfs/data/jxl115330/s2lct/_logs
    paper_dir = root_dir / 'paper' / 'fse23'
    dataset_dir = download_dir / 'datasets'

    FMT_INT = "{:,d}"
    FMT_PER = "{:.1%}"
    FMT_FLOAT = "{:,.2f}"
    
    RAND_SEED = {1: 27, 2: 26, 3: 28} #26(trial2), 27(trial1), 28(trial3)
    MASK = "{mask}"
    
    num_processes = 3

    ## hate speech setting
    hs_task = 'hs'
    nlp_tasks = [hs_task]
    sa_label_map = {'negative': 0, 'positive': 2, 'neutral': 1}
    hs_label_map = {'non-toxic': 0, 'toxic': 1}
    
    datasets = {
        hs_task: ['hatexplain', 'hatecheck'],
    }
    
    # HateXplain dataset (hate speech detection dataset):
    hatexplain_data_file: Path = dataset_dir / 'hs' / 'HateXplain' / 'Data' / 'dataset.json'
    
    # Hurtlex lexicon dataset
    hurtlex_data_file: Path = dataset_dir / 'hs' / 'hurtlex' / 'lexica' / 'EN' / '1.2' / 'hurtlex_EN.tsv'

    # Hatecheck dataset
    hatecheck_data_file: Path = dataset_dir / 'hs' / 'hatecheck' / 'hatecheck-data' / 'test_suite_annotations.csv'
    hatecheck_testsuite_file: Path = result_dir / 'hatecheck' / 'hs_hatecheck_testsuite.pkl'

    # SentiWordNet
    swn_data_file: Path = dataset_dir / 'SentiWordNet' / 'data' / 'SentiWordNet_3.0.0.txt'

    BASELINES = {
        datasets[hs_task][-1]: {
            "testsuite_file": hatecheck_data_file
        }
    }

    # Testsuite
    num_cfg_exp_elem = -1 # number of syntax suggestions used in Generator
    num_suggestions_on_exp_grammer_elem = 20 # number of word suggestions used in Suggest
    nsamples = 1000000 # 500
    max_num_seeds = 50 # maximum number of selected sentences
    max_num_sents = 100000 # number of testcase sentences
    max_num_sents_for_perturb = 1000 # number of sentences for perturbation
    num_synonym_placeholders = 5
    max_num_synonyms = 10 # number of synonyms to be used when there are too many placeholders in one sentence

    hs_models_file = download_dir / "models" / "hs_models.txt"

    # Self-Bleu
    selfbleu_result_dir: Path = result_dir / 'selfbleu'
    
    # Production Rule Coverage
    pdr_cov_result_dir: Path = result_dir / 'pdr_cov'
    
    OUR_LC_LIST = [
        'Derogation::Expression of strong negative emotions (explicit)',
        'Derogation::Description using very negative attributes (explicit)',
        'Derogation::Dehumanisation (explicit)',
        'Derogation::Implicit derogation',
        'Threatening language::Direct threat',
        'Threatening language::Threat as normative statement',
        'Slur usage::Hate expressed using slur',
        'Slur usage::Non-hateful use of slur',
        'Profanity usage::Hate expressed using profanity',
        'Profanity usage::Non-Hateful use of profanity',
        'Pronoun reference::Hate expressed through reference in subsequent clauses',
        'Pronoun reference::Hate expressed through reference in subsequent sentences',
        'Negation::Hate expressed using negated positive statement',
        'Negation::Non-hate expressed using negated hateful statement',
        'Phrasing::Hate phrased as a question',
        'Phrasing::Hate phrased as a opinion',
        'Non-hate grp. ident.::Neutral statements using protected group identifiers',
        'Non-hate grp. ident.::Positive statements using protected group identifiers',
        'Counter speech::Denouncements of hate that quote it',
        'Counter speech::Denouncements of hate that make direct reference to it',
    ]
    
    # ChatGPT
    # ==========
    openai_chatgpt_engine_name = 'gpt-3.5-turbo-instruct'
    openai_chatgpt4_engine_name = 'gpt-4'
    openai_chatgpt_hs_prompt = "write if the following sentence is hate speech or not and respond with 'the sentence is {toxic/non-toxic}' and nothing else:"
    resp_temp = 0.
    llm_resp_max_len = 100