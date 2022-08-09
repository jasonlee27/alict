# This script is for defining all macros used in scripts in slpproject/python/hdlp/

from typing import *
from pathlib import Path

import os

class Macros:

    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__))) # s2lct/python/utils
    root_dir: Path = this_dir.parent.parent.parent # s2lct/
    storage_dir: Path = Path("/glusterfs/data/jxl115330/s2lct")
    result_dir: Path = storage_dir / "_results" # /glusterfs/data/jxl115330/s2lct/_results
    # result_dir: Path = root_dir / "_results" # s2lct/_results
    
    download_dir: Path = storage_dir / "_downloads" # /glusterfs/data/jxl115330/s2lct/_downloads
    log_dir: Path = storage_dir / "_logs" # /glusterfs/data/jxl115330/s2lct/_results
    paper_dir = root_dir / "paper" / "ase22"
    dataset_dir = download_dir / "datasets"

    FMT_INT = "{:,d}"
    FMT_PER = "{:.1%}"
    FMT_FLOAT = "{:,.2f}"
    
    SEED = 27
    MASK = "{mask}"

    ## hate speech setting
    hs_task = 'hs'
    nlp_tasks = [hs_task]
    sa_label_map = {'negative': 0, 'positive': 2, 'neutral': 1}
    hs_label_map = {'toxic': 0, 'non-toxic': 1}

    # HateXplain dataset (hate speech detection dataset):
    datasets = {
        hs_task: ['hatexplain'],
    }
    hatexplain_data_file: Path = dataset_dir / 'hs' / 'HateXplain' / 'Data' / 'dataset.json'
    
    # Hurtlex lexicon dataset
    hurtlex_data_file: Path = dataset_dir / 'hs' / 'hurtlex' / 'lexica' / 'EN' / '1.2' / 'hurtlex_EN.tsv'

    # Hatecheck dataset
    hatecheck_data_file: Path = dataset_dir / 'hs' / 'hatecheck' / 'hatecheck-data' / 'test_suite_annotations.csv'

    # SentiWordNet
    swn_data_file: Path = dataset_dir / "SentiWordNet" / "data" / "SentiWordNet_3.0.0.txt"

    # Testsuite
    num_cfg_exp_elem = -1 # number of syntax suggestions used in Generator
    num_suggestions_on_exp_grammer_elem = 20 # number of word suggestions used in Suggest
    nsamples = 11000 # 500
    max_num_seeds = 50 # maximum number of selected sentences
    max_num_sents = 100000 # number of testcase sentences
    max_num_sents_for_perturb = 1000 # number of sentences for perturbation
    num_synonym_placeholders = 5
    max_num_synonyms = 10 # number of synonyms to be used when there are too many placeholders in one sentence

    hs_models_file = download_dir / "models" / "hs_models.txt"

    OUR_LC_LIST = [
        'Expression of strong negative emotions(explicit)',
        'Description using very negative attributes(explicit)',
        'Dehumanisation (explicit)',
        'Implicit derogation',
        'Direct threat',
        'Threat as normative statement',
        'Hate expressed using slur',
        'Non-hateful homonyms of slurs',
        'Reclaimed slurs',
        'Hate expressed using profanity',
        'Non-hateful use of profanity',
        'Hate expressed through reference in subsequent clauses',
        'Hate expressed through reference in subsequent sentences',
        'Hate expressed using negated positive statement',
        'Non-hate expressed using negated hateful statement',
        'Hate phrased as a question',
        'Hate phrased as an opinion',
        'Neutral statements using protected group identifiers',
        'Positive statements using protected group identifiers',
        'Denouncements of hate that quote it',
        'Denouncements of hate that make direct reference to it',
        'Abuse targeted at objects',
        'Abuse targeted at individuals',
        'Abuse targeted at non-protected groups'
    ]
    
