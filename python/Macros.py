# This script is for defining all macros used in scripts in slpproject/python/hdlp/

from typing import *

import os
from pathlib import Path


class Macros:

    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__))) # nlptest/python/
    root_dir: Path = this_dir.parent # nlptest/
    
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
    datasets = {
        "sentiment_analysis": "stanfordSentimentTreebank"
    }

    # SST
    sst_datasent_file: Path = dataset_dir / "stanfordSentimentTreebank" / "datasetSentences.txt"
    sst_dict_file: Path = dataset_dir / "stanfordSentimentTreebank" / "dictionary.txt"
    sst_label_file: Path = dataset_dir / "stanfordSentimentTreebank" / "sentiment_labels.txt"

    # SentiWordNet
    swn_data_file: Path = download_dir / "SentiWordNet" / "data" / "SentiWordNet_3.0.0.txt"

    # Test
    checklist_dir: Path = download_dir / "checklist"
    checklist_data_dir: Path = checklist_dir / "release_data"
    google_sa_pred_path: Path = checklist_data_dir / "sentiment" / "predictions" / "google"
    ms_sa_pred_path: Path = checklist_data_dir / "sentiment" / "predictions" / "microsoft"
    ama_sa_pred_path: Path = checklist_data_dir / "sentiment" / "predictions" / "amazon"
    bert_sa_pred_path: Path = checklist_data_dir / "sentiment" / "predictions" / "bert"
    
    
    
