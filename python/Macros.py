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

    ## NLP dataset
    datasets = {
        "sentiment_analysis": "stanfordSentimentTreebank"
    }

    # SST
    sst_datasent_file = dataset_dir / "stanfordSentimentTreebank" / "datasetSentences.txt"
    sst_dict_file = dataset_dir / "stanfordSentimentTreebank" / "dictionary.txt"
    sst_label_file = dataset_dir / "stanfordSentimentTreebank" / "sentiment_labels.txt"
