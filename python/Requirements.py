# This script reads linguisiti capabilities described
# and generates its requirements

from typing import *

import re, os
import sys
import json
import random

from pathlib import Path
from Macros import Macros
from Utils import Utils

class Requirements:

    # dataset for NLP task
    # key: NLP task
    # value: name of dataset for a NLP task
    datasets = {
        "sentiment_analysis": "stanfordSentimentTreebank"
    }
    
    @classmethod
    def convert_test_type_txt_to_json(cls):
        for task, dataset_name in cls.datasets.items():
            dataset_path = Macros.result_dir / f"test_type_{task}.txt"
            result = None
            with open(dataset_path, 'r') as f:
                result = {
                    l.split("::")[0].strip(): [
                        _l.strip()
                        for _l in l.split("::")[-1].split("|")
                    ]
                    for l in f.readlines()
                }
            # end with
            if result is not None:
                 Utils.write_json(result,
                                  Macros.result_dir / f"test_type_{task}.json",
                                  pretty_format=True)
            # end if                
        # end for
        return

    @classmethod
    def get_requirements(cls):
        all_reqs = dict()
        for task, dataset_name in cls.datasets.items():
            test_type_file = Macros.result_dir / f"test_type_{task}.json"
            if os.path.exists(test_type_file):
                # cap_decp:
                # key: liguistic capability to be evaluated
                # value: each test type description
                cap_desc = Utils.read_json(test_type_file)
                reqs = list()
                for cap, descs in cap_desc.items():
                    for d in descs:
                        if d.lower()=="short sentences with neutral adjectives and nouns":
                            reqs.append({
                                "capability": cap,
                                "length": 10,
                                "contains": ["neutral adjective", "noun"]
                            })
                        # end if
                    # end for
                # end for
                all_reqs[task] = reqs
            # end if
        # end for
        return all_reqs


if __name__=="__main__":
    reqs = Requirements.get_requirements()
    print(reqs)
