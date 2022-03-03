# This script reads linguisiti capabilities described
# and generates its requirements

from typing import *

import re, os
import sys
import json
import random

from pathlib import Path
from ..utils.Macros import Macros
from ..utils.Utils import Utils

# dataset for NLP task
# key: NLP task
# value: name of dataset for a NLP task
datasets = Macros.datasets

class Requirements:
    
    @classmethod
    def convert_test_type_txt_to_json(cls):
        task = Macros.qqp_task
        dataset_path = Macros.result_dir / f"test_type_{task}.txt"
        result = None
        if os.path.exists(dataset_path):
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
        # end if
        return

    @classmethod
    def get_requirements(cls, task):
        req_file = Macros.result_dir / f"requirements_{task}.json"
        if os.path.exists(req_file):
            return Utils.read_json(req_file)
        # end if
        
        reqs = None
        test_type_file = Macros.result_dir / f"test_type_{task}.json"

        # cap_decp:
        # key: liguistic capability to be evaluated
        # value: each test type description
        if not os.path.exists(test_type_file):
            cap_desc = cls.convert_test_type_txt_to_json()
        else:
            cap_desc = Utils.read_json(test_type_file)
        # end if
        reqs = list()
        for cap, descs in cap_desc.items():
            for d in descs:
                if d.lower()=="adding an adjective makes questions non-duplicates":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search_pairs": False,
                        "search": [{
                            "length": "<10",
                            "exclude": {
                                "POS": ["adj"],
                                "word": None,
                            },
                        }],
                        "transform": {
                            "MFT": "add adj",
                        }
                    })
                elif d.lower()=="add modifiers that preserve question semantics (e.g. 'really')":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search_pairs": False,
                        "search": [{
                            "include": {
                                "POS": None,
                                "word": ['really|truly|actually|indeed|in fact'],
                                "search_pairs": False
                            },
                        }],
                        "transform":  {
                            "MFT": "remove really|truly|actually|indeed|in fact",
                        }
                    })
                # elif d.lower()=="ask the same question about two different noun, expect prediction to be 0":
                #     reqs.append({
                #         "capability": cap,
                #         "description": d,
                #         "search_pairs": False,
                #         "search": [
                #             {
                #                 "include": {
                #                     "POS": ["noun"],
                #                     "word": None
                #                 }
                #             },
                #         ],
                #         "transform": {
                #             "MFT": "replace noun",
                #         }
                #     })
                elif d.lower()=="synonyms in templates":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search_pairs": False,
                        "search": [
                            {
                                "include": {
                                    "POS": None,
                                    "word": ["<synonym>"],
                                }
                            },
                        ],
                        "transform": {
                            "MFT": "replace synonyms",
                        }
                    })
                elif d.lower()=="more x = less antonym(x)":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search_pairs": False,
                        "search": [{
                                "include": {
                                    "POS": None,
                                    "word": ["more|less"],
                                }
                            },
                        ],
                        "transform": {
                            "MFT": "replace more_less",
                        }
                    })
                elif d.lower()=="is , used to be, non-duplicate":
                    pass
                elif d.lower()=="before is not equal to after, non-duplicate":
                    pass
                elif d.lower()=="before becoming is not equal to after becoming":
                    pass
                elif d.lower()=="simple negation, non-duplicate":
                    pass
                elif d.lower()=="negation of antonym, should be duplicate":
                    pass
                # end if
            # end for
        # end for
        Utils.write_json(reqs,
                         req_file,
                         pretty_format=True)
        return reqs

    # @classmethod
    # def search_seed_with_requirements(cls, task, requirements):
    #     if task not in datasets.keys():
    #         raise f"{task} is not defined in this work"
    #     # end if
    #     requirements = cls.get_requirements(task)
    #     dataset_dir = Macros.dataset_dir / datasets[task]
    #     if task=="sentiment_analysis":


# if __name__=="__main__":
#     Requirements.convert_test_type_txt_to_json()
#     for task in datasets.keys():
#         reqs = Requirements.get_requirements(task)
#     # end for
    
    
# Requirements.convert_test_type_txt_to_json()