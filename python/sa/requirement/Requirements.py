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
        for task, dataset_name in datasets.items():
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
        # end for
        return

    @classmethod
    def get_requirements(cls, task):
        # dataset_name = datasets[task]
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
                if d.lower()=="short sentences with neutral adjectives and nouns":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [{
                            "length": "<10",
                            "include": {
                                "POS": ["neutral adjs", "neutral nouns"], # elements in list is AND relationship
                                "word": None
                            },
                            "exclude": {
                                "POS": ["positive adjs", "negative adjs", "positive nouns", "negative nouns"],
                                "word": None
                            },
                            "label": "neutral"
                        }],
                        "transform": None
                    })
                elif d.lower()=="short sentences with sentiment-laden adjectives":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [{
                            "length": "<10",
                            "include": {
                                "POS": ["positive adjs"],
                                "word": None
                            },
                            "exclude": {
                                "POS": ["negative adjs", "negative verbs", "negative nouns"],
                                "word": None
                            },
                            "label": "positive"
                        }, {
                            "length": "<10",
                            "include": {
                                "POS": ["negative adjs"],
                                "word": None
                            },
                            "exclude": {
                                "POS": ["negative verbs", "negative nouns", "positive adjs", "positive verbs", "positive nouns"],
                                "word": None
                            },
                            "label": "negative"
                        }],
                        "transform": None
                    })
                elif d.lower()=="replace neutral words with other neutral words":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [
                            {
                                "include": {
                                    "POS": [
                                        "neutral adjs"
                                    ],
                                    "word": None
                                }
                            },
                            {
                                "include": {
                                    "POS": [
                                        "neutral verbs"
                                    ],
                                    "word": None
                                }
                            },
                            {
                                "include": {
                                    "POS": [
                                        "neutral nouns"
                                    ],
                                    "word": None
                                }
                            }
                        ],
                        "transform": {
                            "INV": "replace neutral word",
                            "DIR": None
                        }
                    })
                elif d.lower()=="add positive phrases, fails if sent. goes down by > 0.1":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [],
                        "transform": {
                            "INV": None,
                            "DIR": "add positive phrase"
                        }
                    })
                elif d.lower()=="add negative phrases, fails if sent. goes up by > 0.1":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [],
                        "transform": {
                            "INV": None,
                            "DIR": "add negative phrase"
                        }
                    })
                elif d.lower()=="add randomly generated URLs and handles":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [],
                        "transform": {
                            "INV": "add random URL_handles",
                            "DIR": None
                        }
                    })
                elif d.lower()=="strip punctuation and/or add \".\"":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [
                            {
                                "include": {
                                    "POS": None,
                                    "word": [
                                        "<punctuation>"
                                    ]
                                }
                            }
                        ],
                        "transform": {
                            "INV": "strip punctuation",
                            "DIR": None
                        }
                    })
                elif d.lower()=="swap two adjacent characters":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [],
                        "transform": {
                            "INV": "swap one two_adjacent_characters",
                            "DIR": None
                        }
                    })
                elif d.lower()=="swap two adjacent_characters":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [],
                        "transform": {
                            "INV": "swap two two_adjacent_characters",
                            "DIR": None
                        }
                    })
                elif d.lower()=="contract or expand contractions":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [
                            {
                                "include": {
                                    "POS": None,
                                    "word": [
                                        "<contraction>"
                                    ]
                                }
                            }
                        ],
                        "transform": {
                            "INV": "contract/expand contraction",
                            "DIR": None
                        }
                    })
                elif d.lower()=="replace names with other common names":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [
                            {
                                "include": {
                                    "POS": None,
                                    "word": [
                                        "<person_name>"
                                    ]
                                }
                            }
                        ],
                        "transform": {
                            "INV": "replace person_name",
                            "DIR": None
                        }
                    })
                elif d.lower()=="replace city or country names with other cities or countries":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [
                            {
                                "include": {
                                    "POS": None,
                                    "word": [
                                        "<location_name>"
                                    ]
                                }
                            }
                        ],
                        "transform": {
                            "INV": "replace location_name",
                            "DIR": None
                        }
                    })
                elif d.lower()=="replace integers with random integers within a 20% radius of the original":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [
                            {
                                "include": {
                                    "POS": None,
                                    "word": [
                                        "<number>"
                                    ]
                                }
                            }
                        ],
                        "transform": {
                            "INV": "replace number",
                            "DIR": None
                        }
                    })
                # elif d.lower()=="sentiment change over time, present should prevail":
                #     reqs.append({
                #         "capability": cap,
                #         "description": d,
                #         "search": [
                #             {
                #                 "label": "positive"
                #             },
                #             {
                #                 "label": "negative"
                #             }
                #         ],
                #         "transform": {
                #             "MFT": "add temporal_awareness"
                #         }
                #     })J9
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
