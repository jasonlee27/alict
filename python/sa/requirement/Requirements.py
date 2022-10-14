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

    req_dir = Macros.result_dir / 'reqs'
    
    @classmethod
    def convert_test_type_txt_to_json(cls):
        for task, dataset_name in datasets.items():
            dataset_path = cls.req_dir / f"requirements_desc_{task}.txt"
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
                                     cls.req_dir / f"test_type_{task}.json",
                                     pretty_format=True)
                # end if
            # end if
        # end for
        return

    @classmethod
    def get_requirements(cls, task):
        # dataset_name = datasets[task]
        req_file = cls.req_dir / f"requirements_{task}.json"
        if os.path.exists(req_file):
            return Utils.read_json(req_file)
        # end if
        reqs = None
        test_type_file = cls.req_dir / f"test_type_{task}.json"
        cls.req_dir.mkdir(parents=True, exist_ok=True)

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
                        "expansion": ["neutral"],
                        "transform": None,
                        "transform_req": None
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
                                "POS": ["positive verbs", "positive nouns", "negative adjs", "negative verbs", "negative nouns"],
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
                        "expansion": ["neutral"],
                        "transform": None,
                        "transform_req": None
                    })
                elif d.lower()=="sentiment change over time, present should prevail":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [
                            {
                                "length": "<20",
                                "label": "positive"
                            },
                            {
                                "length": "<20",
                                "label": "negative"
                            }
                        ],
                        "expansion": ["neutral"],
                        "transform": "add temporal_awareness",
                        "transform_req": [
                            {
                                "label": "positive"
                            },
                            {
                                "label": "negative"
                            }
                        ]
                    })
                elif d.lower()=="negated negative should be positive or neutral":
                    # AUX : auxilary verb
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [
                            {
                                "label": "negative",
                                "include": {
                                    "POS": ["<^demonstratives_AUXBE>"],
                                    "word": None
                                }
                            },
                        ],
                        "expansion": ["neutral"],
                        "transform": "negate ^demonstratives_AUXBE",
                        "transform_req": [
                            {
                                "label": ["positive", "neutral"]
                            }
                        ]
                    })
                elif d.lower()=="negated neutral should still be neutral":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [
                            {
                                "label": "neutral",
                                "include": {
                                    "POS": ["<^demonstratives_AUXBE>"],
                                    "word": None
                                }
                            },
                        ],
                        "expansion": ["neutral"],
                        "transform": "negate ^demonstratives_AUXBE",
                        "transform_req": None
                    })
                elif d.lower()=="negation of negative at the end, should be positive or neutral":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [
                            {
                                "label": "negative",
                            },
                        ],
                        "expansion": ["neutral"],
                        "transform": "negate AUXBE$",
                        "transform_req": [
                            {
                                "label": ["positive", "neutral"]
                            }
                        ]
                    })
                elif d.lower()=="negated positive with neutral content in the middle":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [
                            {
                                "length": "<20",
                                "label": "positive",
                            },
                            {
                                "length": "<20",
                                "label": "neutral",
                            },
                        ],
                        "expansion": ["neutral"],
                        "transform": "negate positive",
                        "transform_req":  [
                            {
                                "label": "negative"
                            }
                        ]
                    })
                elif d.lower()=="author sentiment is more important than of others":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [
                            {
                                "label": "positive",
                            },
                            {
                                "label": "negative",
                            },
                        ],
                        "expansion": ["neutral"],
                        "transform": "srl",
                        "transform_req": None
                    })
                elif d.lower()=="parsing sentiment in (question, yes) form":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [
                            {
                                "label": "positive",
                            },
                            {
                                "label": "negative",
                            }
                        ],
                        "expansion": ["neutral"],
                        "transform": "questionize yes",
                        "transform_req": None
                    })
                elif d.lower()=="parsing sentiment in (question, no) form":
                    reqs.append({
                        "capability": cap,
                        "description": d,
                        "search": [
                            {
                                "label": "positive",
                            },
                            {
                                "label": "negative",
                            }
                        ],
                        "expansion": ["neutral"],
                        "transform": "questionize no",
                        "transform_req": [
                            {
                                "label": "negative"
                            },
                            {
                                "label": ['positive', 'neutral']
                            }
                        ]
                    })
                # end if
            # end for
        # end for
        Utils.write_json(reqs,
                         req_file,
                         pretty_format=True)
        return reqs
