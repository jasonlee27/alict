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
            dataset_path = cls.req_dir / f"test_type_{task}.txt"
            result = None
            if os.path.exists(dataset_path):
                with open(dataset_path, 'r') as f:
                    result = {
                        l.split('::')[0].strip(): [
                            _l.strip()
                            for _l in l.split('::')[-1].split('|')
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
                if d.lower()=='expression of strong negative emotions(explicit)':
                    # Search: seed={label: 1(hate speech); include: “I {hate synonyms} {hurtlex nouns}”}
                    # Transform: N/A
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [{
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I <hate_syn> <hurtlex_n>']
                            },
                            'label': 'hate'
                        }],
                        'expansion': ['neutral'],
                        'transform': None,
                        'transform_req': None
                    })
                elif d.lower()=='description using very negative attributes(explicit)':
                    pass
                elif d.lower()=='dehumanisation (explicit)':
                    pass
                elif d.lower()=='implicit derogation':
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
    #     if task=='sentiment_analysis':


# if __name__=='__main__':
#     Requirements.convert_test_type_txt_to_json()
#     for task in datasets.keys():
#         reqs = Requirements.get_requirements(task)
#     # end for
    
    
# Requirements.convert_test_type_txt_to_json()
