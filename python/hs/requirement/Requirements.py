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
                if d.lower()=='expression of strong negative emotions (explicit)':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'use_testcase': 'hatecheck',
                        'expansion': ['neutral']
                    })
                elif d.lower()=='description using very negative attributes (explicit)':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'use_testcase': 'hatecheck',
                        'expansion': ['neutral']
                    })
                elif d.lower()=='dehumanisation (explicit)':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'use_testcase': 'hatecheck',
                        'expansion': ['neutral']
                    })
                elif d.lower()=='implicit derogation':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'use_testcase': 'hatecheck',
                        'expansion': ['neutral']
                    })
                elif d.lower()=='direct threat':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'use_testcase': 'hatecheck',
                        'expansion': ['neutral']
                    })
                elif d.lower()=='threat as normative statement':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'use_testcase': 'hatecheck',
                        'expansion': ['neutral']
                    })
                elif d.lower()=='hate expressed using slur':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [
                            {
                                'include': {
                                    'word': ['<hatecheck_slur>']
                                },
                                'label': 'toxic'
                            }
                        ],
                        'expansion': ['neutral']
                    })
                elif d.lower()=='non-hateful use of slur':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [
                            {
                                'include': {
                                    'word': ['<hatecheck_slur>']
                                },
                                'label': 'non-toxic'
                            }
                        ],
                        'expansion': ['neutral']
                    })
                elif d.lower()=='hate expressed using profanity':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [
                            {
                                'include': {
                                    'word': ['<hatecheck_profanity>']
                                },
                                'label': 'toxic'
                            }
                        ],
                        'expansion': ["neutral"]
                    })
                elif d.lower()=='non-hateful use of profanity':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [
                            {
                                'include': {
                                    'word': ['<hatecheck_profanity>']
                                },
                                'label': 'non-toxic'
                            }
                        ],
                        'expansion': ['neutral']
                    })
                elif d.lower()=='hate expressed through reference in subsequent clauses':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [
                            {
                                'length': None,
                                'include': {
                                    'POS': None,
                                    'word': ['<hatecheck_ph>']
                                },
                                'label': 'toxic'
                            }, {
                                'length': None,
                                'include': {
                                    'POS': None,
                                    'word': ['<hatecheck_ph>']
                                },
                                'label': 'non-toxic'
                            }
                        ],
                        'expansion': ['neutral'],
                        'transform': 'template f12',
                        'transform_req': [
                            {
                                'label': 'toxic'
                            }
                        ]
                    })
                elif d.lower()=='hate expressed through reference in subsequent sentences':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [
                            {
                                'length': None,
                                'include': {
                                    'POS': None,
                                    'word': ['<hatecheck_ph>']
                                },
                                'label': 'toxic'
                            }, {
                                'length': None,
                                'include': {
                                    'POS': None,
                                    'word': ['<hatecheck_ph>']
                                },
                                'label': 'non-toxic'
                            }
                        ],
                        'expansion': ['neutral'],
                        'transform': 'template f13',
                        'transform_req': [
                            {
                                'label': 'toxic'
                            }
                        ]
                    })
                elif d.lower()=='hate expressed using negated positive statement':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [
                            {
                                'label': 'non-toxic'
                            }
                        ],
                        'expansion': ['neutral'],
                        'transform': 'template f14',
                        'transform_req': [
                            {
                                'label': 'toxic'
                            }
                        ]
                    })
                elif d.lower()=='non-hate expressed using negated hateful statement':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [
                            {
                                'label': 'toxic'
                            }
                        ],
                        'expansion': ['neutral'],
                        'transform': 'template f15',
                        'transform_req': [
                            {
                                'label': 'non-toxic'
                            }
                        ]
                    })
                elif d.lower()=='hate phrased as a question':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [
                            {
                                'label': 'toxic'
                            }
                        ],
                        'expansion': ['neutral'],
                        'transform': 'template f16',
                        'transform_req': [
                            {
                                'label': 'toxic'
                            }
                        ]
                    })
                elif d.lower()=='hate phrased as a opinion':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [
                            {
                                'label': 'toxic'
                            }
                        ],
                        'expansion': ['neutral'],
                        'transform': 'template f17',
                        'transform_req': [
                            {
                                'label': 'toxic'
                            }
                        ]
                    })
                elif d.lower()=='neutral statements using protected group identifiers':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [
                            {
                                'length': None,
                                'include': {
                                    'word': ['<hatecheck_ph>']
                                },
                                'exclude': {
                                    'POS': ['positive adjs', 'negative adjs', 'positive nouns', 'negative nouns', 'positive verbs', 'negative verbs'],
                                },
                                'label': 'non-toxic'
                            }
                        ],
                        'expansion': ['neutral'],
                    })
                elif d.lower()=='positive statements using protected group identifiers':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [
                            {
                                'length': None,
                                'include': {
                                    'word': ['<hatecheck_ph>']
                                },
                                'exclude': {
                                    'POS': ['negative adjs', 'negative nouns', 'negative verbs'],
                                },
                                'label': 'non-toxic'
                            }
                        ],
                        'expansion': ['neutral'],
                    })
                elif d.lower()=='denouncements of hate that quote it':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [
                            {
                                'label': 'toxic'
                            }
                        ],
                        'expansion': ['neutral'],
                        'transform': 'template f20',
                        'transform_req': [
                            {
                                'label': 'non-toxic'
                            }
                        ]
                    })
                elif d.lower()=='denouncements of hate that make direct reference to it':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [
                            {
                                'label': 'toxic'
                            }
                        ],
                        'expansion': ['neutral'],
                        'transform': 'template f21',
                        'transform_req': [
                            {
                                'label': 'non-toxic'
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
