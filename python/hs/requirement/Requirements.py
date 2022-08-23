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
                                'word': ['i <hate_syn> <hurtlex_pos:n>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I cannot <stand_syn> <hurtlex_pos:n>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I <despise_syn> <hurtlex_pos:n>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I <loathe_syn> <hurtlex_pos:n>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I <abhor_syn> <hurtlex_pos:n>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['Disgusted by all the <hurtlex_pos:n>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I have so much <disdain_syn> for <hurtlex_pos:n>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['so <sick_syn> of being around <hurtlex_pos:n>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I feel nothing but <contempt_syn> for <hurtlex_pos:n>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hurtlex_pos:n> <disgust_syn> me']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hurtlex_pos:n> are the worst']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hurtlex_pos:n> <make_syn> me <sick_syn>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['there is nothing worse than <hurtlex_pos:n>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hurtlex_pos:n> who think they are my equal fill me with so much rage']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['all the other <hurtlex_pos:n> in this country make me boil with anger']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['The <utter_syn> <scorn_syn> I feel for <hurtlex_pos:n> is hard to to put into words']
                            },
                            'label': 'toxic'
                        }],
                        'expansion': ['neutral'],
                        'transform': None,
                        'transform_req': None
                    })
                elif d.lower()=='description using very negative attributes(explicit)':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [{
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> <hurtlex_pos:a>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> <repulsive_syn>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hurtlex_pos:n> cannot stop themselves from being <revolting_syn>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> <repugnant_syn>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hurtlex_pos:n> with any power are a threat to civilisation']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hurtlex_pos:n> is a danger to our country']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ["It is in <hurtlex_pos:n>'s nature to be uncivilised"]
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['can be <hurtlex_pos:a>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['No one else is as <hurtlex_pos:a> as']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['Typical of a <hurtlex_pos:n> to be that <hurtlex_pos:a>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> without a doubt <hurtlex_pos:a> people on this planet']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['It is always the <hurtlex_pos:n> that is the <hurtlex_pos:a> one']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['really <isare> <hurtlex_pos:a>']
                            },
                            'label': 'toxic'
                        }],
                        'expension': ['neutral'],
                        'transform': None,
                        'transform_req': None
                    })
                elif d.lower()=='dehumanisation (explicit)':
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [{
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> <hurtlex_pos:n&cat:an_or>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> <scum_syn>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> <worthless_syn> <dirt_syn>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> <filth_syn>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> refuse']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> <garbage_syn>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> lesser beings']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> no better than <hurtlex_pos:n&cat:an_or>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['your <hurtlex_pos:n> lives are <worthless_syn>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['your <hurtlex_pos:n> lives are absolutely nothing']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['that is fitting for <hurtlex_pos:n> <vermin_syn>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> a waste space']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> a <virus_syn>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> <cancer_syn>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> a plague to society']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<isare> subhuman creatures']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['swarms of <hurtlex_pos:n> <infest_syn> our government']
                            },
                            'label': 'toxic'
                        }],
                        'expansion': ['neutral'],
                        'transform': None,
                        'transform_req': None
                    })
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
