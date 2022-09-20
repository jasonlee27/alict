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
                if d.lower()=='derogation':
                    # Search: seed={label: 1(hate speech); include: “I {hate synonyms} {hurtlex nouns}”}
                    # Transform: N/A
                    reqs.append({
                        'capability': cap,
                        'description': d,
                        'search': [{
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['i <hate_syn> <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I cannot <stand_syn> <hatecheck_ph>']
                            },
                            'Label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I <despise_syn> <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I <loathe_syn> <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I <abhor_syn> <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['Disgusted by all the <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I have so much <disdain_syn> for <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['so <sick_syn> of being around <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I feel nothing but <contempt_syn> for <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hatecheck_ph> <disgust_syn> me']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hatecheck_ph> are the worst']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hatecheck_ph> <make_syn> me <sick_syn>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['there is nothing worse than <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hatecheck_ph> who think they are my equal fill me with so much rage']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['all the other <hatecheck_ph> in this country make me boil with anger']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['The <utter_syn> <scorn_syn> I feel for <hatecheck_ph> is hard to to put into words']
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
                                'word': ['<hatecheck_ph> with any power are a threat to civilisation']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hatecheck_ph> is a danger to our country']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ["It is in <hatecheck_ph> 's nature to be uncivilised"]
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
                                'word': ['Typical of a <hatecheck_ph> to be that <hurtlex_pos:a>']
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
                                'word': ['It is always the <hatecheck_ph> that is the <hurtlex_pos:a> one']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['really <isare> <hurtlex_pos:a>']
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
                                'word': ['your <hatecheck_ph> lives are <worthless_syn>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['your <hatecheck_ph> lives are absolutely nothing']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['that is fitting for <hatecheck_ph> <vermin_syn>']
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
                                'word': ['swarms of <hatecheck_ph> <infest_syn> our government']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hatecheck_ph> belong in a zoo']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hatecheck_ph> who are sick should be taken to a vet, not a doctor']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['My dog is smarter than any <hatecheck_ph> could ever be']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['That monkey is more intelligent than all of you <hatecheck_ph> combined']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hatecheck_ph> should be dumped in a landfill']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I would never eat food that has been touched by a <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I care more about the dirt under my shoe than about <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['The life of any insect is more valuable than that of a <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['We should test new drugs on <hatecheck_ph> before testing them on animals']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I don\'t see any difference between a dead rat and a dead <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['It baffles me why people care even the slightest about the lives of <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['The life of any insect is more valuable than that of a <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hatecheck_ph> should all be locked up in cages']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['Putting a <hatecheck_ph> in charge is the biggest mistake anyone can make']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['Allowing any more <hatecheck_ph> into government will ruin our country']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['Never trust a single word said by a <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I would rather die than date a <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['I\'d rather have cancer than be anywhere near a <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['Never trust a single word said by a <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['If I woke up as a <hatecheck_ph>, I would kill myself']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['There is nothing in the world I like less than a <hatecheck_ph>']
                            },
                            'label': 'toxic'
                        }, {
                            'length': None,
                            'include': {
                                'POS': None,
                                'word': ['<hatecheck_ph> have no place in a civilised country like this']
                            },
                            'label': 'toxic'
                        }],
                        'expansion': ['neutral'],
                        'transform': None,
                        'transform_req': None
                    })
                elif d.lower()=='threatening language':
                    pass
                elif d.lower()=='slur usage':
                    pass
                elif d.lower()=='profanity usage':
                    pass
                elif d.lower()=='negation':
                    pass
                elif d.lower()=='phrasing':
                    pass
                elif d.lower()=='non-hate grp. ident.':
                    pass
                elif d.lower()=='counter speech':
                    pass
                elif d.lower()=='abuse against non-prot. targets':
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
