# This script is to validate expanded sentences
# given a seed input and their expanded CFGs.
# using BERT-like language model for word
# suggestion

from typing import *

import re, os
import nltk
import spacy
import copy
import time
import random
import numpy as np
# import multiprocessing

from pathlib import Path
# from scipy.special import softmax
# from spacy_wordnet.wordnet_annotator import WordnetAnnotator
# # from nltk.tokenize import word_tokenize as tokenize

# import checklist
# from checklist.editor import Editor
# from checklist.perturb import Perturb

from ..utils.Macros import Macros
from ..utils.Utils import Utils

from ..seed.Search import SearchOperator, SENT_DICT
from ..seed.Transform import F12_TEMPLATES, \
    F13_TEMPLATES, \
    F14_TEMPLATES, \
    F15_TEMPLATES, \
    F16_TEMPLATES, \
    F17_TEMPLATES, \
    F20_TEMPLATES, \
    F21_TEMPLATES


class Validate:

    LABELS = list(Macros.hs_label_map.keys()) # [non-toxic, toxic]

    TRANFORM_TEMPLATE_MAP = {
        'template f12': {
            LABELS[0]: [v for k, v in F12_TEMPLATES.items() if k.endswith('nt')],
            LABELS[1]: [v for k, v in F12_TEMPLATES.items() if k.endswith('tx')]
        },
        'template f13': {
            LABELS[0]: [v for k, v in F13_TEMPLATES.items() if k.endswith('nt')],
            LABELS[1]: [v for k, v in F13_TEMPLATES.items() if k.endswith('tx')]
        },
        'template f14': list(F14_TEMPLATES.values()),
        'template f15': list(F15_TEMPLATES.values()),
        'template f16': list(F16_TEMPLATES.values()),
        'template f17': list(F17_TEMPLATES.values()),
        'template f20': list(F20_TEMPLATES.values()),
        'template f21': list(F21_TEMPLATES.values())
    }

    @classmethod
    def get_templates(cls, transform_spec):
        templates = cls.TRANFORM_TEMPLATE_MAP[transform_spec]
        return templates
    
    @classmethod
    def is_new_token_in_template(cls, sent, label, transform_spec):
        templates = cls.get_templates(transform_spec)
        conformance_list_over_temps = list()
        if type(templates)==dict:
            templates = templates[label]
        # end if
        for template in templates:
            conformance_list = list()
            for tp_key in template.keys():
                if any(template[tp_key]):
                    tp_vals = template[tp_key]
                    conformance = [
                        True if re.search(re.escape(tp_val), sent) else False
                        for tp_val in tp_vals
                    ]
                    conformance_val = True if any(conformance) else False
                    conformance_list.append(conformance_val)
                # end if
            # end for
            conformance_list_over_temps.append(all(conformance_list))
        # end for
        return any(conformance_list_over_temps)
    
    @classmethod
    def is_conform_to_template(cls, **kwargs):
        transform_spec = kwargs['transform_spec']
        sent = kwargs['sent']
        label = kwargs['label']
        return cls.is_new_token_in_template(
            sent,
            label,
            transform_spec
        )

    @classmethod
    def validate_lc_template(cls, **kwargs):
        transform_spec = kwargs['transform_spec']
        sents = kwargs['sents']
        labels = kwargs['labels']
        valid_sents = [
            s for s in sents
            if cls.is_conform_to_template(
                    sent=sent,
                    transform_spec=transform_spec
            )
        ]
        return valid_sents
    
