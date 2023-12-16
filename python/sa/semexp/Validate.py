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
from ..seed.Transform import WORD2POS_MAP, \
    NEG_OF_NEG_AT_THE_END_PHRASE_TEMPLATE, \
    DISAGREEMENT_PHRASE, \
    CUR_NEG_TEMPORAL_PHRASE_TEMPLATE, \
    CUR_POS_TEMPORAL_PHRASE_TEMPLATE, \
    SRL_PHASE_TEMPLATE, \
    QUESTIONIZE_PHRASE_TEMPLATE


class Validate:

    TRANFORM_TEMPLATE_MAP = {
        'add temporal_awareness': {
            'negative': CUR_NEG_TEMPORAL_PHRASE_TEMPLATE,
            'positive': CUR_POS_TEMPORAL_PHRASE_TEMPLATE
        },
        'negate ^demonstratives_AUXBE': [
            WORD2POS_MAP
        ],
        'negate AUXBE$': [
            NEG_OF_NEG_AT_THE_END_PHRASE_TEMPLATE
        ],
        'negate positive': [
            DISAGREEMENT_PHRASE
        ],
        'srl': [
            SRL_PHASE_TEMPLATE
        ],
        'questionize yes': [
            QUESTIONIZE_PHRASE_TEMPLATE
        ],
        'questionize no': [
            QUESTIONIZE_PHRASE_TEMPLATE
        ],
    }

    @classmethod
    def get_templates(cls, transform_spec):
        templates = cls.TRANFORM_TEMPLATE_MAP.get(transform_spec, None)
        return templates
    
    @classmethod
    def is_new_token_in_template(cls, sent, label, transform_spec):
        templates = cls.get_templates(transform_spec)
        if templates is not None:
            conformance_list = list()
            template = templates[label] if type(templates)==dict else templates[0]
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
            return all(conformance_list)
        else:
            return True
        # end if

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
