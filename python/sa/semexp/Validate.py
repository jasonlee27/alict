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
from ..seed.Transform import NEG_OF_NEG_AT_THE_END_PHRASE_TEMPLATE, \
    DISAGREEMENT_PHRASE, \
    CUR_NEG_TEMPORAL_PHRASE_TEMPLATE, \
    CUR_POS_TEMPORAL_PHRASE_TEMPLATE, \
    SRL_PHASE_TEMPLATE, \
    QUESTIONIZE_PHRASE_TEMPLATE


class Validate:

    @classmethod
    def get_templates(cls, transform_spec):
        pass
    
    @classmethod
    def is_new_token_in_template(cls, sent, transform_spec):
        template = cls.get_templates(transform_spec)
        conformance_list = list()
        for tp_key in template.keys():
            if any(template[tp_key]) and not conformance_list:
                tp_vals = template[tp_key]
                conformance = [
                    True if re.search(tp_val, sent) else False
                    for tp_val in tp_vals
                ]
                if any(conformance):
                    conformance_list.append(True)
                # end if
            # end if
        # end for
        return all(conformance_list)
    
    @classmethod
    def is_conform_to_template(cls, **kwargs):
        transform_spec = kwargs['transform_spec']
        sent = kwargs['sent']
        return cls.is_new_token_in_template(sent, transform_spec)

    @classmethod
    def validate_lc_template(cls, **kwargs):
        transform_spec = kwargs['transform_spec']
        sents = kwargs['sents']
        valid_sents = [
            s for s in sents
            if cls.is_conform_to_template(
                    sent=sent,
                    transform_spec=transform_spec
            )
        ]
        return valid_sents
