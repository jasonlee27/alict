# This script perturb and transform inputs in datasets that meet requirements

from typing import *

import re, os
import sys
import json
import spacy
import random
import string
import checklist
import numpy as np

# from checklist.editor import Editor
from itertools import product
from checklist.expect import Expect
from pathlib import Path

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..semexp.Synonyms import Synonyms
from .sentiwordnet.Sentiwordnet import Sentiwordnet


# get pos/neg/neu words from SentiWordNet
SENT_WORDS = Sentiwordnet.get_sent_words()
SENT_DICT = {
    'positive_adj': [w['word'] for w in SENT_WORDS.values() if w['POS']=='adj' and w['label']=='positive'],
    'negative_adj': [w['word'] for w in SENT_WORDS.values() if w['POS']=='adj' and w['label']=='negative'],
    'neutral_adj': [w['word'] for w in SENT_WORDS.values() if w['POS']=='adj' and w['label']=='pure neutral'],
    'positive_verb': [w['word'] for w in SENT_WORDS.values() if w['POS']=='verb' and w['label']=='positive'],
    'negative_verb': [w['word'] for w in SENT_WORDS.values() if w['POS']=='verb' and w['label']=='negative'],
    'neutral_verb': [w['word'] for w in SENT_WORDS.values() if w['POS']=='verb' and w['label']=='pure neutral'],
    'positive_noun': [w['word'] for w in SENT_WORDS.values() if w['POS']=='noun' and w['label']=='positive'],
    'negative_noun': [w['word'] for w in SENT_WORDS.values() if w['POS']=='noun' and w['label']=='negative'],
    'neutral_noun': [w['word'] for w in SENT_WORDS.values() if w['POS']=='noun' and w['label']=='pure neutral']
}

F12_TEMPLATES = {
    'tx-to-tx': {
        'prefix': ['people say', 'I heard that'],
        'sent': [], # toxic
        'conj': ['and'],
        'postfix': ['i hate them too', 'i also hate them']
    },
    'nt-to-tx': {
        'prefix': ['people say', 'i heard that'],
        'sent': [], # non-toxic
        'conj': ['but'],
        'postfix': ['i hate them', 'i dislike them']
    },
    'tx-to-nt': {
        'prefix': ['people say', 'i heard that'],
        'sent': [], # toxic
        'conj': ['but'],
        'postfix': ['i do not hate them', 'i like them']
    },
    'nt-to-nt': {
        'prefix': ['people say', 'i heard that'],
        'sent': [], # non-toxic
        'conj': ['and'],
        'postfix': ['i also do not hate them', 'i do not hate them too']
    }
}

F13_TEMPLATES = {
    'tx-to-tx': {
        'prefix': ['people say', 'i heard that'],
        'sent': [], # toxic
        'conj': ['.'],
        'postfix': ['i hate them too', 'i also hate them']
    },
    'nt-to-tx': {
        'prefix': ['people say', 'i heard that'],
        'sent': [], # non-toxic
        'conj': ['.'],
        'postfix': ['however i hate them', 'i dislike them though']
    },
    'tx-to-nt': {
        'prefix': ['people say', 'i heard that'],
        'sent': [], # toxic
        'conj': ['.'],
        'postfix': ['however i do not hate them', 'regardless i like them']
    },
    'nt-to-nt': {
        'prefix': ['people say', 'i heard that'],
        'sent': [], # non-toxic
        'conj': ['.'],
        'postfix': ['i also do not hate them', 'i do not hate them too']
    }
}

F14_TEMPLATES = {
    'temp1': {
        'prefix': ['I do not think that', 'I do not agree with that', 'I do not believe that'],
        'sent': [], # non-toxic
    }
}

F15_TEMPLATES = {
    'temp1': {
        'prefix': ['I do not think that', 'I do not agree with that', 'I do not believe that'],
        'sent': [], # toxic
    }
}

F16_TEMPLATES = {
    'temp1': {
        'prefix': ['why do you agree with that', 'why do you like to say that'],
        'sent': [], # toxic
        'postfix': ['?']
    }
}

F17_TEMPLATES = {
    'temp1': {
        'prefix': ['i think that', 'i believe that'],
        'sent': [], # toxic
    }
}

F20_TEMPLATES = {
    'temp1': {
        'prefix': ['statements like "'],
        'sent': [], # toxic
        'postfix': ['" are hurtful', '" are harmful', '" are deleterious']
    },
    'temp2': {
        'prefix': ['you should stop saying like "'],
        'sent': [], # toxic
        'postfix': ['"']
    }
}

F21_TEMPLATES = {
    'temp1': {
        'prefix': ['saying that'],
        'sent': [], # toxic
        'postfix': ['shows how misguided you are']
    },
    'temp2': {
        'prefix': ['i cannot believe that'],
        'sent': [], # toxic
    }
}



class TransformOperator:

    def __init__(self,
                 requirements,
                 editor=None
                 ):
        self.editor = editor # checklist.editor.Editor()
        self.capability = requirements['capability']
        self.description = requirements['description']
        self.transform_reqs = requirements.get('transform', None)
        self.transform_func = None
        self.transform_props = None
        if self.transform_reqs is not None:
            self.transform_func = self.transform_reqs.split()[0]
            if len(self.transform_reqs.split())>1:
                self.transform_props = self.transform_reqs.split()[1]
            # end if
        # end if
        
    def transform(self, sents):
        transform_func_map = {
            'template': self.perturb_template
        }
        new_sents = transform_func_map[self.transform_func](sents, self.transform_props)
        return new_sents

    def perturb_template(self, sents, transform_props):
        if transform_props=='f12':
            _sents = self.perturb_template_f12(sents)
        elif transform_props=='f13':
            _sents = self.perturb_template_f13(sents)
        elif transform_props=='f14':
            _sents = self.perturb_template_f14(sents)
        elif transform_props=='f15':
            _sents = self.perturb_template_f15(sents)
        elif transform_props=='f16':
            _sents = self.perturb_template_f16(sents)
        elif transform_props=='f17':
            _sents = self.perturb_template_f17(sents)
        elif transform_props=='f20':
            _sents = self.perturb_template_f20(sents)
        elif transform_props=='f21':
            _sents = self.perturb_template_f21(sents)
        # end if
        return _sents

    def perturb_template_f12(self, sents):
        # Pronoun reference::Hate expressed through reference in subsequent clauses
        nh_sents = [s for s in sents if s[-1]==list(Macros.hs_label_map.keys())[0]]
        hs_sents = [s for s in sents if s[-1]==list(Macros.hs_label_map.keys())[1]]
        f12_hs1_temp = F12_TEMPLATES['tx-to-tx']
        f12_hs2_temp = F12_TEMPLATES['nt-to-tx']
        f12_nh1_temp = F12_TEMPLATES['tx-to-nt']
        f12_nh2_temp = F12_TEMPLATES['nt-to-nt']
        f12_hs1_new_label = list(Macros.hs_label_map.keys())[1]
        f12_hs2_new_label = list(Macros.hs_label_map.keys())[1]
        f12_nh1_new_label = list(Macros.hs_label_map.keys())[0]
        f12_nh2_new_label = list(Macros.hs_label_map.keys())[0]
        results = list()
        result_ind = 0
        for s in hs_sents:
            f12_hs1_temp['sent'].append(s[1])
            f12_nh1_temp['sent'].append(s[1])
        # end for
        for s in nh_sents:
            f12_hs2_temp['sent'].append(s[1])
            f12_nh2_temp['sent'].append(s[1])
        # end for
        
        f12_hs1_word_product = [dict(zip(f12_hs1_temp, v)) for v in product(*f12_hs1_temp.values())]
        f12_hs2_word_product = [dict(zip(f12_hs2_temp, v)) for v in product(*f12_hs2_temp.values())]
        f12_nh1_word_product = [dict(zip(f12_nh1_temp, v)) for v in product(*f12_nh1_temp.values())]
        f12_nh2_word_product = [dict(zip(f12_nh2_temp, v)) for v in product(*f12_nh2_temp.values())]
        
        for wp in f12_hs1_word_product:
            results.append((
                result_ind, " ".join(list(wp.values())), f12_hs1_new_label
            ))
            result_ind += 1
        # end for

        for wp in f12_hs2_word_product:
            results.append((
                result_ind, " ".join(list(wp.values())), f12_hs2_new_label
            ))
            result_ind += 1
        # end for

        for wp in f12_nh1_word_product:
            results.append((
                result_ind, " ".join(list(wp.values())), f12_nh1_new_label
            ))
            result_ind += 1
        # end for

        for wp in f12_nh2_word_product:
            results.append((
                result_ind, " ".join(list(wp.values())), f12_nh2_new_label
            ))
            result_ind += 1
        # end for
        return results

    def perturb_template_f13(self, sents):
        # Pronoun reference::Hate expressed through reference in subsequent sentences
        nh_sents = [s for s in sents if s[-1]==list(Macros.hs_label_map.keys())[0]]
        hs_sents = [s for s in sents if s[-1]==list(Macros.hs_label_map.keys())[1]]
        f13_hs1_temp = F13_TEMPLATES['tx-to-tx']
        f13_hs2_temp = F13_TEMPLATES['nt-to-tx']
        f13_nh1_temp = F13_TEMPLATES['tx-to-nt']
        f13_nh2_temp = F13_TEMPLATES['nt-to-nt']
        f13_hs1_new_label = list(Macros.hs_label_map.keys())[1]
        f13_hs2_new_label = list(Macros.hs_label_map.keys())[1]
        f13_nh1_new_label = list(Macros.hs_label_map.keys())[0]
        f13_nh2_new_label = list(Macros.hs_label_map.keys())[0]
        results = list()
        result_ind = 0
        for s in hs_sents:
            f13_hs1_temp['sent'].append(s[1])
            f13_nh1_temp['sent'].append(s[1])
        # end for
        for s in nh_sents:
            f13_hs2_temp['sent'].append(s[1])
            f13_nh2_temp['sent'].append(s[1])
        # end for
        
        f13_hs1_word_product = [dict(zip(f13_hs1_temp, v)) for v in product(*f13_hs1_temp.values())]
        f13_hs2_word_product = [dict(zip(f13_hs2_temp, v)) for v in product(*f13_hs2_temp.values())]
        f13_nh1_word_product = [dict(zip(f13_nh1_temp, v)) for v in product(*f13_nh1_temp.values())]
        f13_nh2_word_product = [dict(zip(f13_nh2_temp, v)) for v in product(*f13_nh2_temp.values())]
        for wp in f13_hs1_word_product:
            results.append((
                result_ind, " ".join(list(wp.values())), f13_hs1_new_label
            ))
            result_ind += 1
        # end for

        for wp in f13_hs2_word_product:
            results.append((
                result_ind, " ".join(list(wp.values())), f13_hs2_new_label
            ))
            result_ind += 1
        # end for

        for wp in f13_nh1_word_product:
            results.append((
                result_ind, " ".join(list(wp.values())), f13_nh1_new_label
            ))
            result_ind += 1
        # end for

        for wp in f13_nh2_word_product:
            results.append((
                result_ind, " ".join(list(wp.values())), f13_nh2_new_label
            ))
            result_ind += 1
        # end for
        return results

    def perturb_template_f14(self, sents):
        # Negation::Hate expressed using negated positive statement
        results = list()
        result_ind = 0
        new_label = list(Macros.hs_label_map.keys())[1]
        for temp_key in F14_TEMPLATES.keys():
            for s in sents:
                F14_TEMPLATES[temp_key]['sent'].append(s[1])
            # end for
            word_product = [
                dict(zip(F14_TEMPLATES[temp_key], v))
                for v in product(*F14_TEMPLATES[temp_key].values())
            ]
            for wp in word_product:
                results.append((
                    result_ind, " ".join(list(wp.values())), new_label
                ))
                result_ind += 1
            # end for
        # end for
        return results

    def perturb_template_f15(self, sents):
        # Negation::Non-hate expressed using negated hateful statement
        results = list()
        result_ind = 0
        new_label = list(Macros.hs_label_map.keys())[0]
        for temp_key in F15_TEMPLATES.keys():
            for s in sents:
                F15_TEMPLATES[temp_key]['sent'].append(s[1])
            # end for
            word_product = [
                dict(zip(F15_TEMPLATES[temp_key], v))
                for v in product(*F15_TEMPLATES[temp_key].values())
            ]
            for wp in word_product:
                results.append((
                    result_ind, " ".join(list(wp.values())), new_label
                ))
                result_ind += 1
            # end for
        # end for
        return results

    def perturb_template_f16(self, sents):
        # Phrasing::Hate phrased as a question
        results = list()
        result_ind = 0
        new_label = list(Macros.hs_label_map.keys())[1]
        for temp_key in F16_TEMPLATES.keys():
            for s in sents:
                F16_TEMPLATES[temp_key]['sent'].append(s[1])
            # end for
            word_product = [
                dict(zip(F16_TEMPLATES[temp_key], v))
                for v in product(*F16_TEMPLATES[temp_key].values())
            ]
            for wp in word_product:
                results.append((
                    result_ind, " ".join(list(wp.values())), new_label
                ))
                result_ind += 1
            # end for
        # end for
        return results

    def perturb_template_f17(self, sents):
        # Phrasing::Hate phrased as a opinion
        results = list()
        result_ind = 0
        new_label = list(Macros.hs_label_map.keys())[1]
        for temp_key in F17_TEMPLATES.keys():
            for s in sents:
                F17_TEMPLATES[temp_key]['sent'].append(s[1])
            # end for
            word_product = [
                dict(zip(F17_TEMPLATES[temp_key], v))
                for v in product(*F17_TEMPLATES[temp_key].values())
            ]
            for wp in word_product:
                results.append((
                    result_ind, " ".join(list(wp.values())), new_label
                ))
                result_ind += 1
            # end for
        # end for
        return results

    def perturb_template_f20(self, sents):
        # Counter speech::Denouncements of hate that quote it
        results = list()
        result_ind = 0
        new_label = list(Macros.hs_label_map.keys())[0]
        for temp_key in F20_TEMPLATES.keys():
            for s in sents:
                F20_TEMPLATES[temp_key]['sent'].append(s[1])
            # end for
            word_product = [
                dict(zip(F20_TEMPLATES[temp_key], v))
                for v in product(*F20_TEMPLATES[temp_key].values())
            ]
            for wp in word_product:
                results.append((
                    result_ind, " ".join(list(wp.values())), new_label
                ))
                result_ind += 1
            # end for
        # end for
        return results


    def perturb_template_f21(self, sents):
        # Counter speech::Denouncements of hate that make direct reference to it
        results = list()
        result_ind = 0
        new_label = list(Macros.hs_label_map.keys())[0]
        for temp_key in F20_TEMPLATES.keys():
            for s in sents:
                F21_TEMPLATES[temp_key]['sent'].append(s[1])
            # end for
            word_product = [
                dict(zip(F21_TEMPLATES[temp_key], v))
                for v in product(*F21_TEMPLATES[temp_key].values())
            ]
            for wp in word_product:
                results.append((
                    result_ind, " ".join(list(wp.values())), new_label
                ))
                result_ind += 1
            # end for
        # end for
        return results
