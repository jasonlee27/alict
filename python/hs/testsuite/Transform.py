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
from .sentiwordnet.Sentiwordnet import Sentiwordnet
from .Synonyms import Synonyms


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

F12_HS_TEMPLATES = [
    {
        "prefix": ["people say", "I heard that"],
        "sent": [], # toxic
        "conj": ['and'],
        "postfix": ["I hate them too", "I also hate them"]
        "sent_type": ''
    }, {
        "prefix": ["people say", "I heard that"],
        "sent": [], # non-toxic
        "conj": ['but'],
        "postfix": ["I hate them", "I dislike them"]
    }, {
        "prefix": ["people say", "I heard that"],
        "sent": [], # toxic
        "conj": ['but'],
        "postfix": ["[I do not hate them", "I like them"]
    }, {
        "prefix": ["people say", "I heard that"],
        "sent": [], # non-toxic
        "conj": ['and'],
        "postfix": ["I also do not hate them", "I do not hate them too"]
    }
]


class TransformOperator:

    def __init__(self,
                 requirements,
                 editor=None
                 ):
        self.editor = editor # checklist.editor.Editor()
        self.capability = requirements['capability']
        self.description = requirements['description']
        self.transform_reqs = requirements['transform']
        self.transform_func = self.transform_reqs.split()[0]
        self.transform_props = None
        if len(self.transform_reqs.split())>1:
            self.transform_props = self.transform_reqs.split()[1]
        # end if

    def transform(self, sents):
        transform_func_map = {
            'template': self.perturb_template
            'add': self.add,
            'negate': self.negate,
            'srl': self.srl,
            'questionize': self.questionize
        }
        new_sents = transform_func_map[self.transform_func](sents, self.transform_props)
        return new_sents

    # def _change_temporalness_template(self, sents):
    #     # sents: List[(s_i, sentence, label)]
    #     # convert every generate templates into temporal awareness formated templates
    #     # each template keys: sent, values, label
    #     results = list()
    #     res_idx = 0
    #     for sent in sents:
    #         new_sents = list()
    #         word_dict = dict()
    #         label = sent[2]
    #         new_label = None
    #         if label=='positive': # posive previously, but negative now
    #             word_dict = CUR_NEG_TEMPORAL_PHRASE_TEMPLATE
    #             word_dict['sent'] = [f"\"{sent[1]}\","]
    #             new_label = 'negative'
    #         elif label=='negative':
    #             word_dict = CUR_POS_TEMPORAL_PHRASE_TEMPLATE
    #             word_dict['sent'] = [f"\"{sent[1]}\","]
    #             new_label = 'positive'
    #         else:
    #             raise(f"label \"{label}\" is not available")
    #         # end if
    #         word_product = [dict(zip(word_dict, v)) for v in product(*word_dict.values())]
    #         for wp in word_product:
    #             new_sent = " ".join(list(wp.values()))
    #             results.append((f'new_{sent[0]}_{res_idx}', new_sent, new_label, None))
    #             res_idx += 1
    #         # end for
    #     # end for
    #     random.shuffle(results)
    #     return results

    # def add(self, sents, props):
    #     if props=='temporal_awareness':
    #         return self._change_temporalness_template(sents)
    #     # end if
    #     return sents
    
    # def _get_negationpattern_to_wordproduct(self, negation_pattern, value_dict):
    #     results = list()
    #     pos_dict = {
    #         p: value_dict[p]
    #         for p in negation_pattern.split('_')
    #     }
    #     word_product = [dict(zip(pos_dict, v)) for v in product(*pos_dict.values())]
    #     for wp in word_product:
    #         results.append(" ".join(list(wp.values())))
    #     # end for
    #     return results
    
    # def negate(self, sents, negation_pattern):
    #     # sents: List[(s_i, sentence, label)]
    #     from itertools import product
    #     results = list()
    #     if negation_pattern=='AUXBE$':
    #         # negation of negative at the end
    #         negation_pattern = negation_pattern[:-1]
    #         res_idx = 0
    #         for sent in sents:
    #             word_dict = NEG_OF_NEG_AT_THE_END_PHRASE_TEMPLATE
    #             word_dict['sent'] = [f"\"{sent[1]}\","]
    #             label = sent[2]
    #             word_product = [dict(zip(word_dict, v)) for v in product(*word_dict.values())]
    #             for wp in word_product:
    #                 new_sent = " ".join(list(wp.values()))
    #                 new_label = ['positive', 'neutral']
    #                 results.append((f'new_{sent[0]}_{res_idx}', new_sent, new_label, None))
    #                 res_idx += 1
    #             # end for
    #         # end for
    #         random.shuffle(results)
    #         return results
    #     elif negation_pattern=='positive':
    #         # negated of positive with neutral content in the middle
    #         # first, search neutral sentences
    #         positive_sents = [s for s in sents if s[2]=='positive']
    #         random.shuffle(positive_sents)
    #         neutral_sents = [s[1] for s in sents if s[2]=='neutral']
    #         random.shuffle(neutral_sents)
    #         neutral_selected = [f"given {s}," for s in neutral_sents[:3]]
            
    #         word_dict = DISAGREEMENT_PHRASE
    #         word_dict['middlefix'] = neutral_selected
    #         res_idx = 0
    #         for sent in positive_sents:
    #             word_dict['sent'] = [sent[1]]
    #             label = sent[2]
    #             word_product = [dict(zip(word_dict, v)) for v in product(*word_dict.values())]
    #             for wp in word_product:
    #                 new_sent = " ".join(list(wp.values()))
    #                 new_label = 'negative'
    #                 results.append((f'new_{sent[0]}_{res_idx}', new_sent, new_label, None))
    #                 res_idx += 1
    #             # end for
    #         # end for
    #         random.shuffle(results)
    #         return results
    #     # end if

    #     # negated neutral should still be neutral &
    #     # negated negative should be positive or neutral
    #     # search sents by tag of pos organization
    #     prefix_pat, postfix_pas = '',''
    #     if negation_pattern.startswith('^'):
    #         negation_pattern = negation_pattern[1:]
    #         prefix_pat = '^'
    #     # end if

    #     res_idx = 0
    #     for pat in self._get_negationpattern_to_wordproduct(negation_pattern, WORD2POS_MAP):
    #         _pat = prefix_pat+pat
    #         for sent in sents:
    #             if re.search(_pat, sent[1]):
    #                 new_sent = re.sub(_pat, f"{pat} not", sent[1])
    #                 label = sent[2]
    #                 new_label = None
    #                 if label=='negative':
    #                     new_label = ['positive', 'neutral']
    #                 elif label=='neutral':
    #                     new_label = 'neutral'
    #                 # end if
    #                 results.append((f'new_{sent[0]}_{res_idx}', new_sent, new_label, None))
    #                 res_idx += 1
    #             # end if
    #         # end for
    #     # end for
    #     random.shuffle(results)
    #     return results

    # def srl(self, sents, na_param):
    #     positive_sents = [s[1] for s in sents if s[2]=='positive']
    #     random.shuffle(positive_sents)
    #     negative_sents = [s[1] for s in sents if s[2]=='negative']
    #     random.shuffle(negative_sents)
        
    #     word_dict = SRL_PHASE_TEMPLATE
    #     res_idx = 0
    #     results = list()
    #     for sent in sents:
    #         label = sent[2]
    #         if label=='positive':
    #             word_dict['sent1'] = [f"\"{s}\"," for s in negative_sents[:3]]
    #         elif label=='negative':
    #             word_dict['sent1'] = [f"\"{s}\"," for s in positive_sents[:3]]
    #         # end if
    #         word_dict['sent2'] = [f"\"{sent[1]}\""]
            
    #         word_product = [dict(zip(word_dict, v)) for v in product(*word_dict.values())]
    #         for wp in word_product:
    #             new_sent = " ".join(list(wp.values()))
    #             results.append((f'new_{sent[0]}_{res_idx}', new_sent, label, None))
    #             res_idx += 1
    #         # end for
    #     # end for
    #     random.shuffle(results)
    #     return results

    # def questionize(self, sents, answer):
    #     word_dict = QUESTIONIZE_PHRASE_TEMPLATE
    #     res_idx = 0
    #     results = list()
    #     for sent in sents:
    #         word_dict['sent'] = [f"\"{sent[1]}\"?"]
    #         word_dict['answer'] = [answer]
    #         label = sent[2]
    #         if label=='positive' and answer=='yes':
    #             new_label = 'positive'
    #         elif label=='positive' and answer=='no':
    #             new_label = 'negative'
    #         elif label=='negative' and answer=='yes':
    #             new_label = 'negative'
    #         elif label=='negative' and answer=='no':
    #             new_label = ['positive', 'neutral']
    #         # end if
    #         word_product = [dict(zip(word_dict, v)) for v in product(*word_dict.values())]
    #         for wp in word_product:
    #             new_sent = " ".join(list(wp.values()))
    #             results.append((f'new_{sent[0]}_{res_idx}', new_sent, new_label, None))
    #             res_idx += 1
    #         # end for
    #     # end for
    #     random.shuffle(results)
    #     return results


    def perturb_template_f12(self, sents):
        nh_sents = [s if s['label']==Macros.hs_label_map.keys()[0] for s in sents]
        hs_sents = [s if s['label']==Macros.hs_label_map.keys()[1] for s in sents]
        f12_hs1_new_label = Macros.hs_label_map.keys()[1]
        f12_hs2_new_label = Macros.hs_label_map.keys()[1]
        f12_nh1_new_label = Macros.hs_label_map.keys()[0]
        f12_nh2_new_label = Macros.hs_label_map.keys()[0]
        results = list()
        for s in hs_sents:
            F12_HS_TEMPLATES['sent'].append(s['sent'])
            F12_NH_TEMPLATE1['sent'].append(s['sent'])
        # end for
        for s in nh_sents:
            F12_HS_TEMPLATE2['sent'].append(s['sent'])
            F12_NH_TEMPLATE2['sent'].append(s['sent'])
        # end for

        f12_hs1_word_product = [dict(zip(F12_HS_TEMPLATE1, v)) for v in product(*F12_HS_TEMPLATE1.values())]
        f12_hs2_word_product = [dict(zip(F12_HS_TEMPLATE2, v)) for v in product(*F12_HS_TEMPLATE2.values())]
        f12_nh1_word_product = [dict(zip(F12_NH_TEMPLATE1, v)) for v in product(*F12_NH_TEMPLATE1.values())]
        f12_nh2_word_product = [dict(zip(F12_NH_TEMPLATE2, v)) for v in product(*F12_NH_TEMPLATE2.values())]
        for wp in f12_hs1_word_product:
            results.append({
                'func': s['func'],
                'sent': " ".join(list(wp.values())),
                'label': f12_hs1_new_label
            })
        # end for

        for wp in f12_hs2_word_product:
            results.append({
                'func': s['func'],
                'sent': " ".join(list(wp.values())),
                'label': f12_hs2_new_label
            })
        # end for

        for wp in f12_nh1_word_product:
            results.append({
                'func': s['func'],
                'sent': " ".join(list(wp.values())),
                'label': f12_nh1_new_label
            })
        # end for

        for wp in f12_nh2_word_product:
            results.append({
                'func': s['func'],
                'sent': " ".join(list(wp.values())),
                'label': f12_nh2_new_label
            })
        # end for
        return results


