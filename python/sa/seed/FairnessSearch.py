# This script searches inputs in datasets that meet requirements

from typing import *
from pathlib import Path

import re, os
import sys
import json
import spacy
import random
import checklist
import numpy as np

from itertools import product
from checklist.test_types import MFT, INV, DIR

from ..requirement.Requirements import Requirements
from .Transform import TransformOperatorForFairness

from ...hs.seed.hatecheck.Hatecheck import Hatecheck

from ..utils.Macros import Macros
from ..utils.Utils import Utils


# identity words obtained from hatecheck
IDENTITY_GROUPS: Dict = Hatecheck.get_placeholder_values()

# requirement for fairness linguistic capability
FAIRNESS_REQ = {
    'capability': 'Fairness',
    'description': 'Switching identity group should not change predictions',
    'search': [
        {
            'include': {
                'word': ['<hatecheck_identity>'],
                'POS': None
            }
        },
        {
            'include': {
                'word': ['he'],
                'POS': None
            }
        },
        {
            'include': {
                'word': ['his'],
                'POS': None
            }
        },
        {
            'include': {
                'word': ['him'],
                'POS': None
            }
        },
        {
            'include': {
                'word': ['she'],
                'POS': None
            }
        },
        {
            'include': {
                'word': ['her'],
                'POS': None
            }
        },
        {
            'include': {
                'word': ['hers'],
                'POS': None
            }
        },
        {
            'include': {
                'word': ['they'],
                'POS': None
            }
        },
        {
            'include': {
                'word': ['their'],
                'POS': None
            }
        },
        {
            'include': {
                'word': ['them'],
                'POS': None
            }
        },
        {
            'include': {
                'word': ['you'],
                'POS': None
            }
        },
        {
            'include': {
                'word': ['your'],
                'POS': None
            }
        },
        {
            'include': {
                'word': ['yours'],
                'POS': None
            }
        }
    ],
    "expansion": ["neutral"],
    'transform': 'replace pronouns_with_<hatecheck_identity>',
    'transform_req': None
}

class SearchOperator: 
    
    def __init__(
        self, 
        requirements
    ):
        self.capability = requirements['capability']
        self.description = requirements['description']
        self.search_reqs_list = requirements['search']
        self.search_method = {
            'include': self.search_by_include,
            'label': self.search_by_label
        }

    def search(
        self, 
        sents
    ):
        selected = list()
        if self.search_reqs_list is None:
            return sents
        # end if
        for search_reqs in self.search_reqs_list:
            _sents = sents.copy()
            for op, param in search_reqs.items():
                if len(_sents)>0:
                    _sents = self.search_method[op](_sents, search_reqs)
                # end if
            # end for
            for s in _sents:
                if s not in selected:
                    selected.append(s)
                # end if
            # end for
        # end for
        return selected

    def search_by_label(
        self, 
        sents, 
        search_reqs
    ):
        label = search_reqs['label']
        return [
            (s_i,s[1],s[2],s[3])
            for s_i, s in enumerate(sents) 
            if label==s[2]
        ]

    def _search_by_hatecheck_identity_include(
        self, 
        sents
    ):
        selected = list()
        for key in IDENTITY_GROUPS.keys():
            identity_group_words = IDENTITY_GROUPS[key]
            for w in identity_group_words:
                for s in sents:
                    if (w in s[1] or \
                        w.capitalize() in s[1]) and \
                        s not in selected:
                        selected.append(s)
                    # end if
                # end for
            # end for
        # end for
        return selected

    def _search_by_specific_word_include(
        self,
        sents,
        word_cond
    ):
        selected = list()
        for s in sents:
            if (word_cond in s['tokens'] or \
                word_cond.capitalize() in s['tokens']) and \
                s not in selected:
                selected.append(s)
            # end if
        # end for
        return selected

    def _search_by_word_include(self, sents, word_cond):
        search = re.search("\<([^\<\>]+)\>", word_cond)
        selected = list()
        if search:
            # if word condition searches a specific group of words
            # such as "name of person, location"
            word_group = search.group(1)
            word_list = list()
            if word_group=="hatecheck_identity":
                selected = self._search_by_hatecheck_identity_include(sents)
            else:
                selected = self._search_by_specific_word_include(sents, word_cond)
            # end if
        else:
            selected = [sent for sent in sents if word_cond in sent[1]]
        # end if
        return selected
        
    def search_by_include(self, sents, search_reqs):
        params = search_reqs["include"]
        if type(params)==dict:
            params = [params]
        # end if
        _sents = sents.copy()
        if len(sents[0])==4:
            _sents = [(s_i,Utils.tokenize(s),l,sc) for s_i, s, l, sc in _sents]
        else:
            _sents = [(s_i,Utils.tokenize(s),l) for s_i, s, l in _sents]
        # end if
        selected_indices = list()
        for param in params:
            word_include = param.get('word', None)
            tpos_include = param.get('POS', None)
            
            if word_include is not None:
                for w in word_include: # AND relationship
                    _sents = self._search_by_word_include(_sents, w)
                # end for
            # end if

            for sent in _sents:
                if sent[0] not in selected_indices:
                    selected_indices.append(sent[0])
                # end if
            # end for
        # end for
        result = list()
        if len(sents[0])==4:
            result = [
                (s[0],Utils.detokenize(s[1]),s[2],s[3])
                for s in _sents if s[0] in selected_indices
            ]
        else:
            result = [
                (s[0],Utils.detokenize(s[1]),s[2])
                for s in _sents if s[0] in selected_indices
            ]
        # end if
        return result


class Sst:

    @classmethod
    def get_sents(cls, sent_file, label_file, phrase_dict_file):
        # sents: List of [sent_index, sent]
        sents = [
            (l.split("\t")[0].strip(),l.split("\t")[1].strip()) 
            for l in Utils.read_txt(sent_file)[1:]
        ]
        label_scores = {
            l.split("|")[0].strip():l.split("|")[1].strip() # id: score
            for l in Utils.read_txt(label_file)[1:]
        }
        phrases = {
            l.split("|")[0].strip(): l.split("|")[1].strip() # phrase: id
            for l in Utils.read_txt(phrase_dict_file)
        }
        result = list()
        for s_i, s in sents:
            s = Utils.replace_non_english_letter(s)
            if s in phrases.keys():
                phrase_id = phrases[s]
                label_score = float(label_scores[phrase_id])
                label = "neutral"
                if label_score<=0.4:
                    label = "negative"
                elif label_score>0.6:
                    label = "positive"
                # end if
                result.append((s_i, s, label, label_score))
            # end if
        # end for
        return result
    
    @classmethod
    def search(cls, req):
        sents = cls.get_sents(
            Macros.sst_datasent_file, 
            Macros.sst_label_file, 
            Macros.sst_dict_file
        )
        req_obj = SearchOperator(req)
        if req_obj.search_reqs_list is not None:
            selected_sents = req_obj.search(sents)
            selected = list()
            for s in selected_sents:
                if len(s)==4:
                    selected.append((s[0],s[1],s[2],s[3]))
                else:
                    selected.append((s[0],s[1],s[2]))
                # end if
            # end for
            selected = sorted(selected, key=lambda x: x[0])
        else:
            if len(sents[0])==4:
                selected = [(s_i,Utils.tokenize(s),l,sc) for s_i, s, l, sc in sents]
                selected = [(s_i,Utils.detokenize(s),l,sc) for s_i, s, l, sc in selected]
            else:
                selected = [(s_i,Utils.tokenize(s),l) for s_i, s, l in sents]
                selected = [(s_i,Utils.detokenize(s),l) for s_i, s, l in selected]
            # end if
        # end if
        random.shuffle(selected)
        return selected


class FairnessSearch:

    SEARCH_FUNC = {
        Macros.sa_task : {
            Macros.datasets[Macros.sa_task][0]: Sst.search
        },
        Macros.mc_task : {},
        Macros.qqp_task : {}
    }

    @classmethod
    def search_sentiment_analysis_for_fairness(cls, req, dataset):
        func = cls.SEARCH_FUNC[Macros.sa_task][dataset]
        selected = func(req)
            
        # if req["transform"] is not None and \
        #    dataset!=Macros.datasets[Macros.sa_task][1]:
        #     transform_obj = TransformOperatorForFairness(req)
        #     selected = transform_obj.transform(selected)
        # # end if
        return {
            "requirement": req,
            "selected_inputs": selected
        }
