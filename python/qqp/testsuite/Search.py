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

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..requirement.Requirements import Requirements
from .Synonyms import Synonyms
# from .sentiwordnet.Sentiwordnet import Sentiwordnet

# get name and location data
basic = Utils.read_json(Macros.dataset_dir / 'checklist' / 'lexicons' / 'basic.json')
names = Utils.read_json(Macros.dataset_dir / 'checklist' / 'names.json')
name_set = { x:set(names[x]) for x in names }
NAME_LOC_DICT = {
    'name': names,
    'name_set': name_set,
    'city': basic['city'],
    'country': basic['country'],
}

random.seed(27)

class SearchOperator: 
    
    def __init__(self, requirements):
        self.capability = requirements['capability']
        self.description = requirements['description']
        self.search_reqs_list = requirements['search']
        self.search_pairs = requirements['search_pairs']
        self.search_method = {
            "length": self.search_by_len,
            "include": self.search_by_include,
            "exclude": self.search_by_exclude,
            "label": self.search_by_label,
        }

    def search(self, sents):
        # sents: question sentences or pairs of questions
        # issentpair: indicator if sents are question sentences or pairs of questions
        selected = list()
        for search_reqs in self.search_reqs_list:
            _sents = sents.copy()
            for op, param in search_reqs.items():
                if len(_sents)>0:
                    _sents = self.search_method[op](_sents, search_reqs)
                # end if
            # end for
            selected.extend(_sents)
        # end for
        return list(set(selected))

    def search_by_len(self, sents, search_reqs):
        param = search_reqs["length"]
        match = re.search(r"([<>]=?|==)(\d+)", param)
        op, _len = match.groups()
        selected = list()
        if self.search_pairs:
            _sents = [(s_i,Utils.tokenize(s1),Utils.tokenize(s2),l) for s_i, s1, s2, l in sents]
            for s_i, s1, s2, l in _sents:
                if s1[-1]=="." and (eval(f"{len(s1)-1}{op}{_len}")):
                    if s2[-1]=="." and (eval(f"{len(s2)-1}{op}{_len}")):
                        selected.append((s_i,Utils.detokenize(s1),Utils.detokenize(s2),l))
                    elif s2[-1]!="." and (eval(f"{len(s2)}{op}{_len}")):
                        selected.append((s_i,Utils.detokenize(s1),Utils.detokenize(s2),l))
                    # end if
                elif s1[-1]!="." and (eval(f"{len(s1)}{op}{_len}")):
                    if s2[-1]=="." and (eval(f"{len(s2)-1}{op}{_len}")):
                        selected.append((s_i,Utils.detokenize(s1),Utils.detokenize(s2),l))
                    elif s2[-1]!="." and (eval(f"{len(s2)}{op}{_len}")):
                        selected.append((s_i,Utils.detokenize(s1),Utils.detokenize(s2),l))
                    # end if
                # end if
            # end for
        else:
            _sents = [(s_i,Utils.tokenize(s)) for s_i, s in sents]
            for s_i, s in _sents:
                if s[-1]=="." and (eval(f"{len(s)-1}{op}{_len}")):
                    selected.append((s_i,Utils.detokenize(s)))
                elif s[-1]!="." and (eval(f"{len(s)}{op}{_len}")):
                    selected.append((s_i,Utils.detokenize(s)))
                # end if
            # end for
        # end if
        return selected

    def search_by_label(self, sents, search_reqs):
        assert self.search_pairs
        # label: 0: different, 1: same
        label = search_reqs["label"]
        if label==0 or label==1:
            _sents = [sent for sent in sents if sent[-1]==label]
            return _sents
        else:
            raise(f"Label ({label}) is not available for QQP")
        # end if

    def _search_by_synonym_existence(self, sents, isinclude=True):
        # sents: (s_i, tokenized sentence, label)
        nlp = spacy.load("en_core_web_sm")
        selected = list()
        for sent in sents:
            doc1 = nlp(Utils.detokenize(sent[1]))
            synonyms = None
            for t in doc:
                synonyms = Synonyms.get_synosyms(nlp, t, t.pos_)
                if any(synonyms) and isinclde:
                    selected.append(sent)
                    break
                # end if
            # end for
            if synonyms is None and not isinclude:
                selected.append(sent)
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
            if word_group=="synonym":
                selected = self._search_by_synonym_existence(sents)
            # end if
        else:
            word_or_list = word_cond.split("|")
            for sent in sents:
                isfound = [True for t in sent[1] if t in word_or_list]
                if any(isfound):
                    selected.append(sent)
                # end if
            # end for
        # end if
        return selected
    
    def _search_by_pos_include(self, sents, pos_cond):
        # sents: (s_i, tokenized sentence, label)
        # pos_cond: e.g. adj, verb, noun
        pos_map = {'adj': 'JJ', 'verb': 'VB', 'noun': 'NN'}
        nlp = spacy.load("en_core_web_sm")
        selected = list()
        for sent in sents:
            doc = nlp(Utils.detokenize(sent[1]))
            for t in doc:
                if t.tag_.startswith(pos_map[pos_cond]):
                    selected.append(sent)
                    break
                # end if
            # end for
        # end for
        return selected
        
    def search_by_include(self, sents, search_reqs):
        params = search_reqs["include"]
        if type(params)==dict:
            params = [params]
        # end if
        _sents = sents.copy()
        if self.search_pairs:
            _sents = [(s_i,Utils.tokenize(s1),Utils.tokenize(s2),l) for s_i, s1, s2, l in _sents]
        else:
            _sents = [(s_i,Utils.tokenize(s)) for s_i, s in _sents]
        # end if
        selected_indices = set()
        for param in params:
            word_include = param["word"]
            pos_include = param["POS"]
            
            if word_include is not None:
                for w in word_include: # AND relationship
                    _sents = self._search_by_word_include(_sents, w)
                # end for
            # end if
        
            if pos_include is not None:
                temp_sents = list()
                for p in pos_include:
                    _sents = self._search_by_pos_include(_sents, p)
                # end for
            # end if
            
            for sent in _sents:
                selected_indices.add(sent[0])
            # end for
        # end for
        result = list()
        if self.search_pairs:
            result = [(s[0],Utils.detokenize(s[1]),Utils.detokenize(s[2]),s[3]) for s in _sents if s[0] in selected_indices]
        else:
            result = [(s[0],Utils.detokenize(s[1])) for s in _sents if s[0] in selected_indices]
        # end if
        return result

    def _search_by_word_exclude(self, sents, word_cond):
        search = re.search("\<([^\<\>]+)\>", word_cond)
        selected = list()
        if search:
            # if word condition searches a specific group of words
            # such as "name of person, location"
            word_group = search.group(1)
            if word_group=="synonym":
                selected = self._search_by_synonym_existence(sents, isinclude=False)
            # end if
        else:
            word_or_list = word_cond.split("|")
            for sent in sents:
                isfound = [True for t in sent[1] if t in word_or_list]
                if not any(isfound):
                    selected.append(sent)
                # end if
            # end for
        # end if
        return selected

    def _search_by_pos_exclude(self, sents, pos_cond):
        # sents: (s_i, tokenized sentence, label)
        # pos_cond: e.g. adj, verb, noun
        pos_map = {'adj': 'JJ', 'verb': 'VB', 'noun': 'NN'}
        nlp = spacy.load("en_core_web_sm")
        selected = list()
        for sent in sents:
            pos_found = False
            doc = nlp(Utils.detokenize(sent[1]))
            for t in doc:
                if t.tag_.startswith(pos_map[pos_cond]):
                    pos_found = True
                    break
                # end if
            # end for
            if not pos_found:
                selected.append(sent)
            # end if
        # end for
        return selected

    def search_by_exclude(self, sents, search_reqs):
        params = search_reqs["exclude"]
        _sents = sents.copy()
        if self.search_pairs:
            _sents = [(s_i,Utils.tokenize(s1),Utils.tokenize(s2),l) for s_i, s1, s2, l in sents]
        else:
            _sents = [(s_i,Utils.tokenize(s)) for s_i, s in sents]
        # end if
        selected_indices = set()
        if type(params)==dict:
            params = [params]
        # end if
        for param in params:
            word_exclude = param["word"]
            pos_exclude = param["POS"]
            if word_exclude is not None:
                for w in word_exclude:
                    _sents = self._search_by_word_exclude(_sents, w)
                # end for
            # end if

            if pos_exclude is not None:
                for p in pos_exclude:
                    _sents = self._search_by_pos_exclude(_sents, p)
                # end for
            # end if

            for sent in _sents:
                selected_indices.add(sent[0])
            # end for
        # end for
        
        result = list()
        if self.search_pairs:
            result = [(s_i,Utils.detokenize(s1),Utils.detokenize(s2),l) for s_i, s1, s2, l in _sents if s_i in selected_indices]
        else:
            result = [(s_i,Utils.detokenize(s)) for s_i, s in _sents if s_i in selected_indices]
        # end if
        return result

    
class Qqp:

    @classmethod
    def get_sents(cls, qqp_file, issentpair):
        # sents: all_qs, qpairs, labels
        all_qs, qpairs, labels = Utils.read_tsv(qqp_file)
        if issentpair:
            return [(q_i,q[0],q[1],l) for q_i,(q,l) in enumerate(zip(qpairs, labels))]
        # end if
        return [(q_i,q) for q_i, q in enumerate(all_qs)]
    
    @classmethod
    def search(cls, req):
        # sents: Dict
        req_obj = SearchOperator(req)
        sents = cls.get_sents(Macros.qqp_valid_file, req_obj.search_pairs)
        selected = None
        if len(req_obj.search_reqs_list)>0:
            selected = sorted([s for s in req_obj.search(sents)], key=lambda x: x[0])
        else:
            selected = sents
        # end if
        random.shuffle(selected)
        return selected


# class ChecklistTestsuite:

#     @classmethod
#     def get_sents(cls, testsuite_file):
#         tsuite, tsuite_dict = read_testsuite(testsuite_file)
#         sents, raw_labels = list(), list()
#         for test_name in test_names:
#             # sents: List of sent
#             # label: 0(neg), 1(neu) and 2(pos)
#             sents.extend(tsuite.tests[test_name].data) 
#             raw_labels.extend(tsuite.tests[test_name].labels)
#         # end for        
#         labels = dict()
#         for s_i, s in enumerate(raw_labels):
#             if s=='0':
#                 labels[s_i] = "negative"
#             elif s=='1':
#                 labels[s_i] = "neutral"
#             else:
#                 labels[s_i] = "positive"
#             # end if
#         #end for
#         return [(s_i, s, labels[s_i]) for s_i, s in enumerate(sents)]
    
#     @classmethod
#     def search(cls, req):
#         # sent: (index, sentence)
#         # label: (index, label score)
#         sents = cls.get_sents(Macros.checklist_sa_dataset_file)
#         req_obj = SearchOperator(req)
#         selected = sorted([(s[0],s[1],s[2]) for s in req_obj.search(sents)], key=lambda x: x[0])
#         random.shuffle(selected)
#         return selected

    
class Search:

    SEARCH_FUNC = {        
        Macros.datasets[0]: Qqp.search,
        Macros.datasets[1]: None
    }

    @classmethod
    def search_qqp(cls, requirements, dataset):
        func = cls.SEARCH_FUNC[dataset]
        for req in requirements:
            selected = func(req)
            yield {
                "requirement": req,
                "selected_inputs": selected
            }
        # end for
        return


