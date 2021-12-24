# This script searches inputs in datasets that meet requirements

from typing import *

import re, os
import sys
import json
import random

from nltk.tokenize import word_tokenize as tokenize

from pathlib import Path

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..requirement.Requirements import Requirements
from .sentiwordnet.Sentiwordnet import Sentiwordnet


# get pos/neg/neu words from SentiWordNet
SENT_WORDS = Sentiwordnet.get_sent_words()
SENT_DICT = {
    "positive_adj": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="adj" and SENT_WORDS[w]["label"]=="positive"],
    "negative_adj": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="adj" and SENT_WORDS[w]["label"]=="negative"],
    "neutral_adj": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="adj" and SENT_WORDS[w]["label"]=="pure neutral"],
    "positive_verb": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="verb" and SENT_WORDS[w]["label"]=="positive"],
    "negative_verb": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="verb" and SENT_WORDS[w]["label"]=="negative"],
    "neutral_verb": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="verb" and SENT_WORDS[w]["label"]=="pure neutral"],
    "positive_noun": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="noun" and SENT_WORDS[w]["label"]=="positive"],
    "negative_noun": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="noun" and SENT_WORDS[w]["label"]=="negative"],
    "neutral_noun": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="noun" and SENT_WORDS[w]["label"]=="pure neutral"]
}
random.seed(27)

class SearchOperator: 
    
    def __init__(self, requirements):
        self.capability = requirements['capability']
        self.description = requirements['description']
        self.search_reqs_list = requirements['search']
        self.search_method = {
            "length": self.search_by_len,
            "include": self.search_by_include,
            "exclude": self.search_by_exclude,
            "label": self.search_by_label
            # "replace": cls.search_by_replacement,
            # "add": cls.search_by_add,
            # "remove": cls.search_by_remove
        }
        self.transform_methos = dict() # TODO

    def search(self, sents):
        selected = list()
        for search_reqs in self.search_reqs_list:
            _sents = sents.copy()
            for op, param in search_reqs.items():
                _sents = self.search_method[op](_sents, search_reqs)
            # end for
            selected.extend(_sents)
        # end for
        return list(set(selected))

    def search_by_len(self, sents, search_reqs):
        param = search_reqs["length"]
        match = re.search(r"([<>]=?|==)(\d+)", param)
        op, _len = match.groups()
        _sents = [(s_i,tokenize(s),l) for s_i, s, l in sents]
        return [(s_i," ".join(s),l) for s_i, s, l in _sents if len(s) < int(_len)]

    def search_by_label(self, sents, search_reqs):
        label = search_reqs["label"]
        if label=="neutral" or label=="positive" or label=="negative":
            _sents = [(s_i,s,l) for s_i, s, l in sents if l==label]
        # elif label.startswith("same as"):
        #     print(label)
        #     match = re.search(r"same as\s?(include|exclude)?\s?(POS|word)?", label)
        #     in_or_ex_clude, search_key = match.groups()
        #     if in_or_ex_clude is None:
        #         in_or_ex_clude = "include"
        #     # end if
        #     poss = search_reqs[in_or_ex_clude][search_key]
        #     _sents = list()
        #     for pos in poss:
        #         pos_sentiment = pos.split()[0]
        #         _sents.extend([(s_i,s,l) for s_i, s, l in sents if l==pos_sentiment])
        #     # end for
        else:
            return sents
        # end if
        return _sents

    def _search_by_word_include(self, sents, word_cond):
        pass

    def _search_by_pos_include(self, sents, cond_key, cond_number):
        # sents: (s_i, tokenizes sentence, label)
        target_words = SENT_DICT[cond_key]
        selected = list()
        for s_i, s, l in sents:
            found_w = list()
            for w in s:
                if w.lower() in target_words:
                    found_w.append(w)
                # end if
            # end for
            if cond_number>0 and len(found_w)==cond_number:
                selected.append((s_i, s, l))
            elif cond_number<0 and len(found_w)>0:
                selected.append((s_i, s, l))
            # end if
        # end for
        return selected
        
    def search_by_include(self, sents, search_reqs):
        _sents = sents.copy()
        _sents = [(s_i, tokenize(s), l) for s_i, s, l in sents]
        params = search_reqs["include"]
        if type(param)==dict:
            params = [params]
        # end if
        selected_indices = list()
        for param in params:
            word_include = param["word"]
            tpos_include = param["POS"]
            if word_include is not None:
                for w in word_include:
                    _sents = [(s_i, s, l) for s_i, s, l in _sents if w in s]
                # end for
            # end if
        
            if tpos_include is not None:
                temp_sents = list()
                for cond in tpos_include:
                    match = re.search(r"(\d+)?\s?(positive|negative|neutral)?\s?(adj|noun|verb)?(s)?", cond)
                    num, sentiment, pos, is_plural = match.groups()
                    if pos is None: raise("Tag of POS is not valid!")
                    num = -1
                    if num is None and not is_plural:
                        num = 1
                    # end if
                
                    if sentiment is None:
                        for _sentiment in ["neutral","positive","negative"]:
                            temp_sents.extend(self._search_by_pos_include(_sents, f"{_sentiment}_{pos}", num))
                        # end for
                    else:
                        temp_sents = self._search_by_pos_include(_sents, f"{sentiment}_{pos}", num)
                    # end if
                    _sents = temp_sents
                # end for
            # end if
            for s_i, _, _ in _sents:
                if s_i not in selected_indices:
                    selected_indices.append(s_i)
                # end if
            # end for
        # end for
        return [(s_i," ".join(s),l) for s_i, s, l in selected is s_i in selected_indices]

    def _search_by_exclude(self, sents, cond_key):
        # sents: (s_i, tokenizes sentence, label)
        target_words = SENT_DICT[cond_key]
        selected = list()
        for s_i, s, l in sents:
            found_w = list()
            for w in s:
                if w.lower() in target_words:
                    found_w.append(w)
                # end if
            # end for
            if len(found_w)==0:
                selected.append((s_i, s, l))
            # end if
        # end for
        return selected

    def search_by_exclude(self, sents, search_reqs):
        param = search_reqs["exclude"]
        _sents = sents.copy()
        _sents = [(s_i, tokenize(s), l) for s_i, s, l in sents]
        params = search_reqs["include"]
        selected_indices = list()
        if type(param)==dict:
            params = [params]
        # end if
        for param in params:

            
            word_exclude = param["word"]
            tpos_exclude = param["POS"]
            if word_exclude is not None:
                for w in word_exclude:
                    _sents = [(s_i, s, l) for s_i, s, l in _sents if w not in s]
                # end for
            # end if
        
            if tpos_exclude is not None:
                temp_sents = list()
                for cond in tpos_exclude:
                    match = re.search(r"(positive|negative|neutral)?\s?(adj|noun|verb)?(s)?", cond)
                    sentiment, pos, is_plural = match.groups()
                    if pos is None: raise("Tag of POS is not valid!")
                
                    if sentiment is None:
                        for _sentiment in ["neutral","positive","negative"]:
                            temp_sents.extend(self._search_by_exclude(_sents, f"{_sentiment}_{pos}"))
                        # end for
                    else:
                        temp_sents = self._search_by_exclude(_sents, f"{sentiment}_{pos}")
                    # end if
                    _sents = temp_sents
                # end for
            # end if

            for s_i, _, _ in _sents:
                if s_i not in selected_indices:
                    selected_indices.append(s_i)
                # end if
            # end for
        # end for
        return [(s_i," ".join(s),l) for s_i, s, l in selected is s_i in selected_indices]


class TransformOperator:

    def __init__(self, requirements):
        self.capability = requirements['capability']
        self.description = requirements['description']
        self.transform_reqs = requirements['transform']
        self.transform_methods = {
            "get": self.extract,
            # "add": cls.search_by_add,
            # "remove": cls.search_by_remove
        }
        # print(f"{self.capability}: {self.description}")
        

    def replace(self, sents, replace_rules: str):
        replace_from, replace_to = replace_rule.split('->')
        replace_from_search = re.search(r"(positive|negative|neutral)?\::?(adj|noun|verb)?(s)?", cond)
        if replace_from in ["neutral","positive","negative"] and \
           replace_to in ["neutral","positive","negative"]:
            # SA task
            pos_from = replace_from.split('::')
            
            for sent in sents:
                replace_word_indices = find_words_w_sentiment 
            
        # end if
        return sents
        # sents_res = list()
        # for req_key in self.transform_reqs.keys():
        #     sents = self.transform_methods[req_key](sents, self.transform_reqs[req_key])
        # # end for
        # return sents

    def extract(self, sents, cond_key):
        pass
        # _sents = list()
        # for s in sents:
            
        # # end for
        # return _sents

class Sst:

    @classmethod
    def get_sents(cls, sent_file, label_file):
        # sents: List of [sent_index, sent]
        sents = [tuple(l.split("\t")) for l in Utils.read_txt(sent_file)[1:]]
        label_scores = [tuple(l.split("|")) for l in Utils.read_txt(label_file)[1:]]
        labels = dict()
        for s_i, s in label_scores:
            s = float(s)
            labels[s_i] = "neutral"
            if s<=0.4:
                labels[s_i] = "negative"
            elif s>0.6:
                labels[s_i] = "positive"
            # end if
        #end for
        return [(s_i,s,labels[s_i]) for s_i, s in sents]
    
    @classmethod
    def search(cls, req):
        # sent: (index, sentence)
        # label: (index, label score)
        sents = cls.get_sents(Macros.sst_datasent_file, Macros.sst_label_file)
        req_obj = SearchOperator(req)
        selected = sorted([(s[0],s[1].strip()[:-1],s[2]) if s[1].strip()[-1]=="." else (s[0],s[1].strip(),s[2]) for s in req_obj.search(sents)], key=lambda x: x[0])
        random.shuffle(selected)
        return selected


class ChecklistTestsuite:

    @classmethod
    def get_sents(cls, testsuite_file):
        tsuite, tsuite_dict = read_testsuite(testsuite_file)
        sents, raw_labels = list(), list()
        for test_name in test_names:
            # sents: List of sent
            # label: 0(neg), 1(neu) and 2(pos)
            sents.extend(tsuite.tests[test_name].data) 
            raw_labels.extend(tsuite.tests[test_name].labels)
        # end for        
        labels = dict()
        for s_i, s in enumerate(raw_labels):
            if s=='0':
                labels[s_i] = "negative"
            elif s=='1':
                labels[s_i] = "neutral"
            else:
                labels[s_i] = "positive"
            # end if
        #end for
        return [(s_i, s, labels[s_i]) for s_i, s in enumerate(sents)]
    
    @classmethod
    def search(cls, req):
        # sent: (index, sentence)
        # label: (index, label score)
        sents = cls.get_sents(Macros.checklist_sa_dataset_file)
        req_obj = SearchOperator(req)
        selected = sorted([(s[0],s[1].strip()[:-1],s[2]) if s[1].strip()[-1]=="." else (s[0],s[1].strip(),s[2]) for s in req_obj.search(sents)], key=lambda x: x[0])
        random.shuffle(selected)
        return selected

    
class DynasentRoundOne:

    @classmethod
    def get_sents(cls, src_file):
        sents = list()
        sent_i = 0
        with open(yelp_src_filename) as f:
            for line in f:
                d = json.loads(line)
                sents.append((sent_i, d['sentence'], d['gold_label']))
                sent_i += 1
            # end for
        # end with
        return [(s_i,s,labels[s_i]) for s_i, s in sents]

    @classmethod
    def search(cls, req):
        # sent: (index, sentence)
        # label: (index, label score) 
        sents = cls.get_labels(Macros.dyna_r1_test_src_file)
        req_obj = SearchOperator(req)
        selected = sorted([(s[0],s[1].strip()[:-1],s[2]) if s[1].strip()[-1]=="." else (s[0],s[1].strip(),s[2]) for s in req_obj.search(sents)], key=lambda x: x[0])
        random.shuffle(selected)
        return selected


class Search:

    SEARCH_FUNC = {
        Macros.sa_task : {
            Macros.datasets[Macros.sa_task][0]: Sst.search,
            Macros.datasets[Macros.sa_task][1]: ChecklistTestsuite.search
        },
        Macros.mc_task : {},
        Macros.qqp_task : {}
    }

    @classmethod
    def get_dataset(cls, task_name, dataset_name):
        pass

    @classmethod
    def search_sentiment_analysis(cls, requirements, dataset):
        func = cls.SEARCH_FUNC[Macros.sa_task][dataset]
        for req in requirements:
            selected = func(req)
            yield {
                "requirement": req,
                "selected_inputs": selected
            }
        # end for
        return


