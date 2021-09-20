# This script searches inputs in datasets that meet requirements

from typing import *

import re, os
import sys
import json
import random

from nltk.tokenize import word_tokenize as tokenize

from pathlib import Path
from Macros import Macros
from Utils import Utils

from Requirements import Requirements
from Sentiwordnet import Sentiwordnet


DATASETS = Macros.datasets

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


class SearchOperator: 
    
    def __init__(self, requirements):
        self.capability = requirements['capability']
        self.description = requirements['description']
        self.search_reqs = requirements['search']
        self.transform_reqs = requirements['transform']
        self.search_method = {
            "length": self.search_by_len,
            "contains": self.search_by_contains,
            "label": self.search_by_label
            # "replace": cls.search_by_replacement,
            # "add": cls.search_by_add,
            # "remove": cls.search_by_remove
        }
        self.transform_methos = dict() # TODO
        print(f"{self.capability}: {self.description}")

    def search(self, sents):
        _sents = sents.copy()
        for op, param in self.search_reqs.items():
            print(f"{op}: {param}")
            _sents = self.search_method[op](_sents)
        # end for
        return _sents

    def search_by_len(self, sents):
        param = self.search_reqs["length"]
        match = re.search(r"([<>]=?|==)(\d+)", param)
        op, _len = match.groups()
        _sents = [(s_i,tokenize(s),l) for s_i, s, l in sents]
        return [(s_i," ".join(s),l) for s_i, s, l in _sents if len(s) < int(_len)]

    def search_by_label(self, sents):
        label = self.search_reqs["label"]
        if label=="neutral" or label=="positive" or label=="negative":
            _sents = [(s_i,s,l) for s_i, s, l in sents if l==label]
        elif label.startswith("same as"):
            key = "POS"
            if label.split()[-1]!=key:
                key = label.split()[-1]
            # end if
            poss = self.search_reqs["contains"][key]
            _sents = list()
            for pos in poss:
                pos_sentiment = pos.split()[0]
                _sents.extend([(s_i,s,l) for s_i, s, l in sents if l==pos_sentiment])
            # end for
        # end if
        return _sents

    def _search_by_contains(self, sents, cond_key, cond_number):
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
                    
        
    def search_by_contains(self, sents):
        param = self.search_reqs["contains"]
        _sents = sents.copy()
        _sents = [(s_i, tokenize(s), l) for s_i, s, l in sents]
        word_contain = param["word"]
        tpos_contain = param["POS"]
        if word_contain is not None:
            for w in word_contain:
                _sents = [(s_i, s, l) for s_i, s, l in _sents if w in s]
            # end for
        # end if
        
        if tpos_contain is not None:
            temp_sents = list()
            for cond in tpos_contain:
                match = re.search(r"(\d+)?\s?(positive|negative|neutral)?\s?(adj|noun|verb)?(s)?", cond)
                num, sentiment, pos, is_plural = match.groups()
                if pos is None: raise("Tag of POS is not valid!")
                num = -1
                if num is None and not is_plural:
                    num = 1
                # end if
                
                if sentiment is None:
                    for _sentiment in ["neutral","positive","negative"]:
                        temp_sents.extend(self._search_by_contains(_sents, f"{_sentiment}_{pos}", num))
                    # end for
                else:
                    temp_sents = self._search_by_contains(_sents, f"{sentiment}_{pos}", num)
                # end if
                _sents = temp_sents
            # end for
        # end if
        return [(s_i," ".join(s),l) for s_i, s, l in _sents]


class TransformOperator:

    def __init__(self, requirements):
        self.capability = requirements['capability']
        self.description = requirements['description']
        self.transform_reqs = requirements['transform']
        self.transform_methos = {
            # "replace": cls.search_by_replacement,
            # "add": cls.search_by_add,
            # "remove": cls.search_by_remove
        }
        print(f"{self.capability}: {self.description}")
        

    def transform(self):
        pass
    
    
class Search:

    @classmethod
    def get_label_sst(cls, sent_file, label_file):
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
    def search_sst(cls, requirements):
        # sent: (index, sentence)
        # label: (index, label score) 
        sents = cls.get_label_sst(Macros.sst_datasent_file, Macros.sst_label_file)
        for req in requirements:
            req_obj = SearchOperator(req)
            selected = req_obj.search(sents)
            for s in selected:
                print(s)
            print(f"{len(selected)} out of {len(sents)}")
        # end for
        return


if __name__=="__main__":
    for task in DATASETS.keys():
        reqs = Requirements.get_requirements(task)
        Search.search_sst(reqs)
    # end for
    
