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


datasets = Macros.datasets

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
            _sents = self.search_method[op](_sents, param)
        # end for
        return _sents

    def search_by_len(self, sents, param):
        match = re.search(r"([<>]=?|==)(\d+)", param)
        op, _len = match.groups()
        _sents = [(s_i,tokenize(s),l) for s_i, s, l in sents]
        return [(s_i," ".join(s),l) for s_i, s, l in _sents if len(s) < int(_len)]

    def search_by_label(self, sents, param):
        return [(s_i,s,l) for s_i, s, l in sents if l==param]
    
    def search_by_contains(self, sents, param):
        word_contain = param["word"]
        tpos_contain = param["POS"]
        # for s in sents:
        #     # TODO
        # # end for
        return sents


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
                labels[s_i] = "neg"
            elif s>0.6:
                labels[s_i] = "pos"
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
            # for s in selected:
            #     if s[-1]=='neutral':
            #         print(s)
            print(f"{len(selected)} out of {len(sents)}")
        # end for
        return


if __name__=="__main__":
    for task in datasets.keys():
        reqs = Requirements.get_requirements(task)
        Search.search_sst(reqs)
    # end for
