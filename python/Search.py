# This script searches inputs in datasets that meet requirements

from typing import *

import re, os
import sys
import json
import random

from pathlib import Path
from Macros import Macros
from Utils import Utils

from Requirements import Requirements


datasets = Macros.datasets

class SearchOperator: 
    
    def __init__(self, requirements):
        self.capability = requirements['capability']
        self.description = requirements['description']
        self.label = requirements['label']
        self.reqs = requirements
        self.search_method = {
            "length": self.search_by_len,
            "contains": self.search_by_contains,
            # "replace": cls.search_by_replacement,
            # "add": cls.search_by_add,
            # "remove": cls.search_by_remove
        }
        requirements.pop('capability')
        requirements.pop('description')
        requirements.pop('label')
        print(f"{self.capability}: {self.description}")

    def search(self, sents):
        _sents = sents.copy()
        for r, val in self.reqs.items():
            print(r, val)
            _sents = self.search_method[r](_sents, val)
        # end for
        return _sents

    def search_by_len(self, sents, val):
        _sents = [(s_i,s.split(),l) for s_i, s, l in sents]
        return [(s_i," ".join(s),l) for s_i, s, l in _sents if len(s) < val]

    def search_by_contains(self, sents, val):
        # for s in sents:
        #     # TODO
        # # end for
        return sents

class Search:

    @classmethod
    def get_label_in_sst(cls, sent_file, label_file):
        sents = [tuple(l.split("\t")) for l in Utils.read_txt(sent_file)[1:]]
        label_scores = [tuple(l.split("|")) for l in Utils.read_txt(label_file)[1:]]
        labels = dict()
        for s_i, s in label_scores:
            s = float(s)
            if s<=0.4:
                labels[s_i] = "neg"
            elif s>0.6:
                labels[s_i] = "pos"
            else:
                labels[s_i] = "neutral"
            # end if
        #end for
        return [(s_i,s,labels[s_i]) for s_i, s in sents]
            
    @classmethod
    def search_in_sst(cls, requirements):
        # sent: (index, sentence)
        # label: (index, label score) 
        sents = cls.get_label_in_sst(Macros.sst_datasent_file, Macros.sst_label_file)
        for req in requirements:
            req_obj = SearchOperator(req)
            selected = req_obj.search(sents)
            for s in selected:
                if s[-1]=='neutral':
                    print(s)
            print(f"{len(selected)} out of {len(sents)}")
        # end for
        return


if __name__=="__main__":
    for task in datasets.keys():
        reqs = Requirements.get_requirements(task)
        Search.search_in_sst(reqs)
    # end for
