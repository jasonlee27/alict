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
        self.label = requirements['label']
        requirements.pop('capability')
        requirements.pop('label')
        self.reqs = requirements
        self.search_method = {
            "length": self.search_by_len,
            "contains": self.search_by_contains,
            # "replace": cls.search_by_replacement,
            # "add": cls.search_by_add,
            # "remove": cls.search_by_remove
        }

    def search(self, sents):
        _sents = sents
        for r, val in self.reqs.items():
            print(r, val)
            _sents = self.search_method[r](_sents, val)
        # end for
        return _sents

    def search_by_len(self, sents, val):
        return [(s_i,s) for s_i, s in sents if len(s) < val]

    def search_by_contains(self, sents, val):
        # for s in sents:
        #     # TODO
        # # end for
        return sents

    
class Search:
    
    @classmethod
    def search_in_sst(cls, requirements):
        sents = [tuple(l.split("\t")) for l in Utils.read_txt(Macros.sst_datasent_file)]
        sents = [(s_i, s.split()) for s_i, s in sents]
        for req in requirements:
            print(req)
            req_obj = SearchOperator(req)
            selected = req_obj.search(sents)
            print(f"{len(selected)} out of {len(sents)}")
        # end for
        return


if __name__=="__main__":
    for task in datasets.keys():
        reqs = Requirements.get_requirements(task)
        Search.search_in_sst(reqs)
    # end for
