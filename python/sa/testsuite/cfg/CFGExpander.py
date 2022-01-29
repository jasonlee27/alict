# This script is to perturb cfg of sentences
# given a seed input and reference CFGs.

from typing import *

import re, os
import nltk
import copy
import random
import numpy

from pathlib import Path
from nltk.parse.generate import generate, demo_grammar
from nltk import CFG

from ...utils.Macros import Macros
from ...utils.Utils import Utils
from .CFG import BeneparCFG # , TreebankCFG
from RefPCFG import RefPCFG


random.seed(Macros.SEED)
COMP_LENGTH = 3

# class CFGDiff:

#     def __init__(self, 
#                  cfg_ref: dict, 
#                  cfg_seed: dict,
#                  comp_length=COMP_LENGTH,
#                  is_ref_pcfg=False,
#                  write_diff=False, 
#                  diff_file=None,
#                  pretty_format=True):
#         self.cfg_diff = self.get_cfg_diff(
#             cfg_seed=cfg_seed,
#             cfg_ref=cfg_ref, 
#             is_ref_pcfg=is_ref_pcfg,
#             comp_length = comp_length
#         )
#         # if write_diff and (diff_file is not None):
#         #     self.write_cfg_diff(diff_file, pretty_format=pretty_format)
#         # # end if

#     def check_list_inclusion(self, a_list, b_list):
#         temp = -1
#         a_is = list()
#         for a in a_list:
#             try:
#                 a_i = b_list.index(a)
#                 if temp<a_i:
#                     temp = a_i
#                     a_is.append(a)
#                 else:
#                     a_is.append(None)
#                 # end if
#             except ValueError:
#                 a_is.append(None)
#             # end try
#         # end for
#         if all(a_is):
#             return True
#         # end if
#         return False
    
#     def get_cfg_diff(self, cfg_seed, cfg_ref, is_ref_pcfg=False, comp_length=COMP_LENGTH):
#         cfg_diff = dict()
#         for seed_lhs, seed_rhs in cfg_seed.items():
#             try:
#                 for _sr in seed_rhs:
#                     sr = _sr["pos"]
#                     sr = tuple([sr]) if type(sr)==str else tuple(sr)
#                     if type(sr)==str:
#                         # if not is_ref_pcfg:
#                         #     rule_from_ref = [
#                         #         rr[-1] for rr in cfg_ref[seed_lhs] \
#                         #         if self.check_list_inclusion([sr], rr[-1]) and \
#                         #         [sr]!=rr[-1] and len(rr[-1])<comp_length+len([sr])
#                         #     ]
#                         # else:
#                         #     rule_from_ref = [
#                         #         (rr[1],rr[-1]) for rr in cfg_ref[seed_lhs] \
#                         #         if self.check_list_inclusion([sr], rr[1]) and \
#                         #         [sr]!=list(rr[1]) and len(rr[1])<comp_length+len([sr])
#                         #     ]
#                         # # end if
#                         # if seed_lhs not in cfg_diff.keys() and any(rule_from_ref):
#                         #     cfg_diff[seed_lhs] = {
#                         #         sr: (rule_from_ref, _sr["word"])
#                         #     }
#                         # elif sr not in cfg_diff[seed_lhs].keys() and any(rule_from_ref):
#                         #     cfg_diff[seed_lhs][sr] = (rule_from_ref, _sr["word"])
#                         # # end if
#                         pass
#                     else:
#                         rule_from_ref = list()
#                         if not is_ref_pcfg:
#                             for rr in cfg_ref[seed_lhs]:
#                                 rr = [[r.split('-')[0] for r in rr[0]], rr[-1]]
#                                 if self.check_list_inclusion(list(sr), rr[0]) and \
#                                    len(list(sr))<len(rr[0]) and \
#                                    len(rr[0])<comp_length+len(sr) and \
#                                    rr[0] not in rule_from_ref:
#                                     rule_from_ref.append(rr[0])
#                                 # end if
#                             # end for
#                         else:
#                             for rr in cfg_ref[seed_lhs]:
#                                 rr = [[r.split('-')[0] for r in rr[0]], rr[1], rr[2]]
#                                 if self.check_list_inclusion(list(sr), rr[0]) and \
#                                    len(list(sr))<len(rr[0]) and \
#                                    len(rr[0])<comp_length+len(sr) and \
#                                    rr[0] not in rule_from_ref:
#                                     rule_from_ref.append((rr[0],rr[-1]))
#                                 # end if
#                             # end for
#                         # end if
#                         if seed_lhs not in cfg_diff.keys() and any(rule_from_ref):
#                             cfg_diff[seed_lhs] = {
#                                 sr: (rule_from_ref, _sr["word"])
#                             }
#                         elif sr not in cfg_diff[seed_lhs].keys() and any(rule_from_ref):
#                             cfg_diff[seed_lhs][sr] = (rule_from_ref, _sr["word"])
#                         # end if
#                     # end if
#                 # end for
#             except KeyError:
#                 continue
#             # end try
#         # end for
#         return cfg_diff
    
#     def write_cfg_diff(self, cfg_diff_file, pretty_format=False):
#         with open(cfg_diff_file, 'w') as f:
#             if pretty_format:
#                 json.dump(self.cfg_diff, f, indent=4)
#             else:
#                 json.dump(self.cfg_diff, f)
#             # end if
#         # end with


class CFGExpander:

    def __init__(self, seed_input, ref_corpus='treebank'):
        self.corpus_name = ref_corpus
        self.seed_input: str = seed_input
        tree_dict = self.get_seed_cfg()
        self.tree_seed = tree_dict['tree']
        self.cfg_seed: dict = tree_dict['rule']
        self.pcfg_ref = RefPCFG(corpus_name=self.corpus_name)
        self.cfg_diff: dict = self.get_cfg_diff()
        # self.is_ref_pcfg: bool = is_ref_pcfg
        # self.cfg_ref: dict = self.get_tb_ref_cfg(ref_corpus=ref_corpus)
        del tree_dict
    
    def get_seed_cfg(self, cfg_file=None, pretty_format=False):
        return BeneparCFG.get_seed_cfg(
            self.seed_input,
            cfg_file=cfg_file,
            pretty_format=pretty_format
        )
    
    def get_ref_pcfg(self):
        return pcfg_ref.get_pcfg()
    
    def get_cfg_diff(self):
        # get the cfg components that can be expanded 
        # compared with ref grammer
        diff_obj = CFGDiff(
            pcfg_ref=self.pcfg_ref,
            cfg_seed=self.cfg_seed,
            tree_seed=self.tree_seed
        )

        return diff_obj.cfg_diff
        
