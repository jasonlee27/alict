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
from .CFG import BeneparCFG, TreebankCFG

random.seed(Macros.SEED)
COMP_LENGTH = 3

class CFGDiff:

    def __init__(self, 
                 cfg_ref: dict, 
                 cfg_seed: dict,
                 comp_length=COMP_LENGTH,
                 is_ref_pcfg=False,
                 write_diff=False, 
                 diff_file=None,
                 pretty_format=True):
        self.cfg_diff = self.get_cfg_diff(
            cfg_seed=cfg_seed, cfg_ref=cfg_ref, 
            is_ref_pcfg=is_ref_pcfg,
            comp_length = comp_length
        )
        # if write_diff and (diff_file is not None):
        #     self.write_cfg_diff(diff_file, pretty_format=pretty_format)
        # # end if

    def check_list_inclusion(self, a_list, b_list):
        temp = -1
        a_is = list()
        for a in a_list:
            try:
                a_i = b_list.index(a)
                if temp<a_i:
                    temp = a_i
                    a_is.append(a)
                else:
                    a_is.append(None)
                # end if
            except ValueError:
                a_is.append(None)
            # end try
        # end for
        if all(a_is):
            return True
        # end if
        return False
    
    def get_cfg_diff(self, cfg_seed, cfg_ref, is_ref_pcfg=False, comp_length=COMP_LENGTH):
        cfg_diff = dict()
        for seed_lhs, seed_rhs in cfg_seed.items():
            try:
                for _sr in seed_rhs:
                    sr = _sr["pos"]
                    sr = sr[0] if len(sr)==1 else tuple(sr)
                    if type(sr)==str:
                        if not is_ref_pcfg:
                            rule_from_ref = [
                                rr[-1] for rr in cfg_ref[seed_lhs] \
                                if self.check_list_inclusion([sr], rr[-1]) and \
                                [sr]!=rr[-1] and len(rr[-1])<comp_length+len([sr])
                            ]
                        else:
                            rule_from_ref = [
                                (rr[1],rr[-1]) for rr in cfg_ref[seed_lhs] \
                                if self.check_list_inclusion([sr], rr[1]) and \
                                [sr]!=list(rr[1]) and len(rr[1])<comp_length+len([sr])
                            ]
                        # end if
                        if seed_lhs not in cfg_diff.keys() and any(rule_from_ref):
                            cfg_diff[seed_lhs] = {
                                sr: (rule_from_ref, _sr["word"])
                            }
                        elif sr not in cfg_diff[seed_lhs].keys() and any(rule_from_ref):
                            cfg_diff[seed_lhs][sr] = (rule_from_ref, _sr["word"])
                        # end if
                    else:
                        if not is_ref_pcfg:
                            rule_from_ref = [
                                rr[-1] for rr in cfg_ref[seed_lhs] \
                                if self.check_list_inclusion(list(sr), rr[-1]) and \
                                list(sr)!=rr[-1] and len(rr[-1])<comp_length+len(sr)
                            ]
                        else:
                            rule_from_ref = [
                                (rr[1],rr[-1]) for rr in cfg_ref[seed_lhs] \
                                if self.check_list_inclusion(list(sr), rr[1]) \
                                and list(sr)!=rr[1] and len(rr[1])<comp_length++len(sr)
                            ]
                        # end if
                        if seed_lhs not in cfg_diff.keys() and any(rule_from_ref):
                            cfg_diff[seed_lhs] = {
                                sr: (rule_from_ref, _sr["word"])
                            }
                        elif sr not in cfg_diff[seed_lhs].keys() and any(rule_from_ref):
                            cfg_diff[seed_lhs][sr] = (rule_from_ref, _sr["word"])
                        # end if
                    # end if
                # end for
            except KeyError:
                continue
            # end try
        # end for
        return cfg_diff
    
    def write_cfg_diff(self, cfg_diff_file, pretty_format=False):
        with open(cfg_diff_file, 'w') as f:
            if pretty_format:
                json.dump(self.cfg_diff, f, indent=4)
            else:
                json.dump(self.cfg_diff, f)
            # end if
        # end with


class CFGExpander:

    def __init__(self, seed_input, cfg_ref_file, is_ref_pcfg=False):
        self.seed_input: str = seed_input
        tree_dict = self.get_seed_cfg()
        self.tree_seed = tree_dict["tree"]
        self.cfg_seed: dict = tree_dict["rule"]
        self.is_ref_pcfg: bool = is_ref_pcfg
        self.cfg_ref: dict = self.get_tb_ref_cfg(cfg_ref_file)
        self.cfg_diff: dict = self.get_cfg_diff()
        # self.cfg_comps: dict = self.get_expanded_cfg_component()
        del tree_dict
    
    def get_seed_cfg(self, cfg_file=None, pretty_format=False):
        return BeneparCFG.get_seed_cfg(self.seed_input, cfg_file=cfg_file, pretty_format=pretty_format)

    def get_tb_ref_cfg(self, cfg_ref_file):
        if self.is_ref_pcfg:
            return TreebankCFG.get_pcfgs(cfg_ref_file, pretty_format=True)
        # end if
        return TreebankCFG.get_cfgs(cfg_ref_file, pretty_format=True)

    def get_cfg_diff(self):
        # get the cfg components that can be expanded 
        # compared with ref grammer
        return CFGDiff(cfg_ref=self.cfg_ref,cfg_seed=self.cfg_seed, is_ref_pcfg=self.is_ref_pcfg).cfg_diff
        

# def main():
#     cfg_ref_file = Macros.result_dir / 'treebank_cfg.json'
#     # cfg_seed_file = Macros.result_dir / 'ex_cfg.json'
#     # cfg_diff_file = Macros.result_dir / 'ex_treebank_cfg_diff.json'

#     for task in Macros.datasets.keys():
#         print(task)
#         reqs = Requirements.get_requirements(task)
#         for selected in Search.search_sst(reqs):
#             print(selected["description"])
#             for inp in selected["selected_inputs"]:
#                 _id, seed = inp[0], inp[1]
#                 print(f"{_id}: {seed}")
#                 generator = CFGExpander(seed_input=seed, cfg_ref_file=cfg_ref_file)
#                 print(generator.tree_seed)
#                 print(generator.cfg_seed)
#                 for key, value in generator.cfg_diff.items():
#                     for _key, _value in value.items():
#                         print(f"WORD: {_value[1]}\nLHS: {key}\n\tFROM: {_key}\n\tTO: {_value[0][:5]}\n")
#                 print("\n\n")
#             # end for
#             print()
#         # end for
#     # end for
#     return


# if __name__=='__main__':
#     main()
