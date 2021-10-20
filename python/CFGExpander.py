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

from Macros import Macros
from Utils import Utils
from CFG import BeneparCFG, TreebankCFG
from Search import Search
from Requirements import Requirements

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
        
    # def get_expanded_cfg_component(self, cfg_dict):
    #     # generate new perturbed cfg expanded from seed cfg 
    #     # and cfg difference between the seed and reference cfg
    #     keys = ['N/A']
    #     while(any(keys)):
    #         keys = list()
    #         for lhs, rhss in cfg_dict.items():
    #             _key = [r for rhs in rhss for r in rhs if r not in cfg_dict.keys() and r in self.cfg_ref.keys()]
    #             if any(_key):
    #                 keys += _key
    #             # end if
    #         # end for
    #         for k in keys:
    #             if k in self.cfg_ref.keys():
    #                 if self.is_ref_pcfg:
    #                     cfg_dict[k] = [r[0] for r in self.cfg_ref[k]]
    #                 else:
    #                     cfg_dict[k] = self.cfg_ref[k]
    #             # end if
    #         # end for
    #     # end while
    #     return cfg_dict

    # def generate(cfg_dict, seed_input, items=["S"], num_sents=20):
    #     # generate final perturbed sentences.
    #     frags = []
    #     if len(items) == 1:
    #         if isinstance(items[0], Nonterminal):
    #             for prod in grammar.productions(lhs=items[0]):
    #                 frags.append(self.generate(grammar, prod.rhs()))
    #         else:
    #             frags.append(items[0])
    #     else:
    #         # This is where we need to make our changes
    #         chosen_expansion = choice(items)
    #         frags.append(self.generate,chosen_expansion)
    #     return frags

    # def sample_expandable_tpos_from_seed(self):
    #     return [lhs for lhs in self.cfg_seed.keys() if lhs in self.cfg_diff.keys()]

    # def is_terminal(self, input_str):
    #     if re.search("^\'(.+)\'$", input_str):
    #         return True
    #     # end if
    #     return False

    # def generate_cfg(self, num_candid=5):
    #     # There are two randomness in generating new cfg:
    #     #     1. given a seed input and its tags of pos, which pos you would like to expand?
    #     #     2. given selected a tag of pos, which expanded rule will be replaced with?
    #     # Solution:
    #     #     1.randomly select them

    #     # Generate new cfg over seed input cfg
    #     temp_dict = copy.deepcopy(self.cfg_seed)
    #     lhs_candids = self.sample_expandable_tpos_from_seed()
    #     candids = list()
    #     for lhs_seed, rhs_seed in temp_dict.items():
    #         if lhs_seed in lhs_candids:
    #             rhs_diffs = self.cfg_diff[lhs_seed]
    #             for rhs_from in rhs_seed:
    #                 if len(rhs_from)==1:
    #                     rhs_from_key = rhs_from[0]

    #                 else:
    #                     rhs_from_key = tuple(rhs_from)
    #                 # end if
    #                 sampled_rhs_to = list()
    #                 for _rhs_from, _rhs_to in rhs_diffs.items():
    #                     if _rhs_from==rhs_from_key:
    #                         if self.is_ref_pcfg:
    #                             candids.extend([(lhs_seed, rhs_from_key, r[0], r[1]) for r in _rhs_to])
    #                         else:
    #                             candids.extend([(lhs_seed, rhs_from_key, r) for r in _rhs_to])
    #                     # end if
    #                 # end for
    #             # end for
    #         # end if
    #     # end for
    #     candids = sorted(candids, key=lambda x: x[-1], reverse=True)
    #     cfgs_expanded = list()
    #     num_added_candids = 0
    #     for candid in candids:
    #         cfg_expanded = copy.deepcopy(self.cfg_seed)
    #         if self.is_ref_pcfg:
    #             lhs_seed, rhs_from, rhs_to, prob = candid
    #         else:
    #             prob = -1
    #             lhs_seed, rhs_from, rhs_to = candid
    #         # end if

    #         # Check if expanded rhs already exists in seed cfg
    #         # If exists, we ignore the expansion
    #         already_exists = False
    #         for rhs_i, rhs in enumerate(cfg_expanded[lhs_seed]):
    #             if rhs==rhs_to:
    #                 already_exists = True
    #             # end if
    #         # end for

    #         # If expanded rhs does not exists in seed cfg,
    #         # then we replace rhs with the new expanded rhs.
    #         if not already_exists and num_added_candids<num_candid:
    #             print(f"***\n{lhs_seed} ->\n\tFROM: {rhs_from}\n\tTO:   {tuple(rhs_to)}\n\t(PROB: {prob})")
    #             num_added_candids += 1
    #             for rhs_i, rhs in enumerate(cfg_expanded[lhs_seed]):
    #                 rhs_key = tuple(rhs)
    #                 if len(rhs)==1:
    #                     rhs_key = rhs[0]
    #                 # end if
    #                 if rhs_key==rhs_from:
    #                     cfg_expanded[lhs_seed][rhs_i] = rhs_to
    #                 # end if
    #             # end for
    #             if cfg_expanded==self.cfg_seed:
    #                 print("CFG is not expanded.")
    #             else:
    #                 cfgs_expanded.append(self.get_expanded_cfg_component(cfg_expanded))
    #                 print()
    #             # end if
    #         elif num_added_candids==num_candid:
    #             return cfgs_expanded
    #         # end if
    #     # end for
    #     return cfgs_expanded

        
# def main():
#     cfg_ref_file = Macros.result_dir / 'treebank_pcfg.json'
#     input_file = Macros.this_dir / 'ex.txt'
#     cfg_seed_file = Macros.result_dir / 'ex_cfg.json'
#     cfg_diff_file = Macros.result_dir / 'ex_treebank_cfg_diff.json'

#     seed_input = "I broke my arm playing tennis"
#     #seed_input = "I think this airline was great"
#     generator = CFGExpander(seed_input=seed_input, cfg_ref_file=cfg_ref_file)
#     cfg_expanded = generator.generate_cfg(num_candid=20)

def main():
    cfg_ref_file = Macros.result_dir / 'treebank_cfg.json'
    # cfg_seed_file = Macros.result_dir / 'ex_cfg.json'
    # cfg_diff_file = Macros.result_dir / 'ex_treebank_cfg_diff.json'

    for task in Macros.datasets.keys():
        print(task)
        reqs = Requirements.get_requirements(task)
        for selected in Search.search_sst(reqs):
            print(selected["description"])
            for inp in selected["selected_inputs"]:
                _id, seed = inp[0], inp[1]
                print(f"{_id}: {seed}")
                generator = CFGExpander(seed_input=seed, cfg_ref_file=cfg_ref_file)
                print(generator.tree_seed)
                print(generator.cfg_seed)
                for key, value in generator.cfg_diff.items():
                    for _key, _value in value.items():
                        print(f"WORD: {_value[1]}\nLHS: {key}\n\tFROM: {_key}\n\tTO: {_value[0][:5]}\n")
                print("\n\n")
            # end for
            print()
        # end for
    # end for
    return


if __name__=='__main__':
    main()
