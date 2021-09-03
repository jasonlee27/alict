# This script is to generate perturbed sentences
# given a seed input and CFGs.

from typing import *

import re, os
import nltk
import random
import numpy

from pathlib import Path
from nltk.parse.generate import generate, demo_grammar
from nltk import CFG

from Macros import Macros
from Utils import Utils
from CFG import BeneparCFG, TreebankCFG

random.seed(Macros.SEED)

class CFGDiff:

    def __init__(self, 
                 cfg_ref: dict, 
                 cfg_seed: dict, 
                 write_diff=True, diff_file=None, pretty_format=True):
        self.cfg_ref = cfg_ref
        self.cfg_seed = cfg_seed
        self.cfg_diff = self.get_cfg_diff()
        # if write_diff and (diff_file is not None):
        #     self.write_cfg_diff(diff_file, pretty_format=pretty_format)
        # # end if

    def check_list_inclusion(self, a_list, b_list):
        a_is = [a if a in b_list else None for a_i, a in enumerate(a_list)]
        if all(a_is):
            if a_is==sorted(a_is):
                return True
            # end if
        # end if
        return False
            
    def get_cfg_diff(self):
        cfg_diff = dict()
        for seed_lhs, seed_rhs in self.cfg_seed.items():
            try:
                for sr in seed_rhs:
                    sr = sr[0] if len(sr)==1 else tuple(sr)
                    if type(sr)==str:
                        search = re.search("^terminal\:\:(.+)$", sr)
                        if search:
                            # when terminal
                            sr = f"\'{search.group(1)}\'"
                            if seed_lhs not in cfg_diff.keys():
                                cfg_diff[seed_lhs] = {
                                    "<from>": [sr],
                                    "<to>": self.cfg_ref[seed_lhs] if sr in self.cfg_ref[seed_lhs] else sr
                                }
                            else:
                                cfg_diff[seed_lhs]["from"].append(sr)
                            # end if
                        else:
                            rule_from_ref = [rr for rr in self.cfg_ref[seed_lhs] if self.check_list_inclusion([sr], rr)]
                            if seed_lhs not in cfg_diff.keys():
                                cfg_diff[seed_lhs] = {
                                    sr: [] if len(rule_from_ref)==1 and rule_from_ref[0][0]==sr else rule_from_ref
                                }
                            elif sr not in cfg_diff[seed_lhs].keys():
                                cfg_diff[seed_lhs][sr] = [] if len(rule_from_ref)==1 and rule_from_ref[0][0]==sr else rule_from_ref
                            # end if
                        # end if
                    else:
                        rule_from_ref = [rr for rr in self.cfg_ref[seed_lhs] if self.check_list_inclusion(list(sr), rr)]
                        if seed_lhs not in cfg_diff.keys():
                            cfg_diff[seed_lhs] = {
                                sr: [] if len(rule_from_ref)==1 and rule_from_ref[0][0]==sr else rule_from_ref
                            }
                        elif sr not in cfg_diff[seed_lhs].keys():
                            cfg_diff[seed_lhs][sr] = [] if len(rule_from_ref)==1 and rule_from_ref[0][0]==sr else rule_from_ref
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


class Generator:

    def __init__(self, seed_input, cfg_ref_file):
        self.seed_input: str = seed_input
        self.cfg_seed = Utils.read_json(Macros.result_dir / 'ex_cfg.json')
        # self.cfg_seed: dict = self.get_seed_cfg()
        self.cfg_ref: dict = self.get_tb_ref_cfg(cfg_ref_file)
        self.cfg_comps: dict = self.get_expanded_cfg_component()
    
    def get_seed_cfg(self):
        return BeneparCFG.get_seed_cfg(self.seed_input)

    def get_tb_ref_cfg(self, cfg_ref_file):
        return TreebankCFG.get_cfgs(cfg_ref_file)

    def get_cfg_diff(self):
        # get the cfg components that can be expanded 
        # compared with ref grammer
        return CFGDiff(cfg_ref=self.cfg_ref,cfg_seed=self.cfg_seed).get_cfg_diff()
        
    def get_expanded_cfg_component(self):
        # generate new perturbed cfg expanded from seed cfg 
        # and cfg difference between the seed and reference cfg
        diff_dict = self.get_cfg_diff()
        cfg_comps = diff_dict.copy()
        for lhs, rhss in diff_dict.items():
            if "<from>" not in rhss.keys() and "<to>" not in rhss.keys():
                for rhs_from, rhs_to in rhss.items():
                    if len(rhs_to)>0:
                        for rhs in rhs_to:
                            for r in rhs:
                                if r in self.cfg_ref.keys() and \
                                   r not in cfg_comps.keys():
                                    cfg_comps[r] = self.cfg_ref[r]
                                # end if
                            # end for
                        # end for
                    # end if
                # end for
            # end if
        # end for
        return cfg_comps

    def generate(cfg_dict, seed_input, items=["S"], num_sents=20):
        # generate final perturbed sentences.
        frags = []
        if len(items) == 1:
            if isinstance(items[0], Nonterminal):
                for prod in grammar.productions(lhs=items[0]):
                    frags.append(self.generate(grammar, prod.rhs()))
            else:
                frags.append(items[0])
        else:
            # This is where we need to make our changes
            chosen_expansion = choice(items)
            frags.append(self.generate,chosen_expansion)
        return frags

    def get_tpos_candids(self):
        lhs_seed_list = list()
        for lhs, rhs in self.cfg_seed.items():
            for _from, _to in self.cfg_comps[lhs].items():
                if len(_to)>0 and lhs not in lhs_seed_list:
                    lhs_seed_list.append(lhs)
                    break
                # end if
            # end for
        # end for
        return lhs_seed_list

    def shuffle_tpos_candids(self, num_expansion=1):
        # Choose random rhs in rhs_seed
        lhs_candids = self.get_tpos_candids()
        random.shuffle(lhs_candids)
        if num_expansion>0:
            lhs_candids = lhs_candids[:num_expansion]
        # end if
        return lhs_candids

    def is_terminal(self, input_str):
        if re.search("^\'(.+)\'$", input_str):
            return True
        # end if
        return False


    def generate_cfg(self, num_expansion=1):
        # There are two randomness in generating new cfg:
        #     1. given a seed input and its tags of pos, which pos you would like to expand?
        #     2. given selected a tag of pos, which expanded rule will be replaced with?
        # Solution:
        #     1.randomly select them

        # Generate new cfg over seed input cfg
        temp_cfg = dict()
        lhs_candids = self.shuffle_tpos_candids(num_expansion=num_expansion)
        for lhs_seed, rhs_seed in self.cfg_seed.items():
            temp_cfg[lhs_seed] = rhs_seed
            if lhs_seed in lhs_candids:
                print(f"{lhs_seed} -> {rhs_seed}")
                rhs_indices = numpy.arange(len(rhs_seed))
                random.shuffle(rhs_indices)
                if len(rhs_seed[rhs_indices[0]])==1:
                    rhs_from = rhs_seed[rhs_indices[0]][0]
                else:
                    rhs_from = tuple(rhs_seed[rhs_indices[0]])
                # end if
                candids_to = self.cfg_comps[lhs_seed][rhs_from]
                candids_indices = numpy.arange(len(candids_to))
                random.shuffle(candids_indices)
                print("*****")
                print(temp_cfg[lhs_seed])
                ind = temp_cfg[lhs_seed].index(rhs_seed[rhs_indices[0]])
                temp_cfg[lhs_seed][ind] = candids_to[candids_indices[0]]
                print(temp_cfg[lhs_seed])
                print("*****")
                print()
            # end if
        # end for

        # add new nonterminal from the expanded rule
        new_keys = ["init"]
        new_cfg = None
        while(len(new_keys)>0):
            new_keys = list()
            new_cfg = temp_cfg.copy()
            for lhs, rhss in temp_cfg.items():
                for rhs in rhss:
                    for r in rhs:
                        if r not in new_cfg.keys() and not self.is_terminal(r):
                            new_keys.append(r)
                            new_cfg[r] = self.cfg_ref[r]
                        # end if
                    # end for
                # end for
                # end if
            # end for
            temp_cfg = new_cfg.copy()
        # end while
        return new_cfg


def main():
    cfg_ref_file = Macros.result_dir / 'treebank_cfg.json'
    input_file = Macros.this_dir / 'ex.txt'
    cfg_seed_file = Macros.result_dir / 'ex_cfg.json'
    cfg_diff_file = Macros.result_dir / 'ex_treebank_cfg_diff.json'

    seed_input = "I think this airline was great"
    generator = Generator(seed_input=seed_input, cfg_ref_file=cfg_ref_file)
    new_cfg = generator.generate_cfg()
    for l,rs in new_cfg.items():
        print(f"{l} -> ")
        for r in rs[:20]:
            print(f"        {r}")
        print('\n')

if __name__=='__main__':
    main()