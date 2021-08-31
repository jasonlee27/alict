# This script is to generate perturbed sentences
# given a seed input and CFGs.

from typing import *

import re, os
import nltk
import random

from pathlib import Path
from nltk.parse.generate import generate, demo_grammar
from nltk import CFG

from Macros import Macros
from Utils import Utils
from CFG import BeneparCFG, TreebankCFG


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
                    if type(sr)==str:
                        search = re.search("^terminal\:\:(.+)$", sr)
                        if search:
                            # when terminal
                            sr = [f"\'{search.group(1)}\'"]
                            if seed_lhs not in cfg_diff.keys():
                                cfg_diff[seed_lhs] = {
                                    "from": [sr],
                                    "to": self.cfg_ref[seed_lhs] if sr in self.cfg_ref[seed_lhs] else sr
                                }
                            else:
                                cfg_diff[seed_lhs]["from"].append(sr)
                            # end if
                        else:
                            cfg_diff[seed_lhs] = list()
                            sr = [sr]
                            rule_from_ref = [rr for rr in self.cfg_ref[seed_lhs] if self.check_list_inclusion(sr, rr)]
                            cfg_diff[seed_lhs].append({
                                "from": sr,
                                "to": rule_from_ref if len(rule_from_ref)>0 else [sr]
                            })
                        # end if
                    else:
                        cfg_diff[seed_lhs] = list()
                        rule_from_ref = [rr for rr in self.cfg_ref[seed_lhs] if self.check_list_inclusion(list(sr), rr)]
                        cfg_diff[seed_lhs].append({
                            "from": list(sr),
                            "to": rule_from_ref if len(rule_from_ref)>0 else [list(sr)]
                        })
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
        self.cfg_ref: dict = TreebankCFG.get_cfgs(cfg_ref_file)
    
    def get_seed_cfg(self):
        return BeneparCFG.get_seed_cfg(self.seed_input)

    def get_cfg_diff(self):
        # get the cfg components that can be expanded 
        # compared with ref grammer
        cfg_seed: dict = self.get_seed_cfg()
        return CFGDiff(cfg_ref=self.cfg_ref,cfg_seed=cfg_seed).get_cfg_diff()
        
    def get_new_seed_cfg(self):
        # generate new perturbed cfg expanded from seed cfg 
        # and cfg difference between the seed and reference cfg
        diff_dict = self.get_cfg_diff()
        cfg_dict = diff_dict.copy()
        for lhs, rhss in diff_dict.items():
            if type(rhss)==list:
                for rhs in rhss:
                    for r in rhs["to"]:
                        for _r in r:
                            if _r in self.cfg_ref.keys() and \
                               _r not in cfg_dict.keys():
                                cfg_dict[_r] = self.cfg_ref[_r]
                            # end if
                        # end for
                    # end for
                # end for
            # end if
        # end for
        return cfg_dict

    def generator(self):
        # generate final perturbed sentences.
        pass

    



def main():
    cfg_ref_file = Macros.result_dir / 'treebank_cfg.json'
    input_file = Macros.this_dir / 'ex.txt'
    cfg_seed_file = Macros.result_dir / 'ex_cfg.json'
    cfg_diff_file = Macros.result_dir / 'ex_treebank_cfg_diff.json'

    seed_input = "I think this airline was great"
    generator = Generator(seed_input=seed_input, cfg_ref_file=cfg_ref_file)
    new_cfg_dict = generator.get_new_seed_cfg()


if __name__=='__main__':
    main()