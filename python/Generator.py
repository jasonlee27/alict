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
from CFG import BeneparCFG


class CFGDiff:

    def __init__(self, cfg_ref_file, cfg_ut_file, write_diff=True, diff_file=None, pretty_format=True):
        self.cfg_ref = Utils.read_json(cfg_ref_file)
        self.cfg_ut = Utils.read_json(cfg_ut_file)
        self.cfg_diff = self.get_cfg_diff()
        if write_diff and (diff_file is not None):
            self.write_cfg_diff(diff_file, pretty_format=pretty_format)
        # end if

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
        for ut_lhs, ut_rhs in self.cfg_ut.items():
            cfg_diff[ut_lhs] = list()
            try:
                # print(f"{ut_lhs} -> {ut_rhs}")
                for ur in ut_rhs:
                    cfg_diff[ut_lhs].append({
                        "rule_from_data": ur,
                        "rule_from_ref": [rr for rr in self.cfg_ref[ut_lhs] if self.check_list_inclusion(ur, rr)]
                    })
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

    def __init__(self, seed_input, ref_grammar):
        self.seed_input: str = seed_input
        self.ref_grammar: Dict = ref_grammar
    
    def generator(self):
        # generate final perturbed sentences.
        pass

    def get_new_cfg_comp(self):
        # get the cfg components that can be expanded 
        # compared with ref grammer
        seed_cfg: Dict = BeneparCFG.get_seed_cfg(self.seed_input)


        pass

    def 



def main():
    cfg_ref_file = Macros.result_dir / 'treebank_cfg.json'
    input_file = Macros.this_dir / 'ex.txt'
    cfg_ut_file = Macros.result_dir / 'ex_cfg.json'
    cfg_diff_file = Macros.result_dir / 'ex_treebank_cfg_diff.json'
    
    # compare the loded grammars
    cfg_diff = CFGDiff(
        cfg_ref_file=cfg_ref_file,
        cfg_ut_file=cfg_ut_file,
        write_diff=True, 
        diff_file=cfg_diff_file
    )


if __name__=='__main__':
    main()