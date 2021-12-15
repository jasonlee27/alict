# This script perturbates input and generate more inputs
# given the input.

from typing import *

import re, os
import sys
import json
import nltk
import random

from pathlib import Path
from nltk.parse.generate import generate, demo_grammar
from nltk import CFG

from ..utils.Macros import Macros
from ..utils.Utils import Utils

class InputPerturb:

    def __init__(self, 
                 input_file: Path, 
                 input_cfg_file: Path,
                 cfg_diff_file: Path):
        self.input_sents: List[str] = Utils.read_txt(input_file)
        self.input_cfg: dict = Utils.read_json(input_cfg_file)
        self.input_cfg_str: str = self.cfg_dict_to_str(self.input_cfg)
        self.cfg_diff: dict = Utils.read_json(cfg_diff_file)
        # self.cfg_diff_str: str = self.cfg_dict_to_str(self.cfg_diff)
        for s_i, sent in enumerate(self.input_sents,1):
            print("%3d. %s" % (s_i, sent.strip()))
        
    def cfg_dict_to_str(self, input_cfg):
        res_str = "\n  "
        for lhs, rhs in input_cfg.items():
            res_str += f"{lhs} -> "
            random.shuffle(rhs)
            for r_i, r in enumerate(rhs):
                res_str += " ".join(r)
                if r_i+1<len(rhs):
                    res_str += " | "
                else:
                    res_str += "\n  "
                # end if
            # end for
        # end for
        return res_str

    def get_cfg_grammar(self, cfg_str):
        return CFG.fromstring(cfg_str)

    def gen_random_sentence(self, cfg_str, num_sents=23, max_length=10):
        random.seed(Macros.SEED)
        grammar = self.get_cfg_grammar(cfg_str)
        print("\n***** INPUT CFG GRAMMAR *****")
        print(cfg_str)
        print("*****************************\n")
        print("\n***** GENERATED SENTS *****")
        for n, sent in enumerate(generate(grammar, n=num_sents),1):
            if len(sent)<=max_length:
                print("%3d. %s" % (n, " ".join(sent)))
                continue
            # end if
        # end for
        print("*****************************\n")

    def main(self):
        self.gen_random_sentence(self.input_cfg_str, num_sents=100)
        return



if __name__=="__main__":
    # cfg_ref_file = Macros.result_dir / 'treebank_cfg.json'
    input_file = Macros.this_dir / 'ex.txt'
    cfg_ut_file = Macros.result_dir / 'ex_cfg.json'
    cfg_diff_file = Macros.result_dir / 'ex_treebank_cfg_diff.json'

    ip = InputPerturb(
        input_file=input_file,
        input_cfg_file=cfg_ut_file,
        cfg_diff_file=cfg_diff_file
    )
    ip.main()

    
