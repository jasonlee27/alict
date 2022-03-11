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
from .CFGDiff import CFGDiff
from .RefPCFG import RefPCFG


random.seed(Macros.SEED)

class CFGExpander:

    def __init__(self, seed_input, pcfg_ref, selection_method, ref_corpus='treebank'):
        self.corpus_name = ref_corpus
        self.seed_input: str = seed_input
        self.selection_method = selection_method
        tree_dict = self.get_seed_cfg()
        self.tree_seed = tree_dict['tree']
        self.cfg_seed: dict = tree_dict['rule']
        # self.pcfg_ref = RefPCFG(corpus_name=self.corpus_name)
        self.pcfg_ref = pcfg_ref
        self.cfg_diff: dict = self.get_cfg_diff()
        # self.is_ref_pcfg: bool = is_ref_pcfg
        # self.cfg_ref: dict = self.get_tb_ref_cfg(ref_corpus=ref_corpus)
        del tree_dict
    
    def get_seed_cfg(self):
        return BeneparCFG.get_seed_cfg(self.seed_input)
    
    def get_ref_pcfg(self):
        return self.pcfg_ref.pcfg
    
    def get_cfg_diff(self):
        # get the cfg components that can be expanded 
        # compared with ref grammer
        return CFGDiff(
            pcfg_ref=self.pcfg_ref,
            cfg_seed=self.cfg_seed,
            tree_seed=self.tree_seed,
            selection_method=self.selection_method
        ).cfg_diff
    
