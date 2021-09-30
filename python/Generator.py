# This script is to generate new sentences
# given a seed input and their expanded CFGs.
# using BERT-like language model for word
# suggestion

from typing import *

import re, os
import nltk
import copy
import random
import numpy

from pathlib import Path
from nltk.parse.generate import generate, demo_grammar
from nltk import CFG

import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb


from Macros import Macros
from Utils import Utils
from CFGExpander import CFGExpander



class Generator:

    def __init__(self, expander: CFGExpander):
        self.seed_input: str = expander.seed_input
        self.cfg_seed: dict = expander.cfg_seed
        # self.cfg_ref: dict = expander.cfg_ref
        self.cfg_diff: dict = expander.cfg_diff
        self.editor = Editor()

    def get_masked_input(self):
        pass
        
    def find_all_mask_placeholder(self, masked_input, mask_token="{mask}"):
        return [(m.start(), m.end()) for m in re.finditer(mask_token, masked_sent)]

    def get_word_suggestion(self, masked_input, num_target=10):
        sug_words = self.editor.suggest(masked_input)
        masked_tok_is = self.find_all_mask_placeholder(masked_input)
        rep_sents = list()
        for sug_words in sug_words[:num_target]:
            rep_sent = masked_input
            for (m_start, m_end), w in zip(masked_tok_is, sug_words):
                rep_sent = f"{temp_sent[:m_start]} {w} {temp_sent[m_end:]}"
            # end for
            print(rep_sent)
            rep_sents.append(rep_sent)
        # end for
        return rep_sents
        
    
