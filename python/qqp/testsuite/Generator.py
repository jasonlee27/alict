# This script is to generate new sentences
# given a seed input and their expanded CFGs.
# using BERT-like language model for word
# suggestion

from typing import *

import re, os
import nltk
import copy
import random
import numpy as np

from pathlib import Path
from nltk.parse.generate import generate, demo_grammar
from nltk import CFG

from checklist.editor import Editor

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..requirement.Requirements import Requirements
from .cfg.CFGExpander import CFGExpander
from .Suggest import Suggest

random.seed(Macros.SEED)

class Generator:

    def __init__(self, expander: CFGExpander):
        self.seed = expander
        self.editor = Editor()
        # self.seed_input: str = expander.seed_input
        # self.cfg_seed: dict = expander.cfg_seed
        # # self.cfg_ref: dict = expander.cfg_ref
        # self.cfg_diff: dict = expander.cfg_diff
        
    def masked_input_generator(self):
        seed_input = self.seed.seed_input
        masked_inputs = list()
        result = list()
        for lhs, value in self.seed.cfg_diff.items():
            for rhs_from, _value in value.items():
                words = _value[1]
                old_phrase = " ".join(list(words))
                for rhs_to in _value[0]:
                    word_ids, pos_ids = dict(), dict()
                    for pos, word in zip(rhs_from, words):
                        for r_i, r in enumerate(rhs_to):
                            if pos==r:
                                word_ids[r_i] = word
                                if pos not in pos_ids.keys():
                                    pos_ids[pos] = [r_i]
                                else:
                                    pos_ids[pos].append(r_i)
                            # end if
                        # end for
                    # end for

                    # the pos can be duplicated in expanded cfg.
                    # for example of CFG_FROM: ('DT', 'NN') & CFG_TO: ['DT', 'NN', 'NN', 'S']
                    # NN in CFG_TO is duplicated and we need to specigy which NN in CFG_TO
                    # will be corresponded to NN in CFG_FROM.
                    # for now, I randomly select one NN and consider as corresponding pos.
                    # so the following for loop is to randomly select one pos.
                    for key in pos_ids.keys():
                        if len(pos_ids[key])>1:
                            pos_ids[key] = random.choice(pos_ids[key])
                        else:
                            pos_ids[key] = pos_ids[key][0]
                        # end if
                    # end for

                    new_phrase = list()
                    for w_i,pos in enumerate(rhs_to):
                        if w_i in word_ids.keys() and w_i==pos_ids[pos]:
                            new_phrase.append(word_ids[w_i])
                        elif w_i not in word_ids.keys() and pos not in self.seed.cfg_ref.keys():
                            new_phrase.append(pos)
                        else:
                            new_phrase.append("{mask:"+pos+"}")
                        # end if
                    # end for
                    new_phrase = " ".join(new_phrase)
                    new_input = seed_input.replace(old_phrase, new_phrase)
                    if new_input not in masked_inputs:
                        _masked_input, mask_pos = self.get_pos_from_mask(new_input)
                        result.append({
                            "input": seed_input,
                            "lhs": lhs,
                            "cfg_from": f"{lhs} -> {rhs_from}",
                            "cfg_to": f"{lhs} -> {rhs_to}",
                            "target_phrase": old_phrase,
                            "masked_phrase": new_phrase,
                            "masked_input": (_masked_input, mask_pos)
                        })
                    # end if
                # end for
            # end for
        # end for
        if len(result)>Macros.num_cfg_exp_elem:
            # random sampling N cfg diffs
            idxs = np.random.choice(len(result), Macros.num_cfg_exp_elem, replace=False)
            result = [result[i] for i in idxs]
        # end if
        return result

    def get_pos_from_mask(self, masked_input: str):
        mask_pos = list()
        result = list()
        mask_pos = re.findall(r"\{mask\:([^\}]+)\}", masked_input)
        result = re.sub(r"\{mask\:([^\}]+)\}", Macros.MASK, masked_input)
        return result, mask_pos