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
from ..semexp.Suggest import Suggest, Validate
from .cfg.CFGConverter import CFGConverter


class Generator:

    def __init__(self,
                 seed: str,
                 label: str,
                 pcfg_ref: str,
                 requirement: dict):
        self.seed = seed
        self.label = label
        self.expander = CFGConverter(
            seed_input=seed,
            pcfg_ref=pcfg_ref,
        )
        self.requirement = requirement
        # self.editor = editor
        
    def masked_input_generator(self):
        seed_input = self.expander.seed_input
        masked_inputs = list()
        result = list()
        for lhs, rhs_values in self.expander.cfg_diff.items():
            for rhs_from, value in rhs_values.items():
                rhs_from = eval(rhs_from)
                rhs_to_ref = value[0]
                words = value[1]
                old_phrase = Utils.detokenize(list(words))
                for rhs_to, rhs_to_prob, sent_prob_wo_target in rhs_to_ref:
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
                        elif w_i not in word_ids.keys() and pos not in self.expander.pcfg_ref.pcfg.keys():
                            new_phrase.append(pos)
                        else:
                            new_phrase.append("{mask:"+pos+"}")
                        # end if
                    # end for
                    new_phrase = " ".join(new_phrase)
                    new_input = seed_input.replace(old_phrase, new_phrase)
                    if new_input not in masked_inputs:
                        _masked_input, mask_pos = self.get_pos_from_mask(new_input)
                        is_valid = True
                        if self.requirement.get('transform', None) and \
                           not Validate.is_conform_to_template(
                               sent=_masked_input,
                               label=self.label,
                               transform_spec=self.requirement['transform']):
                            is_valid = False
                        # end if
                        if is_valid:
                            if rhs_to_prob is None and sent_prob_wo_target is None:
                                result.append({
                                    "input": seed_input,
                                    "lhs": lhs,
                                    "cfg_from": f"{lhs} -> {rhs_from}",
                                    "cfg_to": f"{lhs} -> {rhs_to}",
                                    "target_phrase": old_phrase,
                                    "masked_phrase": new_phrase,
                                    "masked_input": (_masked_input, mask_pos),
                                })
                            else:    
                                result.append({
                                    "input": seed_input,
                                    "lhs": lhs,
                                    "cfg_from": f"{lhs} -> {rhs_from}",
                                    "cfg_to": f"{lhs} -> {rhs_to}",
                                    "target_phrase": old_phrase,
                                    "masked_phrase": new_phrase,
                                    "masked_input": (_masked_input, mask_pos),
                                    "prob": rhs_to_prob,
                                    "sent_prob_wo_target": sent_prob_wo_target
                                })
                            # end if
                        # end if
                    # end if
                # end for
            # end for
        # end for
        if len(result)>Macros.num_cfg_exp_elem and Macros.num_cfg_exp_elem>0:
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
    
