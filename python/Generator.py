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
from Search import Search
from Requirements import Requirements
from CFGExpander import CFGExpander

random.seed(Macros.SEED)

class Generator:

    def __init__(self, expander: CFGExpander):
        self.seed = expander
        # self.seed_input: str = expander.seed_input
        # self.cfg_seed: dict = expander.cfg_seed
        # # self.cfg_ref: dict = expander.cfg_ref
        # self.cfg_diff: dict = expander.cfg_diff
        self.editor = Editor()

    def masked_input_generator(self):
        seed_input = self.seed.seed_input
        for key, value in self.seed.cfg_diff.items():
            lhs = key
            for rhs_from, _value in value.items():
                words = _value[1]
                old_phrase = " ".join(list(words))
                for rhs_to in _value[0]:
                    word_ids = dict()
                    for pos, word in zip(rhs_from, words):
                        for r_i, r in enumerate(rhs_to):
                            if pos==r:
                                word_ids[r_i] = word
                            # end if
                        # end for
                    # end for
                    new_phrase = list()
                    for w_i,w in enumerate(rhs_to):
                        if w_i in word_ids.keys():
                            new_phrase.append(word_ids[w_i])
                        elif w_i not in word_ids.keys() and w not in self.seed.cfg_ref.keys():
                            new_phrase.append(w)
                        else:
                            new_phrase.append("{mask:"+w+"}")
                    new_phrase = " ".join(new_phrase)                    
                    new_input = seed_input.replace(old_phrase, new_phrase)
                    print(f"LHS: {lhs}")
                    print(f"\tCFG_FROM: {rhs_from}")
                    print(f"\tCFG_TO: {rhs_to}")
                    print(f"\tOLD_PHRASE: {old_phrase}\n\tNEW_PHRASE: {new_input}\n\n")
                    yield new_input
                # end for
            # end for
        # end for
        
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


def main():
    cfg_ref_file = Macros.result_dir / 'treebank_cfg.json'
    for task in Macros.datasets.keys():
        print(f"TASK: {task}")
        reqs = Requirements.get_requirements(task)
        for selected in Search.search_sst(reqs):
            print(selected["description"])
            for inp in selected["selected_inputs"]:
                _id, seed = inp[0], inp[1]
                print(f"SENT: {_id}: {seed}")
                expander = CFGExpander(seed_input=seed, cfg_ref_file=cfg_ref_file)
                generator = Generator(expander=expander)
                for exp_inp in generator.masked_input_generator():
                    print("~~~~~~~~~~")
            # end for
            print()
        # end for
    # end for
    return

if __name__=="__main__":
    main()
