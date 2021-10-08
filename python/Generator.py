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

from checklist.editor import Editor

from Macros import Macros
from Utils import Utils
from Search import Search
from Requirements import Requirements
from CFGExpander import CFGExpander
from Suggest import Suggest

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
                        result.append({
                            "input": seed_input,
                            "lhs": lhs,
                            "cfg_from": f"{lhs} -> {rhs_from}",
                            "cfg_to": f"{lhs} -> {rhs_to}",
                            "target_phrase": old_phrase,
                            "masked_phrase": new_phrase,
                            "masked_input": new_input
                        })
                    # end if
                # end for
            # end for
        # end for
        return result

    

def main():
    cfg_ref_file = Macros.result_dir / 'treebank_cfg.json'
    for task in Macros.datasets.keys():
        print(f"TASK: {task}")
        reqs = Requirements.get_requirements(task)
        results = list()
        for selected in Search.search_sst(reqs):
            exp_inputs = list()
            new_sug_inputs = list()
            for inp in selected["selected_inputs"]:
                _id, seed = inp[0], inp[1]
                expander = CFGExpander(seed_input=seed, cfg_ref_file=cfg_ref_file)
                generator = Generator(expander=expander)
                gen_inputs = generator.masked_input_generator()
                exp_inputs.extend(gen_inputs)
                for gen_input in gen_inputs:
                    masked_input = gen_input["masked_input"]
                    for new_input in Suggest.get_new_input(generator.editor, masked_input):
                        print(new_input)
                        new_sug_inputs.append(new_input)
                    # end for
                # end for
            # end for
            selected["masked_inputs"] = exp_inputs
            results.append({
                "description": selected["description"],
                "search_requirements": selected["search_requirements"],
                "transform_requirements": selected["transform_requirements"],
                "selected_inputs": exp_inputs,
                "new_suggested_inputs": new_sug_inputs
            })
        # end for
        Utils.write_json(results,
                         Macros.result_dir/f"cfg_expanded_inputs_{task}.json",
                         pretty_format=True)
    # end for
    return


if __name__=="__main__":
    main()
