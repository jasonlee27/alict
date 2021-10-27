# This script is to generate new templates
# for testing given new generated inputs

from typing import *

import re, os
import nltk
import copy
import random
import numpy

from pathlib import Path
from nltk.tokenize import word_tokenize as tokenize

from Macros import Macros
from Utils import Utils
from Generator import Generator
from Synonyms import Synonyms

class Template:

    @classmethod
    def generate_inputs(cls):
        cfg_ref_file = Macros.result_dir / 'treebank_cfg.json'
        for task in Macros.datasets.keys():
            print(f"TASK: {task}")
            reqs = Requirements.get_requirements(task)
            results = list()
            for selected in Search.search_sst(reqs):
                exp_inputs = dict()
                print(f"REQUIREMENT:")
                for _id, seed, seed_label in selected["selected_inputs"]:
                    print(f"SEED: {_id} {seed}, {seed_label}")
                    expander = CFGExpander(seed_input=seed, cfg_ref_file=cfg_ref_file)
                    generator = Generator(expander=expander)
                    gen_inputs = generator.masked_input_generator()
                    new_input_results = list()
                    if len(gen_inputs)>0:
                        gen_inputs = Suggest.get_new_inputs(generator.editor, gen_inputs, num_target=10)
                        _gen_inputs = list()
                        for g_i in range(len(gen_inputs)):
                            eval_results = Suggest.eval_word_suggest(gen_inputs[g_i], seed_label, selected["requirement"])
                            if len(eval_results)>0:
                                del gen_inputs[g_i]["words_suggest"]
                                gen_inputs[g_i]["new_iputs"] = eval_results
                                _gen_inputs.append(gen_inputs[g_i])
                                new_input_results.extend(eval_results)
                            # end if
                        # end for
                    # end if
                    exp_inputs[seed] = {
                        "exp_inputs": new_input_results,
                        "label": seed_label
                    }
                # end for
                results.append({
                    "requirement": selected["requirement"],
                    "inputs": exp_inputs
                })
            # end for
            
            # write raw new inputs
            Utils.write_json(results,
                             Macros.result_dir/f"cfg_expanded_inputs_{task}.json",
                             pretty_format=True)
        # end for
        return results
    
    @classmethod
    def get_new_inputs(cls, input_file):
        if os.path.exists(input_file):
            return Utils.read_json(input_file)
        # end if
        return cls.generate_inputs()

    @classmethod
    def find_pos_from_cfg_seed(cls, token, cfg_seed):
        for key, vals in cfg_seed.items():
            for val in vals:
                if val["pos"]==val["word"] and token==val["word"]:
                    return key
                # end if
            # end for
        # end for
        return
    
    @classmethod
    def get_pos(cls, mask_input: str, mask_pos: List[str], cfg_seed: Dict, words_sug: List[str], exp_input:str):
        tokens = tokenize(mask_input)
        _tokens = list()
        tokens_pos = list()
        tok_i, mask_tok_i = 0, 0
        while tok_i<len(tokens):
            if tokens[tok_i:tok_i+3]==['{', 'mask', '}']:
                _tokens.append('{mask}')
                tok_i += 3
            else:
                _tokens.append(tokens[tok_i])
                tok_i += 1
            # end if
        # end for
        tokens = _tokens
                
        for t in tokens:
            if t=="{mask}":
                if type(words_sug)==str and words_sug==t:
                    tpos = words_sug
                elif ((type(words_sug)==list) or (type(words_sug)==tuple)) and t==words_sug[mask_tok_i]:
                    tpos = mask_pos[mask_tok_i]
                    mask_tok_i += 1
                # end if
            else:
                tpos = cls.find_pos_from_cfg_seed(t, cfg_seed)
            # end if
            tokens_pos.append(tpos)
        # end for
        return tokenize(exp_input), tokens_pos 

    @classmethod
    def get_templates_by_synonyms(cls, tokens: List[str], tokens_pos: List[str]):
        template = list()
        for t, tpos in zip(tokens, tokens_pos):
            syns = Synonyms.get_synonyms(t, tpos)
            if len(syns)>0:
                template.append({
                    "{"+f"{t}_{tpos}"+"}": list(set(syns))
                })
            else:
                template.append(t)
            # end if
       # end for
        return {
            "input": " ".join(tokens),
            "place_holder": template
        }

    @classmethod
    def get_templates(cls):
        for task in Macros.datasets.keys():
            new_input_dicts = cls.get_new_inputs(Macros.result_dir/f"cfg_expanded_inputs_{task}.json")
            # for each testing linguistic capabilities,
            for t_i in range(len(new_input_dicts)):
                inputs_per_req = new_input_dicts[t_i]
                inputs = inputs_per_req["inputs"]
                templates = list()
                for seed_input in inputs.keys():
                    print(seed_input)
                    cfg_seed = inputs[seed_input]["cfg_seed"]
                    label_seed = inputs[seed_input]["label"]
                    exp_inputs = inputs[seed_input]["exp_inputs"]
                    for inp in exp_inputs:
                        (mask_input,cfg_from,cfg_to,mask_pos,word_sug,exp_input,exp_input_label) = inp
                        tokens, tokens_pos = cls.get_pos(mask_input, mask_pos, cfg_seed, word_sug, exp_input)
                        _templates = cls.get_templates_by_synonyms(tokens, tokens_pos)
                        templates.append(_templates)
                    # end for
                # end for

                # Write the template results
                req_cksum = Utils.get_cksum(inputs_per_req["requirement"]["description"])
                res_dir = Macros.result_dir/ f"templates_{task}"
                res_dir.mkdir(parents=True, exist_ok=True)
                Utils.write_json(templates,
                                 res_dir / f"templates_{req_cksum}.json",
                                 pretty_format=True)
            # end for
        # end for
        return

    
if __name__=="__main__":
    Template.get_templates()
    