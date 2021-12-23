# This script is to generate new templates
# for testing given new generated inputs

from typing import *

import re, os
import nltk
import copy
# import random
import numpy
import spacy

from pathlib import Path
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from nltk.tokenize import word_tokenize as tokenize

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..requirement.Requirements import Requirements

from .Generator import Generator
from .Synonyms import Synonyms
from .Search import Search
from .Suggest import Suggest
from .cfg.CFGExpander import CFGExpander


class Template:

    POS_MAP = {
        "NNP": "NP",
        "NNPS": "NPS",
        "PRP": "PP",
        "PRP$": "PP$"
    }

    # SEARCH_MAP = {
    #     Macros.sa_task: Search.search_sentiment_analysis
    #     Macros.mc_task:
    #     Macros.qqp_task: 
    # }
    SEARCH_FUNC = {
        Macros.sa_task: Search.search_sentiment_analysis
    }
    
    @classmethod
    def generate_inputs(cls, task, dataset, n=None, save_to=None):
        cfg_ref_file = Macros.result_dir / 'treebank_cfg.json'
        print(f"***** TASK: {task}, SEARCH_DATASET: {dataset} *****")
        reqs = Requirements.get_requirements(task)
        results = list()
        for selected in cls.SEARCH_FUNC[task](reqs, dataset):
            exp_inputs = dict()
            print(f">>>>> REQUIREMENT:", selected["requirement"]["description"])
            selected_inputs = selected["selected_inputs"][:n] if n is not None else selected["selected_inputs"]
            for _id, seed, seed_label in selected_inputs:
                print(f"\tSELECTED_SEED: {_id} {seed}, {seed_label}")
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
                    "cfg_seed": expander.cfg_seed,
                    "exp_inputs": new_input_results,
                    "label": seed_label
                }
            # end for
            results.append({
                "requirement": selected["requirement"],
                "inputs": exp_inputs
            })
            print(f"<<<<< REQUIREMENT:", selected["requirement"]["description"])
        # end for
            
        # write raw new inputs
        Utils.write_json(results, save_to, pretty_format=True)
        print(f"**********")        
        return results
    
    @classmethod
    def get_new_inputs(cls, input_file, nlp_task, dataset_name, n=None):
        if os.path.exists(input_file):
            return Utils.read_json(input_file)
        # end if
        return cls.generate_inputs(
            task=nlp_task,
            dataset=dataset_name,
            n=n,
            save_to=input_file
        )

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
                if type(words_sug)==str:
                    tpos = words_sug
                elif ((type(words_sug)==list) or (type(words_sug)==tuple)):
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
    def get_templates_by_synonyms(cls, nlp, tokens: List[str], tokens_pos: List[str], prev_synonyms):
        template = list()
        for t, tpos in zip(tokens, tokens_pos):
            key = "{"+f"{t}_{tpos}"+"}"
            if key in prev_synonyms.keys():
                if prev_synonyms[key] is None or len(prev_synonyms[key])==0:
                    template.append(t)
                else:
                    template.append({
                        key: prev_synonyms[key]
                    })
                # end if
            else:
                syns = Synonyms.get_synonyms(nlp, t, tpos)
                if len(syns)>1:
                    _syns = list()
                    for s in list(set(syns)):
                        if len(s.split("_"))>1:
                            _syns.append(" ".join(s.split("_")))
                        else:
                            _syns.append(s)
                        # end if
                    # end for
                    syns_dict = {key: _syns}
                    template.append(syns_dict)
                    if key not in prev_synonyms.keys():
                        prev_synonyms[key] = syns_dict[key]
                    # end if
                else:
                    template.append(t)
                    if key not in prev_synonyms.keys():
                        prev_synonyms[key] = None
                    # end if
                # end if
            # end if
       # end for
        return {
            "input": " ".join(tokens),
            "place_holder": template
        }, prev_synonyms

    @classmethod
    def get_templates(cls, num_seeds, nlp_task, dataset_name):
        assert nlp_task in Macros.nlp_tasks
        assert dataset_name in Macros.datasets[nlp_task]

        # Search inputs from searching dataset and expand the inputs using ref_cfg
        nlp = spacy.load('en_core_web_md')
        nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
        task = nlp_task
        # task = "sentiment_analysis"
        # if nlp_task=="mc":
        #     task = "machine_comprehension"
        # elif nlp_task=="qqp":
        #     task = "qqp"
        # # end if
        new_input_dicts = cls.get_new_inputs(
            Macros.result_dir/f"cfg_expanded_inputs_{task}.json",
            task,
            dataset_name,
            n=num_seeds
        )
        # new_input_dicts = cls.get_new_inputs(
        #     Macros.result_dir/f"cfg_expanded_inputs_{task}.json",
        #     n=num_seeds
        # )

        # Make templates by synonyms
        prev_synonyms = dict()
        # for each testing linguistic capabilities,
        for t_i in range(len(new_input_dicts)):
            inputs_per_req = new_input_dicts[t_i]
            req_cksum = Utils.get_cksum(inputs_per_req["requirement"]["description"])
            inputs = inputs_per_req["inputs"]

            seed_inputs, seed_templates, exp_templates = list(), list(), list()
            for seed_input in inputs.keys():
                print(f"SEED: {seed_input}")
                
                cfg_seed = inputs[seed_input]["cfg_seed"]
                label_seed = inputs[seed_input]["label"]
                exp_inputs = inputs[seed_input]["exp_inputs"]
                seed_inputs.append({
                    "input": seed_input,
                    "place_holder": tokenize(seed_input),
                    "label": label_seed
                })

                # make template for seed input
                tokens, tokens_pos = cls.get_pos(seed_input, [], cfg_seed, [], seed_input)
                _templates, prev_synonyms = cls.get_templates_by_synonyms(nlp, tokens, tokens_pos, prev_synonyms)
                _templates["label"] = label_seed
                seed_templates.append(_templates)
                
                # Make template for expanded inputs
                for inp_i, inp in enumerate(exp_inputs):
                    (mask_input,cfg_from,cfg_to,mask_pos,word_sug,exp_input,exp_input_label) = inp
                    tokens, tokens_pos = cls.get_pos(mask_input, mask_pos, cfg_seed, word_sug, exp_input)
                    _templates, prev_synonyms = cls.get_templates_by_synonyms(nlp, tokens, tokens_pos, prev_synonyms)
                    _templates["label"] = exp_input_label
                    exp_templates.append(_templates)
                    print(".", end="")
                # end for
                print()
            # end for

            # Write the template results
            res_dir = Macros.result_dir/ f"templates_{task}"
            res_dir.mkdir(parents=True, exist_ok=True)
            
            Utils.write_json(seed_inputs,
                             res_dir / f"seeds_{req_cksum}.json",
                             pretty_format=True)
            Utils.write_json(seed_templates,
                             res_dir / f"templates_seed_{req_cksum}.json",
                             pretty_format=True)
            Utils.write_json(exp_templates,
                             res_dir / f"templates_exp_{req_cksum}.json",
                             pretty_format=True)
        # end for
        return


# Write templates
# Template.get_templates(num_seeds=10)
