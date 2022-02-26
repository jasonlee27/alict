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

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..requirement.Requirements import Requirements

from .cfg.RefPCFG import RefPCFG
from .Generator import Generator
from .Synonyms import Synonyms
from .Search import Search
from .Suggest import Suggest


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
    def generate_inputs(cls, task, dataset, n=None, save_to=None, is_random_select=None):
        print("Analyzing CFG ...")
        reqs = Requirements.get_requirements(task)
        nlp = spacy.load('en_core_web_md')
        results = list()
        if os.path.exists(save_to):
            results = Utils.read_json(save_to)
            _reqs = list()
            for req in reqs:
                if not any([True for r in results if r["requirement"]["description"]==req["description"]]):
                    _reqs.append(req)
                # end if
            # end for
            reqs = _reqs
        # end if

        pcfg_ref = RefPCFG()
        for selected in cls.SEARCH_FUNC[task](reqs, dataset):
            exp_inputs = dict()
            print(f">>>>> REQUIREMENT:", selected["requirement"]["description"])
            num_selected_inputs = len(selected["selected_inputs"])
            print(f"\t{num_selected_inputs} inputs are selected.")
            index = 1
            num_seed_for_exp = 0
            tot_num_exp = 0
            for _id, seed, seed_label, seed_score in selected["selected_inputs"][:Macros.max_num_seeds]:
                print(f"\tSELECTED_SEED {index}: {_id}, {seed}, {seed_label}, {seed_score}", end=" :: ")
                index += 1
                generator = Generator(seed,pcfg_ref,is_random_select=is_random_select)
                gen_inputs = generator.masked_input_generator()
                print(f"{len(gen_inputs)} syntax expansions", end=" :: ")
                new_input_results = list()
                tot_num_exp += len(gen_inputs)
                if any(gen_inputs):
                    new_input_results = Suggest.get_exp_inputs(
                        nlp,
                        generator,
                        gen_inputs,
                        seed_label,
                        selected["requirement"],
                        num_target=Macros.num_suggestions_on_exp_grammer_elem,
                        is_random_select=is_random_select
                    )
                # end if
                print(f"{len(new_input_results)} word suggestion by req")
                exp_inputs[seed] = {
                    "cfg_seed": generator.expander.cfg_seed,
                    "exp_inputs": new_input_results,
                    "label": seed_label,
                    "label_score": seed_score
                }
            # end for
            print(f"Total {tot_num_exp} syntactical expansion identified in the requirement out of {num_selected_inputs} seeds")
            results.append({
                "requirement": selected["requirement"],
                "inputs": exp_inputs
            })
            # write raw new inputs for each requirement
            Utils.write_json(results, save_to, pretty_format=True)
            print(f"<<<<< REQUIREMENT:", selected["requirement"]["description"])
        # end for
        
        # # write raw new inputs
        Utils.write_json(results, save_to, pretty_format=True)
        print(f"**********")
        return results
    
    @classmethod
    def get_new_inputs(cls, input_file, nlp_task, dataset_name, n=None, is_random_select=False):
        # if os.path.exists(input_file):
        #     return Utils.read_json(input_file)
        # # end if
        return cls.generate_inputs(
            task=nlp_task,
            dataset=dataset_name,
            n=n,
            save_to=input_file,
            is_random_select=is_random_select
        )

    @classmethod
    def find_pos_from_cfg_seed(cls, token, cfg_seed):
        # when tokenized token can be found in the leaves of cfg
        for key, vals in cfg_seed.items():
            for val in vals:
                if val["pos"]==val["word"] and token==val["word"]:
                    return key
                # end if
            # end for
        # end for
        # when tokenized token cannot be found in the leaves of cfg
        for key, vals in cfg_seed.items():
            for val in vals:
                if type(val["word"])==list and token in val["word"]:
                    tok_idx = val["word"].index(token)
                    return val["pos"][tok_idx]
                # end if
            # end for
        # end for
        return
    
    @classmethod
    def get_pos(cls, mask_input: str, mask_pos: List[str], cfg_seed: Dict, words_sug: List[str], exp_input:str):
        tokens = Utils.tokenize(mask_input)
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
        return Utils.tokenize(exp_input), tokens_pos

    @classmethod
    def get_templates_by_synonyms(cls, nlp, tokens: List[str], tokens_pos: List[str], prev_synonyms):
        template = list()
        for t, tpos in zip(tokens, tokens_pos):
            newt = re.sub(r'\..*', '', t)
            newt = re.sub(r'\[.*\]', '', newt)
            newt = re.sub(r'.*?:', '', newt)
            newt = re.sub(r'\d+$', '', newt)
            key = "{"+f"{newt}_{tpos}"+"}"
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
                    for s in syns:
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
            "input": Utils.detokenize(tokens),
            "place_holder": template
        }, prev_synonyms

    @classmethod
    def get_templates(cls, num_seeds, nlp_task, dataset_name, is_random_select):
        assert nlp_task in Macros.nlp_tasks
        assert dataset_name in Macros.datasets[nlp_task]
        selection_method = 'RANDOM' if is_random_select else 'PROB'
        print(f"***** TASK: {nlp_task}, SEARCH_DATASET: {dataset_name}, SELECTION: {selection_method} *****")
        # Search inputs from searching dataset and expand the inputs using ref_cfg
        nlp = spacy.load('en_core_web_md')
        nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
        task = nlp_task
        
        new_input_dicts = cls.get_new_inputs(
            Macros.result_dir/f"cfg_expanded_inputs_{task}_{dataset_name}_{selection_method}.json",
            task,
            dataset_name,
            n=num_seeds,
            is_random_select=is_random_select
        )

        # Make templates by synonyms
        print("Generate Templates ...")
        prev_synonyms = dict()
        for t_i in range(len(new_input_dicts)):
            # for each testing linguistic capabilities,
            inputs_per_req = new_input_dicts[t_i]
            req_cksum = Utils.get_cksum(inputs_per_req["requirement"]["description"])
            inputs = inputs_per_req["inputs"]

            print(f">>>>> REQUIREMENT:", inputs_per_req["requirement"]["description"])
            seed_inputs, exp_seed_inputs = list(), list()
            seed_templates, exp_templates = list(), list()
            for s_i, seed_input in enumerate(inputs.keys()):
                # print(f"\tSEED {s_i}: {seed_input}")
                    
                cfg_seed = inputs[seed_input]["cfg_seed"]
                label_seed = inputs[seed_input]["label"]
                exp_inputs = inputs[seed_input]["exp_inputs"]
                seed_inputs.append({
                    "input": seed_input,
                    "place_holder": Utils.tokenize(seed_input),
                    "label": label_seed
                })

                if any(exp_inputs):
                    # expanded inputs
                    for inp_i, inp in enumerate(exp_inputs):
                        exp_sent = inp[5]
                        exp_seed_inputs.append({
                            "input": inp[5],
                            "place_holder": Utils.tokenize(inp[5]),
                            "label": inp[6]
                        })
                    # end for
                # end if
                
                # # make template for seed input
                # tokens, tokens_pos = cls.get_pos(seed_input, [], cfg_seed, [], seed_input)
                # _templates, prev_synonyms = cls.get_templates_by_synonyms(nlp, tokens, tokens_pos, prev_synonyms)
                # _templates["label"] = label_seed
                # seed_templates.append(_templates)

                # if any(exp_inputs):
                #     # Make template for expanded inputs
                #     for inp_i, inp in enumerate(exp_inputs):
                #         (mask_input,cfg_from,cfg_to,mask_pos,word_sug,exp_input,exp_input_label) = inp
                #         tokens, tokens_pos = cls.get_pos(mask_input, mask_pos, cfg_seed, word_sug, exp_input)
                #         _templates, prev_synonyms = cls.get_templates_by_synonyms(nlp, tokens, tokens_pos, prev_synonyms)
                #         _templates["label"] = exp_input_label
                #         exp_templates.append(_templates)
                #         print(".", end="")
                #     # end for
                #     print()
                # # end if
            # end for

            # Write the template results
            res_dir = Macros.result_dir/ f"templates_{task}_{dataset_name}_{selection_method}"
            res_dir.mkdir(parents=True, exist_ok=True)

            if any(seed_inputs):
                Utils.write_json(seed_inputs,
                                 res_dir / f"seeds_{req_cksum}.json",
                                 pretty_format=True)
            # end if
            if any(exp_seed_inputs):
                Utils.write_json(exp_seed_inputs,
                                 res_dir / f"exps_{req_cksum}.json",
                                 pretty_format=True)
            # end if
            # if any(seed_templates):
            #     Utils.write_json(seed_templates,
            #                      res_dir / f"templates_seed_{req_cksum}.json",
            #                      pretty_format=True)
            # # end if
            # if any(exp_templates):
            #     Utils.write_json(exp_templates,
            #                      res_dir / f"templates_exp_{req_cksum}.json",
            #                      pretty_format=True)
            # # end if
            print(f"<<<<< REQUIREMENT:", inputs_per_req["requirement"]["description"])
        # end for
        return


# Write templates
# Template.get_templates(num_seeds=10)
