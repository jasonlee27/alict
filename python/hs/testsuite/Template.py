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
from ..utils.Logger import Logger
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
        Macros.hs_task: Search.search_hatespeech
    }
    
    @classmethod
    def generate_inputs(cls, task, dataset, n=None, save_to=None, selection_method=None, logger=None):
        logger.print("Analyzing CFG ...")
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
        for selected in cls.SEARCH_FUNC[task](reqs, dataset, nlp):
            exp_inputs = dict()
            print_str = '>>>>> REQUIREMENT:'+selected["requirement"]["description"]
            num_selected_inputs = len(selected["selected_inputs"])
            logger.print(f"{print_str}\n\t{num_selected_inputs} inputs are selected.")
            index = 1
            num_seed_for_exp = 0
            tot_num_exp = 0
            # for _id, seed, seed_label, seed_score in selected["selected_inputs"][:Macros.max_num_seeds]:
            for seed_dict in selected["selected_inputs"][:Macros.max_num_seeds]:
                _id = seed_dict['post_id']
                seed = Utils.detokenize(seed_dict['post_tokens'])
                seed_label = seed_dict['label']
                logger.print(f"\tSELECTED_SEED {index}: {_id}, {seed}, {seed_label} :: ", end='')
                # index += 1
                # generator = Generator(seed, pcfg_ref)
                # gen_inputs = generator.masked_input_generator()
                # logger.print(f"{len(gen_inputs)} syntax expansions :: ", end='')
                # new_input_results = list()
                # tot_num_exp += len(gen_inputs)
                # if any(gen_inputs):
                #     new_input_results = Suggest.get_exp_inputs(
                #         nlp,
                #         generator,
                #         gen_inputs,
                #         seed_label,
                #         selected["requirement"],
                #         num_target=Macros.num_suggestions_on_exp_grammer_elem,
                #         selection_method=selection_method,
                #         logger=logger
                #     )
                # # end if
                # logger.print(f"{len(new_input_results)} word suggestion by req")
                # exp_inputs[seed] = {
                #     "cfg_seed": generator.expander.cfg_seed,
                #     "exp_inputs": new_input_results,
                #     "label": seed_label,
                #     "label_score": seed_score
                # }
            # end for
            # logger.print(f"Total {tot_num_exp} syntactical expansion identified in the requirement out of {num_selected_inputs} seeds")
            # results.append({
            #     "requirement": selected["requirement"],
            #     "inputs": exp_inputs
            # })
            # write raw new inputs for each requirement
            # Utils.write_json(results, save_to, pretty_format=True)
            print_str = '<<<<< REQUIREMENT:'+selected["requirement"]["description"]
            logger.print(print_str)
        # end for
        
        # # write raw new inputs
        # Utils.write_json(results, save_to, pretty_format=True)
        logger.print('**********')
        return results
    
    @classmethod
    def get_new_inputs(cls, input_file, nlp_task, dataset_name, n=None, selection_method=None, logger=None):
        # if os.path.exists(input_file):
        #     return Utils.read_json(input_file)
        # # end if
        return cls.generate_inputs(
            task=nlp_task,
            dataset=dataset_name,
            n=n,
            save_to=input_file,
            selection_method=selection_method,
            logger=logger
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
    def get_templates(cls, num_seeds, nlp_task, dataset_name, selection_method, log_file):
        assert nlp_task in Macros.nlp_tasks
        assert dataset_name in Macros.datasets[nlp_task]
        # Write the template results
        res_dir = Macros.result_dir/ f"templates_{task}_{dataset_name}_{selection_method}"
        res_dir.mkdir(parents=True, exist_ok=True)
        logger = Logger(logger_file=log_file,
                        logger_name='template')

        logger.print(f"***** TASK: {nlp_task}, SEARCH_DATASET: {dataset_name}, SELECTION: {selection_method} *****")
        # Search inputs from searching dataset and expand the inputs using ref_cfg
        nlp = spacy.load('en_core_web_md')
        nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
        task = nlp_task
        
        new_input_dicts = cls.get_new_inputs(
            Macros.result_dir/f"cfg_expanded_inputs_{task}_{dataset_name}_{selection_method}.json",
            task,
            dataset_name,
            n=num_seeds,
            selection_method=selection_method,
            logger=logger
        )
        return


# Write templates
# Template.get_templates(num_seeds=10)
