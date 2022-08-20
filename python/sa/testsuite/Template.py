# This script is to generate new templates
# for testing given new generated inputs

from typing import *

import re, os
import nltk
import copy
import time
# import random
import numpy
# import spacy
import multiprocessing

from pathlib import Path
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from checklist.editor import Editor

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger
from ..requirement.Requirements import Requirements

from .cfg.RefPCFG import RefPCFG
from .Generator import Generator
from .Synonyms import Synonyms
from .Search import Search
from .Suggest import Suggest

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class Template:

    NUM_PROCESSES = 3 # multiprocessing.cpu_count()
    
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
        Macros.sa_task: Search.search_sentiment_analysis_per_req
    }

    @classmethod
    def generate_exp_inputs(cls,
                            editor,
                            req,
                            cksum_val,
                            index,
                            seed_id,
                            seed,
                            seed_label,
                            seed_score,
                            pcfg_ref,
                            selection_method,
                            cfg_res_file,
                            logger):
        st = time.time()
        generator = Generator(seed, pcfg_ref)
        gen_inputs = generator.masked_input_generator()
        num_syntax_exps = len(gen_inputs)
        new_input_results = list()
        num_words_orig_suggest = 0
        if any(gen_inputs):
            # sug_st = time.time()
            new_input_results, num_words_orig_suggest = Suggest.get_exp_inputs(
                editor,
                generator,
                gen_inputs,
                seed_label,
                req,
                num_target=Macros.num_suggestions_on_exp_grammer_elem,
                selection_method=selection_method,
                logger=logger
            )
            # sug_ft = time.time()
            # print(f"{index} :: Suggest<{round(sug_ft-sug_st,2)} seconds>:: ")
        # end if
        exp_inputs = {
            'cfg_seed': generator.expander.cfg_seed,
            'exp_inputs': new_input_results,
            'label': seed_label,
            'label_score': seed_score
        }
        ft = time.time()
        logger.print(f"\tREQUIREMENT::{cksum_val}::SELECTED_SEED_{index}::{seed_id}, {seed}, {seed_label}, {seed_score}::{num_syntax_exps} syntax expansions::{num_words_orig_suggest} words suggestions::{len(new_input_results)} expansions generated::{round(ft-st,2)}sec::pid{os.getpid()}")
        template_results = Utils.read_json(cfg_res_file)
        ind = [
            tr_i
            for tr_i, tr in enumerate(template_results)
            if tr['requirement']['description']==req['description']
        ]
        if any(ind):
            ind = ind[0]
            template_results[ind]['inputs'][seed] = exp_inputs
        else:
            template_results.append({
                'requirement': req,
                'inputs': {
                    seed: exp_inputs
                }
            })
        # end if
        # write raw new inputs for each requirement
        # print(len(template_results))
        Utils.write_json(template_results, cfg_res_file, pretty_format=True)
        exp_inputs['seed'] = seed
        return exp_inputs
    
    @classmethod
    def generate_inputs(cls,
                        task,
                        req,
                        pcfg_ref,
                        editor,
                        dataset,
                        n=None,
                        selection_method=None,
                        logger=None):
        st = time.time()
        selected = cls.SEARCH_FUNC[task](req, dataset)
        cksum_val = Utils.get_cksum(selected['requirement']['description'])
        num_selected_inputs = len(selected['selected_inputs'])
        print_str = f">>>>> REQUIREMENT::{cksum_val}::"+selected['requirement']['description']
        logger.print(f"{print_str}\n\t{num_selected_inputs} inputs are selected.")
        # nlp = spacy.load('en_core_web_md')
        # nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
        index = 0
        num_seed_for_exp = 0
        tot_num_exp = 0
        exp_inputs = dict()
        exp_results = list()
        seeds = selected['selected_inputs'][:n] if n>0 else selected['selected_inputs']
        pool = multiprocessing.Pool(processes=cls.NUM_PROCESSES)
        args = list()
        cfg_res_file = Macros.result_dir / f"cfg_expanded_inputs2_{task}_{dataset}_{selection_method}.json"
        for index, (_id, seed, seed_label, seed_score) in enumerate(seeds):
            args.append((editor,
                         selected['requirement'],
                         cksum_val,
                         index+1,
                         _id,
                         seed,
                         seed_label,
                         seed_score,
                         pcfg_ref,
                         selection_method,
                         cfg_res_file,
                         logger))
        # end for
        exp_results = pool.starmap_async(cls.generate_exp_inputs,
                                         args,
                                         chunksize=len(seeds)//cls.NUM_PROCESSES).get()
        for r in exp_results:
            # r = _r.get()
            seed = r['seed']
            exp_inputs[seed] = {
                'cfg_seed': r['cfg_seed'],
                'exp_inputs': r['exp_inputs'],
                'label': r['label'],
                'label_score': r['label_score']
            }
            tot_num_exp += len(exp_inputs[seed]['exp_inputs'])
        # end for
        pool.close()
        pool.join()
        ft = time.time()
        logger.print(f"\tREQUIREMENT::{cksum_val}::Total {tot_num_exp} syntactical expansions identified in the requirement out of {num_selected_inputs} seeds")
        logger.print(f"<<<<< REQUIREMENT::{cksum_val}::"+selected["requirement"]["description"]+f"{round(ft-st,2)}sec")
        return {
            'requirement': selected["requirement"],
            'inputs': exp_inputs
        }
    
    @classmethod
    def get_new_inputs(cls,
                       input_file,
                       nlp_task,
                       dataset_name,
                       n=None,
                       selection_method=None,
                       logger=None):
        # if os.path.exists(input_file):
        #     return Utils.read_json(input_file)
        # # end if
        logger.print("Analyzing CFG ...")
        reqs = Requirements.get_requirements(nlp_task)
        editor = Editor()
        pcfg_ref = RefPCFG()
        # results = list()
        args = list()
        if os.path.exists(input_file):
            template_results = Utils.read_json(input_file)
            _reqs = list()
            for req in reqs:
                if not any([True for r in template_results if r["requirement"]["description"]==req["description"]]):
                    _reqs.append(req)
                    # args.append(
                    #     (nlp_task, req, pcfg_ref, editor, dataset_name, n, selection_method, logger)
                    # )
                # end if
            # end for
            reqs = _reqs
        # end if
        
        # sort reqs from smallest number of seeds to largest
        for req in reqs:
            selected = cls.SEARCH_FUNC[nlp_task](req, dataset_name)
            req['num_seeds'] = len(selected['selected_inputs'])
            del selected
        # end for
        reqs = sorted(reqs, key=lambda x: x['num_seeds'])
        for req in reqs:
            cls.cur_req = req
            req_results = cls.generate_inputs(nlp_task,
                                              req,
                                              pcfg_ref,
                                              editor,
                                              dataset_name,
                                              n,
                                              selection_method,
                                              logger)
            # template_results.append(req_results)
            # write raw new inputs for each requirement
            Utils.write_json(results, input_file, pretty_format=True)
        # end for
        Utils.write_json(results, input_file, pretty_format=True)
        return input_dicts

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
    def get_pos(cls,
                mask_input: str,
                mask_pos: List[str],
                cfg_seed: Dict,
                words_sug: List[str],
                exp_input:str):
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
    def get_templates_by_synonyms(cls,
                                  nlp,
                                  tokens: List[str],
                                  tokens_pos: List[str],
                                  prev_synonyms):
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
    def get_templates(cls,
                      num_seeds,
                      nlp_task,
                      dataset_name,
                      selection_method,
                      log_file):
        assert nlp_task in Macros.nlp_tasks
        assert dataset_name in Macros.datasets[nlp_task]
        # Write the template results
        res_dir = Macros.result_dir / f"templates2_{nlp_task}_{dataset_name}_{selection_method}"
        # res_dir = Macros.result_dir/ f"templates_{nlp_task}_{dataset_name}_{selection_method}"
        res_dir.mkdir(parents=True, exist_ok=True)
        logger = Logger(logger_file=log_file,
                        logger_name='template')

        logger.print(f"***** TASK: {nlp_task}, SEARCH_DATASET: {dataset_name}, SELECTION: {selection_method} *****")
        # Search inputs from searching dataset and expand the inputs using ref_cfg
        # nlp = spacy.load('en_core_web_trf')
        # nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
        task = nlp_task

        # new_input_dicts = cls.get_new_inputs(
        #     Macros.result_dir/f"cfg_expanded_inputs_{task}_{dataset_name}_{selection_method}.json",
        #     task,
        #     dataset_name,
        #     n=num_seeds,
        #     selection_method=selection_method,
        #     logger=logger
        # )
        cfg_res_file = Macros.result_dir/f"cfg_expanded_inputs2_{task}_{dataset_name}_{selection_method}.json"
        new_input_dicts = cls.get_new_inputs(
            cfg_res_file,
            task,
            dataset_name,
            n=num_seeds,
            selection_method=selection_method,
            logger=logger
        )

        # Make templates by synonyms
        logger.print("Generate Templates ...")
        prev_synonyms = dict()
        cksum_map_str = ""
        for t_i in range(len(new_input_dicts)):
            # for each testing linguistic capabilities,
            inputs_per_req = new_input_dicts[t_i]
            lc_desc = inputs_per_req["requirement"]["description"]
            req_cksum = Utils.get_cksum(lc_desc)
            cksum_map_str += f"{lc_desc}\t{req_cksum}\n"
            inputs = inputs_per_req["inputs"]
            print_str = '>>>>> REQUIREMENT:'+inputs_per_req["requirement"]["description"]
            logger.print(print_str)
            seed_inputs, exp_seed_inputs = list(), list()
            seed_templates, exp_templates = list(), list()
            for s_i, seed_input in enumerate(inputs.keys()):
                logger.print(f"\tSEED {s_i}: {seed_input}")
                    
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
                        if exp_sent is not None:
                            exp_seed_inputs.append({
                                "input": inp[5],
                                "place_holder": Utils.tokenize(inp[5]),
                                "label": inp[6]
                            })
                        # end if
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
            print_str = '<<<<< REQUIREMENT:'+inputs_per_req["requirement"]["description"]
            logger.print(print_str)
        # end for
        Utils.write_txt(cksum_map_str, res_dir / 'cksum_map.txt')
        return


# Write templates
# Template.get_templates(num_seeds=10)
