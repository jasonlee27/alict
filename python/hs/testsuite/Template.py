# This script is to generate new templates
# for testing given new generated inputs

from typing import *

import re, os
import nltk
import copy
import time
# import random
import numpy
import spacy
import multiprocessing
import torch.multiprocessing

from pathlib import Path
from tqdm import tqdm
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from checklist.editor import Editor

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger

from ..requirement.Requirements import Requirements
from ..seed.Search import Search
from ..synexp.Generator import Generator
from ..synexp.cfg.RefPCFG import RefPCFG
from ..semexp.Suggest import Suggest, Validate
from ..semexp.Synonyms import Synonyms

torch.multiprocessing.set_sharing_strategy('file_system')


class Template:
    
    NUM_PROCESSES = Macros.num_processes # multiprocessing.cpu_count()
    
    POS_MAP = {
        "NNP": "NP",
        "NNPS": "NPS",
        "PRP": "PP",
        "PRP$": "PP$"
    }
    
    SEARCH_FUNC = {
        Macros.hs_task: Search.search_hatespeech_per_req
    }
    
    @classmethod
    def get_seed_of_interest(cls,
                             cur_req,
                             cfg_res_file,
                             orig_seeds):
        if not os.path.exists(str(cfg_res_file)):
            return orig_seeds
        # end if
        template_results = Utils.read_json(cfg_res_file)
        seeds = list()
        if any(template_results) and \
           template_results["requirement"]["description"]==cur_req["description"]:
            for index, (_id, seed, seed_label) in enumerate(orig_seeds):
                if seed not in template_results['inputs'].keys() or \
                   not any(template_results['inputs'][seed]['exp_inputs']):
                    seeds.append((_id, seed, seed_label))
                # end if
            # end for
        # end if
        # print('=====', len(orig_seeds), len(seeds), len(template_results['inputs'].keys()))
        return seeds
    
    @classmethod
    def generate_seed_cfg_parallel(cls,
                                   index,
                                   seed,
                                   seed_label,
                                   pcfg_ref,
                                   req,
                                   logger):
        st_2 = time.time()
        pcs_id = multiprocessing.current_process().ident
        gpu_id = multiprocessing.current_process().name.split('-')[-1]
        generator = Generator(seed, seed_label, pcfg_ref, req)
        gen_inputs = generator.masked_input_generator()
        masked_input_res = {
            'seed': seed,
            'cfg_seed': generator.expander.cfg_seed,
            'label': seed_label,
            'masked_inputs': list()
        }
        
        if any(gen_inputs):
            for gen_input in gen_inputs:
                masked_input_res['masked_inputs'].append({
                    'cfg_from': gen_input['cfg_from'],
                    'cfg_to': gen_input['cfg_to'],
                    'masked_input': gen_input['masked_input'] # (_masked_input, mask_pos)
                })
            # end for
        # end if
        ft_2 = time.time()
        if logger is not None:
            logger.print(f"\tTemplate.generate_masked_inputs::SEED_{index}::{seed}|{seed_label}::{round(ft_2-st_2,3)}sec::pcs{pcs_id}::gpu{gpu_id}")
        # end if
        return masked_input_res
        
    @classmethod
    def generate_masked_inputs(cls,
                               req,
                               seeds,
                               pcfg_ref,
                               cfg_res_file,
                               cuda_device_inds=None,
                               logger=None):
        st = time.time()
        masked_input_res = dict()
        args = list()
        template_results = {
            'requirement': req,
            'inputs': dict()
        }
        if os.path.exists(cfg_res_file):
            template_results = Utils.read_json(cfg_res_file)
        # end if

        if logger is not None:
            logger.print(f"\tTemplate.generate_masked_inputs::{len(seeds)} seeds identified")
        # end if
        if cuda_device_inds is not None:
            assert len(cuda_device_inds)==cls.NUM_PROCESSES
            editors = {
                f"{gpu_id}": Editor(cuda_device_ind=gpu_id)
                for gpu_id in range(len(cuda_device_inds))
            }
            num_sents_per_gpu = len(seeds)//cls.NUM_PROCESSES
        else:
            editor = Editor()
            num_sents_per_gpu = len(seeds)
        # end if
        
        for index, (_id, seed, seed_label) in enumerate(seeds):
            args.append((index, seed, seed_label, pcfg_ref, req, logger))
        # end for

        if any(args):
            num_pcss = cls.NUM_PROCESSES if len(args)>=cls.NUM_PROCESSES else 1
            pool = multiprocessing.Pool(processes=num_pcss)
            results = pool.starmap_async(cls.generate_seed_cfg_parallel,
                                         args,
                                         chunksize=len(args)//num_pcss).get()
            for r in results:
                template_results['inputs'][r['seed']] = {
                    'cfg_seed': r['cfg_seed'],
                    'exp_inputs': list(),
                    'label': r['label'],
                }
                if any(r['masked_inputs']):
                    template_results['inputs'][r['seed']]['masked_inputs'] = r['masked_inputs']
                # end if
            # end for
            pool.close()
            pool.join()
            
            # write batch results into result file
            Utils.write_json(template_results, cfg_res_file, pretty_format=True)
        # end if

        ft = time.time()
        if logger is not None:
            logger.print(f"\tTemplate.generate_masked_inputs::{round(ft-st,3)}sec")
        # end if
        return masked_input_res

    @classmethod
    def get_word_suggestions(cls,
                             cfg_res_file,
                             num_target=Macros.num_suggestions_on_exp_grammer_elem,
                             selection_method=None,
                             gpu_ids=None,
                             logger=None):
        st = time.time()
        template_results = Utils.read_json(cfg_res_file)
        
        # collect all masked sents
        masked_sents: Dict = dict()
        no_mask_key = '<no_mask>'
        for index, seed in enumerate(template_results['inputs'].keys()):
            seed_label = template_results['inputs'][seed]['label']
            cfg_seed = template_results['inputs'][seed]['cfg_seed']
            if 'masked_inputs' in template_results['inputs'][seed].keys():
                if not any(template_results['inputs'][seed]['masked_inputs']):
                    if no_mask_key not in masked_sents.keys():
                        masked_sents[no_mask_key] = {
                            'inputs': list(),
                            'word_sug': list()
                        }
                    # end if
                    masked_sent_obj = (
                        seed,
                        seed_label,
                        cfg_seed,
                        None,
                        None,
                        None
                    )
                    if masked_sent_obj not in masked_sents[no_mask_key]['inputs']:
                        masked_sents[no_mask_key]['inputs'].append(masked_sent_obj)
                    # end if
                else:
                    for m in template_results['inputs'][seed]['masked_inputs']:
                        cfg_from = m['cfg_from']
                        cfg_to = m['cfg_to']
                        key = m['masked_input'][0] # m['masked_input'] = (_masked_input, mask_pos)
                        # num_mask_tokens = len([t for t in key.split() if Macros.MASK in t])
                        num_mask_tokens = len(eval(cfg_to.split(' -> ')[-1]))-len(eval(cfg_from.split(' -> ')[-1]))
                        if num_mask_tokens<10:
                            if key not in masked_sents.keys():
                                masked_sents[key] = {
                                    'inputs': list(),
                                    'word_sug': list()
                                }
                            # end if
                            masked_sent_obj = (
                                seed,
                                seed_label,
                                cfg_seed,
                                cfg_from,
                                cfg_to,
                                m['masked_input'][1]
                            )
                            if masked_sent_obj not in masked_sents[key]['inputs']:
                                masked_sents[key]['inputs'].append(masked_sent_obj)
                            # end if
                        # end if
                    # end for
                # end if
            # end if
        # end for
        
        # get word suggestions
        if any(masked_sents):
            masked_sents = Suggest.get_word_suggestions_over_seeds(masked_sents,
                                                                   num_target=num_target,
                                                                   selection_method=selection_method,
                                                                   no_mask_key=no_mask_key,
                                                                   cuda_device_inds=gpu_ids,
                                                                   logger=logger)
        # end if
        ft = time.time()
        if logger is not None:
            logger.print(f"\tTemplate.get_word_suggestions::{round(ft-st,3)}sec")
        # end if
        return masked_sents
    
    @classmethod
    def generate_exp_inputs(cls,
                            editor,
                            req,
                            cksum_val,
                            index,
                            seed_id,
                            seed,
                            seed_label,
                            pcfg_ref,
                            selection_method,
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
        }
        ft = time.time()
        logger.print(f"\tREQUIREMENT::{cksum_val}::SELECTED_SEED_{index}::{seed_id}, {seed}, {seed_label}::{num_syntax_exps} syntax expansions::{num_words_orig_suggest} words suggestions::{len(new_input_results)} expansions generated::{round(ft-st,2)}sec::pid{os.getpid()}")
        return {
            'seed': seed,
            'cfg_seed': generator.expander.cfg_seed,
            'exp_inputs': new_input_results,
            'label': seed_label,
        }

    @classmethod
    def temp_generate_inputs(cls,
                             nlp,
                             task,
                             req,
                             pcfg_ref,
                             dataset,
                             res_dir,
                             num_seeds=None,
                             selection_method=None,
                             gpu_ids=None,
                             logger=None):
        st = time.time()
        # generate seeds
        
        selected = cls.SEARCH_FUNC[task](req, dataset, nlp)
        seeds = selected['selected_inputs']
        cksum_val = Utils.get_cksum(selected['requirement']['description'])
        cfg_res_file = res_dir / f"cfg_expanded_inputs_{cksum_val}.json"
        cfg_res = Utils.read_json(cfg_res_file)
        print(req)
        seed_sents = dict()
        for seed in seeds:
            (_id, seed_sent, seed_label) = seed
            seed_sents[seed_sent] = seed_label
        # end for
        
        if len(seed_sents)<len(cfg_res['inputs'].keys()):
            sents = list()
            keys = list(cfg_res['inputs'].keys())
            for s in keys:
                if s not in seed_sents.keys():
                    sents.append(s)
                    removed_value = cfg_res['inputs'].pop(s, 'No Key found')
                    # del cfg_res['inputs'][s]
                else:
                     if seed_sents[s]!=cfg_res['inputs'][s]['label']:
                         cfg_res['inputs'][s]['label'] = seed_sents[s]
                    # end if
                # end if
            # end for    
        else:
            keys = list(cfg_res['inputs'].keys())
            for s in keys:
                if s not in seed_sents.keys():
                    removed_value = cfg_res['inputs'].pop(s, 'No Key found')
                else:
                    if seed_sents[s]!=cfg_res['inputs'][s]['label']:
                        cfg_res['inputs'][s]['label'] = seed_sents[s]
                    # end if

                # end if
            # end for            
        # end if
        Utils.write_json(cfg_res, cfg_res_file, pretty_format=True)
        print('==========')
        return
        
    
    @classmethod
    def generate_inputs(cls,
                        nlp,
                        task,
                        req,
                        pcfg_ref,
                        dataset,
                        res_dir, # cfg_res_file,
                        num_seeds=None,
                        selection_method=None,
                        gpu_ids=None,
                        logger=None):
        st = time.time()
        # generate seeds
        
        selected = cls.SEARCH_FUNC[task](req, dataset, nlp)
        cksum_val = Utils.get_cksum(selected['requirement']['description'])
        cfg_res_file = res_dir / f"cfg_expanded_inputs_{cksum_val}.json"
        seeds = selected['selected_inputs'][:num_seeds] if num_seeds>0 else selected['selected_inputs']
        seeds = cls.get_seed_of_interest(req, cfg_res_file, seeds)
        num_selected_inputs = len(selected['selected_inputs'])
        print_str = f">>>>> REQUIREMENT::{cksum_val}::"+selected['requirement']['description']
        if logger is not None:
            logger.print(f"{print_str}\n\t{len(seeds)} inputs are selected out of {num_selected_inputs}.")
        # end if
        index = 0
        num_seed_for_exp = 0
        tot_num_exp = 0
        exp_inputs = dict()
        exp_results = list()

        # anlyze cfg and get masked input for all seeds of interest
        cls.generate_masked_inputs(req,
                                   seeds,
                                   pcfg_ref,
                                   cfg_res_file,
                                   logger=logger)

        # get the suggested words for the masked inputs using bert:
        num_target=Macros.num_suggestions_on_exp_grammer_elem
        masked_inputs_w_word_sug = cls.get_word_suggestions(cfg_res_file,
                                                            num_target=num_target,
                                                            selection_method=selection_method,
                                                            gpu_ids=gpu_ids,
                                                            logger=logger)
        
        # validate word suggestion
        Suggest.eval_word_suggestions_over_seeds(masked_inputs_w_word_sug,
                                                 req,
                                                 cfg_res_file,
                                                 num_target=num_target,
                                                 selection_method=selection_method,
                                                 logger=logger)

        # template_results = Utils.read_json(cfg_res_file)
        ft = time.time()
        logger.print(f"<<<<< REQUIREMENT::{cksum_val}::"+selected["requirement"]["description"]+f"{round(ft-st,2)}sec")
        return
    
    @classmethod
    def get_new_inputs(cls,
                       res_dir, # cfg_res_file,
                       nlp_task,
                       dataset_name,
                       num_seeds=None,
                       selection_method=None,
                       gpu_ids=None,
                       logger=None):
        # if os.path.exists(input_file):
        #     return Utils.read_json(input_file)
        # # end if
        logger.print("Analyzing CFG ...")
        reqs = Requirements.get_requirements(nlp_task)
        reqs = [
            r for r in reqs
            if r.get('use_testcase', None)!='hatecheck'
        ]
        # editor = Editor()
        pcfg_ref = RefPCFG()
        nlp = spacy.load('en_core_web_md')
        nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
        # for r_i, req in enumerate(reqs):
        #     cls.temp_generate_inputs(nlp,
        #                         nlp_task,
        #                         req,
        #                         pcfg_ref,
        #                         dataset_name,
        #                         res_dir, # cfg_res_file,
        #                         num_seeds=num_seeds,
        #                         selection_method=selection_method,
        #                         gpu_ids=gpu_ids,
        #                         logger=logger)
        # # end for
        # raise()
        for r_i, req in enumerate(reqs):
            print(req)
            cls.generate_inputs(nlp,
                                nlp_task,
                                req,
                                pcfg_ref,
                                dataset_name,
                                res_dir, # cfg_res_file,
                                num_seeds=num_seeds,
                                selection_method=selection_method,
                                gpu_ids=gpu_ids,
                                logger=logger)
        # end for
        return

    # @classmethod
    # def find_pos_from_cfg_seed(cls, token, cfg_seed):
    #     # when tokenized token can be found in the leaves of cfg
    #     for key, vals in cfg_seed.items():
    #         for val in vals:
    #             if val["pos"]==val["word"] and token==val["word"]:
    #                 return key
    #             # end if
    #         # end for
    #     # end for
    #     # when tokenized token cannot be found in the leaves of cfg
    #     for key, vals in cfg_seed.items():
    #         for val in vals:
    #             if type(val["word"])==list and token in val["word"]:
    #                 tok_idx = val["word"].index(token)
    #                 return val["pos"][tok_idx]
    #             # end if
    #         # end for
    #     # end for
    #     return
    
    # @classmethod
    # def get_pos(cls,
    #             mask_input: str,
    #             mask_pos: List[str],
    #             cfg_seed: Dict,
    #             words_sug: List[str],
    #             exp_input:str):
    #     tokens = Utils.tokenize(mask_input)
    #     _tokens = list()
    #     tokens_pos = list()
    #     tok_i, mask_tok_i = 0, 0
    #     while tok_i<len(tokens):
    #         if tokens[tok_i:tok_i+3]==['{', 'mask', '}']:
    #             _tokens.append('{mask}')
    #             tok_i += 3
    #         else:
    #             _tokens.append(tokens[tok_i])
    #             tok_i += 1
    #         # end if
    #     # end for
    #     tokens = _tokens
        
    #     for t in tokens:
    #         if t=="{mask}":
    #             if type(words_sug)==str:
    #                 tpos = words_sug
    #             elif ((type(words_sug)==list) or (type(words_sug)==tuple)):
    #                 tpos = mask_pos[mask_tok_i]
    #                 mask_tok_i += 1
    #             # end if
    #         else:
    #             tpos = cls.find_pos_from_cfg_seed(t, cfg_seed)
    #         # end if
    #         tokens_pos.append(tpos)
    #     # end for
    #     return Utils.tokenize(exp_input), tokens_pos

    # @classmethod
    # def get_templates_by_synonyms(cls,
    #                               nlp,
    #                               tokens: List[str],
    #                               tokens_pos: List[str],
    #                               prev_synonyms):
    #     template = list()
    #     for t, tpos in zip(tokens, tokens_pos):
    #         newt = re.sub(r'\..*', '', t)
    #         newt = re.sub(r'\[.*\]', '', newt)
    #         newt = re.sub(r'.*?:', '', newt)
    #         newt = re.sub(r'\d+$', '', newt)
    #         key = "{"+f"{newt}_{tpos}"+"}"
    #         if key in prev_synonyms.keys():
    #             if prev_synonyms[key] is None or len(prev_synonyms[key])==0:
    #                 template.append(t)
    #             else:
    #                 template.append({
    #                     key: prev_synonyms[key]
    #                 })
    #             # end if
    #         else:
    #             syns = Synonyms.get_synonyms(nlp, t, tpos)
    #             if len(syns)>1:
    #                 _syns = list()
    #                 for s in syns:
    #                     if len(s.split("_"))>1:
    #                         _syns.append(" ".join(s.split("_")))
    #                     else:
    #                         _syns.append(s)
    #                     # end if
    #                 # end for
    #                 syns_dict = {key: _syns}
    #                 template.append(syns_dict)
    #                 if key not in prev_synonyms.keys():
    #                     prev_synonyms[key] = syns_dict[key]
    #                 # end if
    #             else:
    #                 template.append(t)
    #                 if key not in prev_synonyms.keys():
    #                     prev_synonyms[key] = None
    #                 # end if
    #             # end if
    #         # end if
    #     # end for
    #     return {
    #         "input": Utils.detokenize(tokens),
    #         "place_holder": template
    #     }, prev_synonyms

    @classmethod
    def get_templates(cls,
                      nlp_task,
                      dataset_name,
                      selection_method,
                      num_seeds,
                      num_trials,
                      gpu_ids,
                      log_file):
        assert nlp_task in Macros.nlp_tasks
        assert dataset_name in Macros.datasets[nlp_task]
        if num_seeds<0:
            template_out_dir = f"templates{num_trials}_{nlp_task}_{dataset_name}_{selection_method}"
        else:
            template_out_dir = f"templates{num_trials}_{nlp_task}_{dataset_name}_{selection_method}_{num_seeds}seeds"
        # end if
        res_dir = Macros.result_dir / template_out_dir
        res_dir.mkdir(parents=True, exist_ok=True)
        logger = Logger(logger_file=log_file,
                        logger_name='template')
        
        # logger.print(f"***** TASK: {nlp_task}, SEARCH_DATASET: {dataset_name}, SELECTION: {selection_method} *****")
        # # Search inputs from searching dataset and expand the inputs using ref_cfg
        # cls.get_new_inputs(
        #     res_dir, # cfg_res_file,
        #     nlp_task,
        #     dataset_name,
        #     num_seeds=num_seeds,
        #     selection_method=selection_method,
        #     gpu_ids=gpu_ids,
        #     logger=logger
        # )

        # Make templates by synonyms
        logger.print("Generate Templates ...")
        reqs = Requirements.get_requirements(nlp_task)
        prev_synonyms = dict()
        cksum_map_str = ""
        reqs = [
            r for r in reqs
            if r.get('use_testcase', None)!='hatecheck'
        ]
        for t_i, req in enumerate(reqs):
            # for each testing linguistic capabilities,
            lc_desc = req['description']
            req_cksum = Utils.get_cksum(lc_desc)
            cfg_res_file = res_dir / f"cfg_expanded_inputs_{req_cksum}.json"
            inputs_per_req = Utils.read_json(cfg_res_file)
            cksum_map_str += f"{lc_desc}\t{req_cksum}\n"
            inputs = inputs_per_req["inputs"]
            if inputs_per_req["requirement"].get('use_testcase', None)!='hatecheck':
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
                            is_valid = True
                            exp_sent = inp[5]
                            mask_exp_sent = inp[0]
                            if exp_sent is not None:
                                # if req.get('transform', None) and \
                                #    not Validate.is_conform_to_template(
                                #        sent=mask_exp_sent,
                                #        label=label_seed,
                                #        transform_spec=req['transform']):
                                #     is_valid = False
                                # # end if
                                if is_valid:
                                    exp_seed_inputs.append({
                                        "input": inp[5],
                                        "place_holder": Utils.tokenize(inp[5]),
                                        "label": label_seed
                                    })
                                # end if
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
            # end if
        # end for
        Utils.write_txt(cksum_map_str, res_dir / 'cksum_map.txt')
        return


# Write templates
# Template.get_templates(num_seeds=10)
