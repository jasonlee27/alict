# This script is to generate new sentences
# given a seed input and their expanded CFGs.
# using BERT-like language model for word
# suggestion

from typing import *

import re, os
import nltk
import spacy
import copy
import time
import random
import numpy as np
import multiprocessing

from pathlib import Path
from scipy.special import softmax
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
# from nltk.tokenize import word_tokenize as tokenize

import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb

from .Validate import Validate
from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..seed.Search import SearchOperator, SENT_DICT

NUM_TOPK = 5
CPU_GPU_MAP = dict()

class Suggest:

    MASK = Macros.MASK
    NUM_PROCESSES = Macros.num_processes

    @classmethod
    def is_word_suggestion_avail(cls, word_suggest):
        if type(word_suggest) in [tuple, list]:
            empty = [True for w in word_suggest if w=='']
            non_letters = [re.search(r"[^A-Za-z0-9]+", w) for w in word_suggest]
            if (not any(empty)) and (not any(non_letters)):
                return True
            # end if
        else:
            empty = True if word_suggest=='' else None
            non_letters = re.search(r"[^A-Za-z0-9]+", word_suggest)
            if non_letters is None and empty is None:
                return True
            # end if
        # end if
        return False

    @classmethod
    def remove_duplicates(cls, word_suggest):
        results = list()
        for words, score in word_suggest:
            if (words, score) not in results:
                results.append((words, score))
            # end if
        # end for
        return results

    @classmethod
    def get_masked_inputs(cls, gen_inputs):
        return set([inp['masked_input'][0] for inp in gen_inputs])

    @classmethod
    def get_word_suggestion(cls,
                            editor: Editor,
                            gen_inputs,
                            num_target=10):
        results = dict()
        # mi_st = time.time()
        masked_inputs = cls.get_masked_inputs(gen_inputs)
        # mi_ft = time.time()
        # print(f"cls.get_word_suggestion::get_masked_inputs:<{round(mi_ft-mi_st,2)} seconds>::")
        # for_st = time.time()
        for m_inp_i, m_inp in enumerate(masked_inputs):
            # ws_st = time.time()
            word_suggest = editor.suggest(m_inp, return_score=True, remove_duplicates=True)
            word_suggest = sorted(word_suggest, key=lambda x: x[-1], reverse=True)[:num_target]
            word_suggest = [ws for ws in word_suggest if cls.is_word_suggestion_avail(ws[0])]
            word_suggest = cls.remove_duplicates(word_suggest)
            results[m_inp] = word_suggest
            # ws_ft = time.time()
            # print(f"cls.get_word_suggestion::masked_inputs{m_inp_i}::{m_inp}::<{round(ws_ft-ws_st,2)} seconds>::")
        # end for
        # for_ft = time.time()
        # print(f"cls.get_word_suggestion::masked_inputs::for_loop<{round(for_ft-for_st,2)} seconds>::")
        return results
    
    @classmethod
    def match_word_n_pos(cls, nlp, word_suggest, masked_input: str, mask_pos: List):
        # word_suggest = editor.suggest(masked_input, return_score=True, remove_duplicates=True)
        # word_suggest = [ws for ws in word_suggest if cls.is_word_suggestion_avail(ws[0])]
        # word_suggest = cls.remove_duplicates(word_suggest)
        
        # find suggested words with the same pos as we want.
        # and normalizing scores of selected word suggestion
        suggest_res = list()
        for ws_sug, score in word_suggest:
            words_sug_pos, word_sug_prs_string = cls.get_sug_words_pos(nlp, ws_sug)
            if cls.eval_sug_words_by_pos(words_sug_pos, mask_pos):
                suggest_res.append((ws_sug, score))
            # end if
        # end for
        if any(suggest_res):
            probs = softmax([s[1] for s in suggest_res])
            suggest_res = [(s[0],p) for s, p in zip(suggest_res, probs)]
            return suggest_res
        # end if
        return suggest_res

    @classmethod
    def find_all_mask_placeholder(cls, masked_input, target_pattern):
        result = list()
        for match in re.finditer(target_pattern, masked_input):
            result.append((match.start(), match.end()))
        # end for
        return result
        
    @classmethod
    def replace_mask_w_suggestion(cls, masked_input, words_suggest):
        # masked_tok_is = cls.find_all_mask_placeholder(masked_input, cls.MASK)
        _masked_input = masked_input
        if type(words_suggest)==str:
            words_suggest = [words_suggest]
        # end i
        # for t_is, w in zip(masked_tok_is, words_suggest):
        for w in words_suggest:
            search = re.search(r'\{mask\}', _masked_input)
            if search:
                _masked_input = _masked_input[:search.start()] + w + _masked_input[search.end():]
            # end if
        # end for
        return _masked_input

    @classmethod
    def get_word_pos(cls, nlp, word):
        doc = nlp(word)
        return str(doc[0]), doc[0].tag_
        # try:
        #     tree = BeneparCFG.get_word_pos(word)
        #     parse_string = tree._.parse_string
        #     pattern = r"\(([^\:|^\(|^\)]+)\s"+word+r"\)"
        #     search = re.search(pattern, parse_string)
        #     if search:
        #         return word, search.group(1)
        #     # end if
        #     return None, None
        # except IndexError:
        #     print(f"IndexError: {word}")
        #     return None, None

    @classmethod
    def get_sug_words_pos(cls, nlp, word_suggest):
        prs_str, pos = list(), list()
        if type(word_suggest)==str:
            word_suggest = [word_suggest]
        # end if
        for ws in word_suggest:
            parse_string, w_pos = cls.get_word_pos(nlp, ws)
            if parse_string is None or w_pos is None:
                return None, None
            # end if
            pos.append(w_pos)
            prs_str.append(parse_string)
        # end for
        return pos, prs_str

    @classmethod
    def eval_sug_words_by_pos(cls, words_sug_pos, mask_pos):
        match = list()
        if words_sug_pos is None:
            return False
        # end if
        for w_pos, m_pos in zip(words_sug_pos, mask_pos):
            if w_pos==m_pos:
                match.append(True)
            else:
                match.append(False)
            # end if
        # end for
        if all(match):
            return True
        # end if
        return False
    
    @classmethod
    def eval_sug_words_by_req(cls, new_input, requirement, label):
        if requirement['transform_req'] is not None:
            _requirement = {
                'capability': requirement['capability'],
                'description': requirement['description'],
                'search': requirement['transform_req']
            }
        else:
            _requirement = requirement
        # end if
        search_obj = SearchOperator(_requirement)
        search_res = search_obj.search([('1', new_input, label)])
        return any(search_res)
        # if len(search_res)>0:
        #     return True
        # # end if
        # return False

    @classmethod
    def eval_sug_words_by_exp_req(cls, nlp, word_suggest, requirement):
        # find relevant expanded words.
        # in this work, we accept neutral words
        word_suggest = [word_suggest] if type(word_suggest)==str else list(word_suggest)
        if requirement['expansion'] is None:
            return True
        # end if
        all_reqs_met_over_words = list()
        for ws in word_suggest:
            all_reqs_met_over_reqs = list()
            for r in requirement['expansion']:
                is_req_met = False
                if len(r.split())>1:
                    is_req_met = ws in SENT_DICT[r]
                else:
                    if r in list(Macros.sa_label_map.keys()):
                        sentiment_list = [key for key in SENT_DICT.keys() if ws in SENT_DICT[key]]
                        if not any(sentiment_list):
                            is_req_met = False
                        elif len(sentiment_list)==1:
                            if sentiment_list[0].startswith(r):
                                is_req_met = True
                            # end if
                        elif len(sentiment_list)>1:
                            w_tag = nlp(ws)[0].pos_.lower()
                            for key in sentiment_list:
                                if key.endswith(w_tag) and key.startswith(r):
                                    is_req_met = True
                                    break
                                # end if
                            # end for
                        # end if
                    # end if
                # end if
                all_reqs_met_over_reqs.append(is_req_met)
            # end for
            all_reqs_met_over_words.append(all(all_reqs_met_over_reqs))
        # end for
        return all(all_reqs_met_over_words)

    @classmethod
    def get_words_by_prob(cls, words_suggest, gen_input, masked_input, num_target):
        if not any(words_suggest):
            return None
        # end if
        lhs = gen_input['lhs']
        cfg_rhs_from = gen_input['cfg_from'].split(f"{lhs} -> ")[-1]
        cfg_rhs_to = gen_input['cfg_to'].split(f"{lhs} -> ")[-1]
        prob = gen_input['prob']
        sent_prob_wo_target = gen_input['sent_prob_wo_target']
        # cfg_diff = generator.expander.cfg_diff[lhs][cfg_rhs_from]
        # prob, sent_prob_wo_target = -1.,-1.
        # for diff in cfg_diff:
        #     if diff[0]==cfg_rhs_to:
        #         prob, sent_prob_wo_target = diff[1], diff[2]
        #         break
        #     # end if
        # # end for

        sent_probs = list()
        for ws_sug, prob_ws in words_suggest:
            sent = cls.replace_mask_w_suggestion(masked_input, ws_sug)
            sent_probs.append((
                ws_sug,
                sent_prob_wo_target*prob*prob_ws
            ))
        # end for
        sent_probs = sorted(sent_probs, key=lambda x: x[-1], reverse=True)
        sent_probs = [s[0] for s in sent_probs[:num_target]]
        return sent_probs
    
    @classmethod
    def get_new_inputs(cls,
                       nlp,
                       editor,
                       generator,
                       gen_inputs,
                       selection_method,
                       num_target=10,
                       logger=None):
        # editor = generator.editor
        # ws_st = time.time()
        word_suggestions = cls.get_word_suggestion(editor, gen_inputs, num_target=3*num_target)
        # ws_ft = time.time()
        # print(f"cls.get_new_inputs::cls.get_word_suggestion<{round(ws_ft-ws_st,2)} seconds>:: ")
        # mw_st = time.time()
        for g_i in range(len(gen_inputs)):
            gen_input = gen_inputs[g_i]
            masked_input, mask_pos = gen_input['masked_input']
            words_suggest = cls.match_word_n_pos(
                nlp,
                word_suggestions[masked_input],
                masked_input,
                mask_pos,
            )
            if selection_method.lower()=='prob':
                # TODO: fix the probability issue with lower prob when deep tree
                gen_input['words_suggest'] = cls.get_words_by_prob(
                    words_suggest,
                    gen_input,
                    masked_input,
                    num_target=num_target
                )
            elif selection_method.lower()=='random':
                if len(words_suggest)>num_target:
                    idxs = np.random.choice(len(words_suggest), num_target, replace=False)
                    gen_input['words_suggest'] = [words_suggest[i][0] for i in idxs]
                else:
                    gen_input['words_suggest'] = [ws[0] for ws in words_suggest]
                # end if
            elif selection_method.lower()=='bertscore':
                if len(words_suggest)>num_target:
                    word_suggest_sort_by_bertscore = sorted(
                        words_suggest, key=lambda x: x[-1], reverse=True
                    )[:num_target]
                    gen_input['words_suggest'] = [ws[0] for ws in word_suggest_sort_by_bertscore]
                else:
                    gen_input['words_suggest'] = [ws[0] for ws in words_suggest]
                # end if
            elif selection_method.lower()=='noselect':
                gen_input['words_suggest'] = [ws[0] for ws in words_suggest]
            # end if
            gen_inputs[g_i] = gen_input
        # end for
        # mw_ft = time.time()
        # print(f"cls.get_new_inputs::match_word_n_pos<{round(mw_ft-mw_st,2)} seconds>:: ")
        gen_inputs = [g for g in gen_inputs if g['words_suggest'] is not None]
        num_words_suggest = sum([len(g['words_suggest']) for g in gen_inputs])
        # logger.print(f"{num_words_suggest} words suggestions :: ", end='')
        return gen_inputs, num_words_suggest

    @classmethod
    def eval_word_suggest(cls, nlp, gen_input, label: str, requirement):
        results = list()
        masked_input, mask_pos = gen_input['masked_input']
        if gen_input['words_suggest']:
            for w_sug in gen_input['words_suggest']:
                input_candid = cls.replace_mask_w_suggestion(masked_input, w_sug)
                # check sentence and expansion requirements
                if cls.eval_sug_words_by_req(input_candid, requirement, label):
                    if cls.eval_sug_words_by_exp_req(nlp, w_sug, requirement):
                        if requirement.get('transform', None) is not None and \
                           Validate.is_conform_to_template(
                               sent=masked_input,
                               transform_spec=requirement['transform']):
                            results.append((masked_input,
                                            gen_input['cfg_from'],
                                            gen_input['cfg_to'],
                                            mask_pos,
                                            w_sug,
                                            input_candid,
                                            label))
                        # end if
                    # end if
                # end if
            # end for
        # end if
        # if not gen_input['words_suggest']:
        #     results.append((masked_input,
        #                     gen_input['cfg_from'],
        #                     gen_input['cfg_to'],
        #                     mask_pos,
        #                     None,
        #                     None,
        #                     label))
        # else:
        #     for w_sug in gen_input['words_suggest']:
        #         input_candid = cls.replace_mask_w_suggestion(masked_input, w_sug)
        #         # check sentence and expansion requirements
        #         if cls.eval_sug_words_by_req(input_candid, requirement, label):
        #             if cls.eval_sug_words_by_exp_req(nlp, w_sug, requirement):
        #                 results.append((masked_input,
        #                                 gen_input['cfg_from'],
        #                                 gen_input['cfg_to'],
        #                                 mask_pos,
        #                                 w_sug,
        #                                 input_candid,
        #                                 label))
        #             # end if
        #         # end if
        #     # end for
        # # end if
        return results

    @classmethod
    def get_exp_inputs(cls,
                       editor,
                       generator,
                       gen_inputs,
                       seed_label,
                       requirement,
                       selection_method,
                       num_target=10,
                       logger=None):
        # get the word suggesteion at the expended grammar elements
        new_input_results = list()
        nlp = spacy.load('en_core_web_md')
        nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
        gen_inputs, num_words_orig_suggest = cls.get_new_inputs(
            nlp,
            editor,
            generator,
            gen_inputs,
            num_target=num_target,
            selection_method=selection_method,
            logger=logger
        )
        # ews_st = time.time()
        for g_i in range(len(gen_inputs)):
            eval_results = cls.eval_word_suggest(nlp,
                                                 gen_inputs[g_i],
                                                 seed_label,
                                                 requirement)
            if any(eval_results):
                del gen_inputs[g_i]["words_suggest"]
                new_input_results.extend(eval_results)
                # num_seed_for_exp += 1
                # print(f"eval: {g_i} out of {len(gen_inputs)}", end="")
                # print(".", end="")
            # end if
        # end for
        # ews_ft = time.time()
        # print(f"cls.get_exp_inputs::eval_word_suggest<{round(ews_ft-ews_st,2)} seconds>:: ")
        # print()
        return new_input_results, num_words_orig_suggest

    @classmethod
    def get_word_sug_parallel(cls,
                              editors,
                              ms_i,
                              masked_sent,
                              num_target,
                              no_mask_key,
                              gpu_id,
                              logger):
        st = time.time()
        pcs_id = multiprocessing.current_process().ident
        if gpu_id is not None:
            editor = editors[f"{gpu_id}"]
        else:
            editor = editors
        # end if
        word_suggest = list()
        success = False
        if masked_sent!=no_mask_key:
            try:
                word_suggest = editor.suggest(
                    masked_sent,
                    return_score=True,
                    remove_duplicates=True
                )[:3*num_target]
                word_suggest = [
                    ws for ws in word_suggest
                    if cls.is_word_suggestion_avail(ws[0])
                ]
                word_suggest = cls.remove_duplicates(word_suggest)
                success = True
            except RuntimeError:
                # print(f"Suggest.get_word_sug_parallel::CUDA_OOM::MASKED_SENT_{ms_i}::{masked_sent}::pcs{pcs_id}::gpu{gpu_id}")
                word_suggest = list()
                pass
            # end try
        # end if
        ft = time.time()
        if logger is not None:
            logger.print(f"\tSuggest.get_word_sug_parallel::MASKED_SENT_{ms_i}::{masked_sent}::{round(ft-st,3)}sec::pcs{pcs_id}::gpu{gpu_id}::{success}")
        # end if
        return {
            'masked_sent': masked_sent,
            'word_sug': word_suggest
        }
        
    @classmethod
    def get_word_suggestions_over_seeds(cls,
                                        masked_sents: Dict,
                                        num_target=10,
                                        selection_method=None,
                                        no_mask_key='<no_mask>',
                                        cuda_device_inds=None,
                                        logger=None):
        st = time.time()
        results = list()
        args = list()
        editor, editors = None, None
        if logger is not None:
            logger.print(f"\tSuggest.get_word_suggestions_over_seeds::{len(masked_sents.keys())} masked sents identified")
        # end if
        if cuda_device_inds is not None:
            assert len(cuda_device_inds)==cls.NUM_PROCESSES
            editors = {
                f"{gpu_id}": Editor(cuda_device_ind=gpu_id)
                for gpu_id in range(len(cuda_device_inds))
            }
            num_sents_per_gpu = len(masked_sents.keys())//cls.NUM_PROCESSES

            for ms_i, masked_sent in enumerate(masked_sents.keys()):
                start = 0
                gpu_id = None
                for c_i, g_i in enumerate(cuda_device_inds):
                    if c_i+1 == len(cuda_device_inds):
                        end = len(masked_sents.keys())
                    else:
                        end = start + num_sents_per_gpu
                    # end if
                    if ms_i in range(start, end):
                        gpu_id = c_i
                        break
                    # end if
                    start = end
                # end for
                args.append((
                    editors, ms_i, masked_sent, num_target, no_mask_key, gpu_id, logger
                ))
            # end for
            if logger is None:
                logger.print(f"\tSuggest.get_word_suggestions_over_seeds::multiprocessing.Pool..")
            # end if
            num_pcss = cls.NUM_PROCESSES if len(args)>=cls.NUM_PROCESSES else 1
            pool = multiprocessing.Pool(processes=num_pcss)
            results = pool.starmap_async(
                cls.get_word_sug_parallel,
                args,
                chunksize=len(args)//num_pcss
            ).get()
            for r in results:
                if any(r['word_sug']):
                    masked_sents[r['masked_sent']]['word_sug'] = r['word_sug']
                # end if
            # end for
            pool.close()
            pool.join()
        else:
            editor = Editor()
            for ms_i, masked_sent in enumerate(masked_sents.keys()):
                r = cls.get_word_sug_parallel(
                    editor, 
                    ms_i, 
                    masked_sent, 
                    num_target, 
                    no_mask_key, 
                    None, 
                    logger
                )
                if any(r['word_sug']):
                    masked_sents[r['masked_sent']]['word_sug'] = r['word_sug']
                # end if
            # end for
        # end if
        ft = time.time()
        if logger is None:
            logger.print(f"\tSuggest.get_word_suggestions_over_seeds::{round(ft-st,3)}sec")
        # end if
        return masked_sents

    @classmethod
    def _eval_word_suggestions_over_seeds_parallel(cls,
                                                   masked_sent,
                                                   word_sug,
                                                   seed_id,
                                                   seed_objs,
                                                   req,
                                                   num_target=Macros.num_suggestions_on_exp_grammer_elem,
                                                   selection_method=None,
                                                   logger=None):
        st = time.time()
        results = list()
        pcs_id = multiprocessing.current_process().ident
        nlp = spacy.load('en_core_web_md')
        nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
        for seed, seed_label, seed_score, cfg_seed, cfg_from, cfg_to, mask_pos in seed_objs:
            matched_words_sug = cls.match_word_n_pos(
                nlp,
                word_sug,
                masked_sent,
                mask_pos
            )
            if selection_method.lower()=='random':
                if len(matched_words_sug)>num_target:
                    idxs = np.random.choice(len(matched_words_sug), num_target, replace=False)
                    word_suggest = [matched_words_sug[i][0] for i in idxs]
                else:
                    word_suggest = [ws[0] for ws in matched_words_sug]
                # end if
            elif selection_method.lower()=='bertscore':
                if len(matched_words_sug)>num_target:
                    word_suggest = sorted(matched_words_sug,
                                          key=lambda x: x[-1],
                                          reverse=True)[:num_target]
                # end if
                word_suggest = [ws[0] for ws in matched_words_sug]
            else: # noselect
                word_suggest = [ws[0] for ws in matched_words_sug]
            # end if
            for w_sug in word_suggest:
                input_candid = cls.replace_mask_w_suggestion(masked_sent, w_sug)
                # check sentence and expansion requirements
                if cls.eval_sug_words_by_req(input_candid, req, seed_label):
                    if cls.eval_sug_words_by_exp_req(nlp, w_sug, req):
                        results.append((seed,
                                        seed_label,
                                        seed_score,
                                        masked_sent,
                                        cfg_seed,
                                        cfg_from,
                                        cfg_to,
                                        mask_pos,
                                        w_sug,
                                        input_candid))
                    # end if
                # end if
            # end for
        # end for
        ft = time.time()
        print(f"\tSuggest._eval_word_suggestions_over_seeds_parallel::SEED{seed_id}::{round(ft-st,3)}sec::pcs{pcs_id}")
        if logger is not None:
            logger.print(f"\tSuggest._eval_word_suggestions_over_seeds_parallel::SEED{seed_id}::{round(ft-st,3)}sec::pcs{pcs_id}")
        # end if
        return results

    @classmethod
    def eval_word_suggestions_over_seeds(cls,
                                         masked_inputs_w_word_sug,
                                         req,
                                         cfg_res_file,
                                         num_target=Macros.num_suggestions_on_exp_grammer_elem,
                                         selection_method=None,
                                         no_mask_key='<no_mask>',
                                         logger=None):
        st = time.time()
        template_results = Utils.read_json(cfg_res_file)
        args = list()
        for m_i, masked_sent in enumerate(masked_inputs_w_word_sug.keys()):
            word_sug = masked_inputs_w_word_sug[masked_sent]['word_sug']
            seed_objs = masked_inputs_w_word_sug[masked_sent]['inputs']
            if masked_sent!=no_mask_key:
                args.append((
                    masked_sent,
                    word_sug,
                    m_i,
                    seed_objs,
                    req,
                    num_target,
                    selection_method,
                    logger
                ))
            # end if
        # end for
        if any(args):
            verified_template_results = template_results
            num_pcss = cls.NUM_PROCESSES*2 if len(args)>=cls.NUM_PROCESSES*2 else 1
            pool = multiprocessing.Pool(processes=num_pcss)
            results = pool.starmap_async(cls._eval_word_suggestions_over_seeds_parallel,
                                         args,
                                         chunksize=len(args)//num_pcss).get()
            for r in results:
                # remove verified masked inputs
                if any(r):
                    seed, seed_label, seed_score, \
                        masked_sent, cfg_seed, cfg_from, \
                        cfg_to, mask_pos, w_sug, input_candid = r[0]
                    m_inds = list()
                    key = verified_template_results['inputs'].get(seed, None)
                    if key is None:
                        verified_template_results['inputs'][seed] = {
                            'cfg_seed': cfg_seed,
                            'exp_inputs': list(),
                            'label': seed_label,
                            'label_score': seed_score
                        }
                    # end if
                    verified_template_results['inputs'][seed]['exp_inputs'].append((
                        masked_sent,
                        cfg_from,
                        cfg_to,
                        mask_pos,
                        w_sug,
                        input_candid
                    ))
                    if 'masked_inputs' in verified_template_results['inputs'][seed].keys():
                        target_ind = [
                            s_i
                            for s_i, s in enumerate(verified_template_results['inputs'][seed]['masked_inputs'])
                            if s['cfg_from']==cfg_from and s['cfg_to']==cfg_to and s['masked_input'][0]==masked_sent and s['masked_input'][1]==mask_pos
                        ]
                        del verified_template_results['inputs'][seed]['masked_inputs'][target_ind[0]]
                        if not any(verified_template_results['inputs'][seed]['masked_inputs']):
                            del verified_template_results['inputs'][seed]['masked_inputs']
                        # end if
                    # end if
                # end if
            # end for
            pool.close()
            pool.join()

            for seed in verified_template_results['inputs'].keys():
                if 'masked_inputs' in verified_template_results['inputs'][seed].keys():
                    del verified_template_results['inputs'][seed]['masked_inputs']
                # end if
            # end for
            Utils.write_json(verified_template_results, cfg_res_file, pretty_format=True)
        # end if
        ft = time.time()
        if logger is not None:
            logger.print(f"\tSuggest.eval_word_suggestions_over_seeds::{round(ft-st,3)}sec")
        # end if
        return
