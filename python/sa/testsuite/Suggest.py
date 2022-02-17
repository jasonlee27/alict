# This script is to generate new sentences
# given a seed input and their expanded CFGs.
# using BERT-like language model for word
# suggestion

from typing import *

import re, os
import nltk
import spacy
import copy
import random
import numpy

from pathlib import Path
from scipy.special import softmax
# from nltk.tokenize import word_tokenize as tokenize

import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb

from ..utils.Macros import Macros
from ..utils.Utils import Utils
# from .cfg.CFG import BeneparCFG
from .Search import SearchOperator, SENT_DICT

random.seed(Macros.SEED)
NUM_TOPK = 5

class Suggest:

    MASK = Macros.MASK

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
    def get_word_suggestion(cls, editor: Editor, gen_inputs, num_target=10):
        results = dict()
        masked_inputs = cls.get_masked_inputs(gen_inputs)
        for m_inp in masked_inputs:
            word_suggest = editor.suggest(m_inp, return_score=True, remove_duplicates=True)
            word_suggest = sorted(word_suggest, key=lambda x: x[-1], reverse=True)[:num_target]
            word_suggest = [ws for ws in word_suggest if cls.is_word_suggestion_avail(ws[0])]
            word_suggest = cls.remove_duplicates(word_suggest)
            results[m_inp] = word_suggest
        # end for
        return results
    
    @classmethod
    def match_word_n_pos(cls, nlp, word_suggest, masked_input: str, mask_pos: List, num_target=10):
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
            return suggest_res[:num_target]
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
        if len(search_res)>0:
            return True
        # end if
        return False

    @classmethod
    def eval_sug_words_by_exp_req(cls, word_suggest, requirement):
        if requirement['expansion'] is None:
            return True
        # end if
        all_reqs_met = list()
        for r in requirement['expansion']:
            is_req_met = False
            if len(r.split())>1:
                is_req_met = word_suggest in SENT_DICT[r]
            else:
                if r in list(Macros.sa_label_map.keys()):
                    for key in SENT_DICT.keys():
                        if key.startswith(r):
                            is_req_met = word_suggest in SENT_DICT[key]
                        # end if
                    # end for
                # end if
            # end if
            all_reqs_met.append(is_req_met)
        # end for
        return all(all_reqs_met)

    @classmethod
    def get_words_by_prob(cls, words_suggest, gen_input, masked_input):
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
            # sent = cls.replace_mask_w_suggestion(masked_input, ws_sug)
            sent_probs.append((
                ws_sug,
                sent_prob_wo_target*prob*prob_ws
            ))
        # end for
        sent_probs = sorted(sent_probs, key=lambda x: x[-1], reverse=True)
        sent_probs = [s[0] for s in sent_probs[:NUM_TOPK]]
        return sent_probs
    
    @classmethod
    def get_new_inputs(cls, generator, gen_inputs, num_target=10, is_random_select=False):
        editor = generator.editor
        nlp = spacy.load('en_core_web_md')
        word_suggestions = cls.get_word_suggestion(editor, gen_inputs, num_target=3*num_target)
        for g_i in range(len(gen_inputs)):
            # print(f"gen_new_inputs: {g_i} out of {len(gen_inputs)}")
            gen_input = gen_inputs[g_i]
            masked_input, mask_pos = gen_input['masked_input']
            words_suggest = cls.match_word_n_pos(
                nlp,
                word_suggestions[masked_input],
                masked_input,
                mask_pos,
                num_target=num_target
            )
            if not is_random_select:
                gen_input['words_suggest'] = cls.get_words_by_prob(words_suggest, gen_input, masked_input)
            else:
                gen_input['words_suggest'] = [ws[0] for ws in words_suggest]
            # end if
            gen_inputs[g_i] = gen_input
        # end for
        gen_inputs = [g for g in gen_inputs if g['words_suggest'] is not None]
        num_words_suggest = sum([len(g['words_suggest']) for g in gen_inputs])
        print(f"{num_words_suggest} words suggestions", end=" :: ")
        return gen_inputs

    @classmethod
    def eval_word_suggest(cls, gen_input, label: str, requirement):
        results = list()
        masked_input, mask_pos = gen_input['masked_input']
        if not gen_input['words_suggest']:
            results.append((masked_input,
                            gen_input['cfg_from'],
                            gen_input['cfg_to'],
                            mask_pos,
                            None,
                            None,
                            label))
        else:
            for w_sug in gen_input['words_suggest']:
                input_candid = cls.replace_mask_w_suggestion(masked_input, w_sug)
                # check sentence and expansion requirements
                if cls.eval_sug_words_by_req(input_candid, requirement, label):
                    if cls.eval_sug_words_by_exp_req(w_sug, requirement):
                        results.append((masked_input,
                                        gen_input['cfg_from'],
                                        gen_input['cfg_to'],
                                        mask_pos,
                                        w_sug,
                                        input_candid,
                                        label))
                    # end if
                # end if
            # end for
        # end if
        return results

    @classmethod
    def get_exp_inputs(cls, generator, gen_inputs, seed_label, requirement, num_target=10, is_random_select=False):
        # get the word suggesteion at the expended grammar elements
        new_input_results = list()
        gen_inputs = cls.get_new_inputs(
            generator, gen_inputs, num_target=num_target, is_random_select=is_random_select
        )
        for g_i in range(len(gen_inputs)):
            eval_results = cls.eval_word_suggest(gen_inputs[g_i], seed_label, requirement)
            if any(eval_results):
                del gen_inputs[g_i]["words_suggest"]
                new_input_results.extend(eval_results)
                # num_seed_for_exp += 1
                # print(f"eval: {g_i} out of {len(gen_inputs)}", end="")
                # print(".", end="")
            # end if
        # end for
        # print()
        return new_input_results

        

# def main():
#     return

# if __name__=='__main__':
#     main()
