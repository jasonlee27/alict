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
# from nltk.tokenize import word_tokenize as tokenize

import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb

from Macros import Macros
from Utils import Utils
from CFG import BeneparCFG
from Search import SearchOperator
# from Requirements import Requirements

random.seed(Macros.SEED)

class Suggest:

    MASK = Macros.MASK

    @classmethod
    def is_word_suggestion_not_avail(cls, word_suggest):
        if type(word_suggest)==tuple or type(word_suggest)==list:
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
    def get_word_suggestion(cls, editor: Editor, masked_input: str, mask_pos: List, num_target=10):
        word_suggest = editor.suggest(masked_input, remove_duplicates=True)
        suggest_res = [ws for ws in word_suggest if cls.is_word_suggestion_not_avail(ws)]
        return suggest_res[:num_target]

    @classmethod
    def find_all_mask_placeholder(cls, masked_input, target_pattern):
        result = list()
        for match in re.finditer(target_pattern, masked_input):
            result.append((match.start(), match.end()))
        # end for
        return result
        
    @classmethod
    def replace_mask_w_suggestion(cls, masked_input, words_suggest):
        masked_tok_is = cls.find_all_mask_placeholder(masked_input, cls.MASK)
        _masked_input = masked_input
        if type(words_suggest)==str:
            words_suggest = [words_suggest]
        # end i
        for t_is, w in zip(masked_tok_is, words_suggest):
            search = re.search(r'\{mask\}', _masked_input)
            if search:
                _masked_input = _masked_input[:search.start()] + w + _masked_input[search.end():]
            # end if
        # end for
        return _masked_input

    @classmethod
    def get_word_pos(cls, word):
        try:
            tree = BeneparCFG.get_word_pos(word)
            parse_string = tree._.parse_string
            pattern = r"\(([^\:|^\(|^\)]+)\s"+word+r"\)"
            search = re.search(pattern, parse_string)
            if search:
                return parse_string, search.group(1)
            # end if
            return None, None
        except IndexError:
            print(f"IndexError: {word}")
            return None, None

    @classmethod
    def get_sug_words_pos(cls, word_suggest):
        prs_str, pos = list(), list()
        if type(word_suggest)==str:
            word_suggest = [word_suggest]
        # end if
        for ws in word_suggest:
            parse_string, w_pos = cls.get_word_pos(ws)
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
        search_obj = SearchOperator(requirement)
        search_res = search_obj.search([("1", new_input, label)])
        if len(search_res)>0:
            return True
        # end if
        return False

    # @classmethod
    # def get_new_input(cls, editor, masked_input: str, label: str, requirement, num_target=10):
    #     # print(f"\n>>>>> {masked_input}, {mask_pos}")
    #     words_suggest = cls.get_word_suggestion(editor, masked_input, mask_pos, num_target=num_target)
    #     for w_sug in words_suggest:
    #         words_sug_pos, word_sug_prs_string = cls.get_sug_words_pos(w_sug)
    #         # print(f"WORD_SUG_POS: {words_sug_pos}, TGT_POS: {mask_pos}")
    #         print('.', end='')
    #         if cls.eval_sug_words_by_pos(words_sug_pos, mask_pos):
    #             input_candid = cls.replace_mask_w_suggestion(masked_input, w_sug)
    #             # yield input_candid
    #             if cls.eval_sug_words_by_req(input_candid, requirement, label):
    #                 yield input_candid
    #             # end if
    #         # end if
    #     # end for
    #     return

    @classmethod
    def get_new_inputs(cls, editor, gen_inputs, num_target=10):
        # print(f"\n>>>>> {masked_input}, {mask_pos}")
        for g_i in range(len(gen_inputs)):
            gen_input = gen_inputs[g_i]
            masked_input, mask_pos = gen_input["masked_input"]
            gen_input["words_suggest"] = cls.get_word_suggestion(editor, masked_input, mask_pos)
            gen_inputs[g_i] = gen_input
            # for w_sug in words_suggest:
            #     words_sug_pos, word_sug_prs_string = cls.get_sug_words_pos(w_sug)
            #     # print(f"WORD_SUG_POS: {words_sug_pos}, TGT_POS: {mask_pos}")
            #     print('.', end='')
            # # end for
        # end for
        return gen_inputs

    @classmethod
    def eval_word_suggest(cls, gen_input, label: str, requirement):
        results = list()
        masked_input, mask_pos = gen_input["masked_input"]
        for w_sug in gen_input["words_suggest"]:
            words_sug_pos, word_sug_prs_string = cls.get_sug_words_pos(w_sug)
            
            # check pos
            if cls.eval_sug_words_by_pos(words_sug_pos, mask_pos):
                input_candid = cls.replace_mask_w_suggestion(masked_input, w_sug)
                
                # check requirements
                if cls.eval_sug_words_by_req(input_candid, requirement, label):
                    results.append((masked_input,
                                    gen_input["cfg_from"],
                                    gen_input["cfg_to"],
                                    mask_pos,
                                    w_sug,
                                    input_candid,
                                    label))
                # end if
            # end if
        # end for
        return results


# def main():
#     return

# if __name__=='__main__':
#     main()
