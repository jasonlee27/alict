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
from nltk.tokenize import word_tokenize as tokenize

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

    MASK = "{mask}"
    
    @classmethod
    def get_pos_from_mask(cls, masked_input: str):
        mask_pos = list()
        result = list()
        mask_pos = re.findall(r"\{mask\:([^\}|^\:]+)\}", masked_input)
        result = re.sub(r"\{mask\:([^\}|^\:]+)\}", cls.MASK, masked_input)
        return result, mask_pos
    
    @classmethod
    def get_word_suggestion(cls, editor: Editor, masked_input: str, num_target=5):
        word_suggest = editor.suggest(masked_input, remove_duplicates=True)
        suggest_res = list()
        for ws in word_suggest:
            non_letters = [re.search(r"[^A-Za-z0-9]+", w) for w in ws]
            if not any(non_letters):
                suggest_res.append(ws)
            # end if
        # end for
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
        # end if
        for t_is, w in zip(masked_tok_is, words_suggest):
            search = re.search(r'\{mask\}', _masked_input)
            if search:
                _masked_input = _masked_input[:search.start()] + w + _masked_input[search.end():]
            # end if
        # end for
        return _masked_input

    @classmethod
    def get_word_pos(cls, word):
        tree = BeneparCFG.get_word_pos(word)
        parse_string = tree._.parse_string
        pattern = r"\(([^\:|^\(|^\)]+)\s"+word+r"\)"
        search = re.search(pattern, parse_string)
        if search:
            return search.group(1)
        # end if
        return

    @classmethod
    def eval_sug_words_by_pos(cls, word_suggest, mask_pos):
        match = list()
        if any([w for w in word_suggest if w=='']):
            return False
        # end if
        if type(word_suggest)==str:
            word_suggest = [word_suggest]
        # end if
        for w, pos in zip(word_suggest, mask_pos):
            w_pos = cls.get_word_pos(w)
            generic_pos = pos.split("-")[0]
            if w_pos==generic_pos:
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
    def eval_sug_words_by_req(cls, new_input, requirement):
        search_obj = SearchOperator(requirement)
        search_res = search_obj.search([new_input])
        if len(search_res)>0:
            return True
        # end if
        return False

    @classmethod
    def get_new_input(cls, editor, masked_input: str, requirement):
        _masked_input, mask_pos = cls.get_pos_from_mask(masked_input)
        words_suggest = cls.get_word_suggestion(editor, _masked_input)
        for w_sug in words_suggest:
            input_candid = cls.replace_mask_w_suggestion(_masked_input, w_sug)
            print(_masked_input)
            print(mask_pos)
            print(f"INPUT: {input_candid}")
            if cls.eval_sug_words_by_pos(w_sug, mask_pos):
                if cls.eval_sug_words_by_req(input_candid, requirement):
                    yield input_candid
                # end if
            # end if
            print()
        # end for
        return


# def main():
#     return

# if __name__=='__main__':
#     main()
