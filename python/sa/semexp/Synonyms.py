# This script extract synonyms given a word

from typing import *

import re, os
import sys
import json
import random

# from nltk.corpus import wordnet
from ..synexp.cfg.CFG import BeneparCFG
from ..utils.Macros import Macros
from ..utils.Utils import Utils

class Synonyms:

    wordnet_tag_map = {
        'n': 'NN',
        's': 'JJ',
        'a': 'JJ',
        'r': 'RB',
        'v': 'VB'
    }
    
    @classmethod
    def get_synsets(cls, nlp, word: str):
        token = nlp(word)[0]
        return token._.wordnet.synsets()

    @classmethod
    def get_word_pos(cls, nlp, word: str):
        doc = nlp(word)
        return str(doc[0]), doc[0].tag_
        # try:
        #     tree = BeneparCFG.get_word_pos(word)
        #     parse_string = tree._.parse_string
        #     pattern = r"\(([^\:|^\(|^\)]+)\s"+word+r"\)"
        #     search = re.search(pattern, parse_string)
        #     if search:
        #         return parse_string, search.group(1)
        #     # end if
        #     return None, None
        # except IndexError:
        #     print(f"IndexError: {word}")
        #     return None, None

    @classmethod
    def get_words_pos(cls, nlp, words):
        results = list()
        for w in words:
            doc = nlp(w)
            results.append((str(doc[0]), doc[0].tag_))
        # end for
        # trees= BeneparCFG.get_words_pos(words)
        # for tr_i, tree in enumerate(trees):
        #     try:
        #         parse_string = tree._.parse_string
        #         pattern = r"\(([^\:|^\(|^\)]+)\s"+str(tree)+r"\)"
        #         search = re.search(pattern, parse_string)
        #         if search:
        #             results.append((parse_string, search.group(1)))
        #         else:
        #             results.append((None, None))
        #         # end if
        #     except IndexError:
        #         print(f"IndexError: {words[tr_i]}")
        #         results.append((None, None))
        #     # end try
        # # end for
        return results
        
    @classmethod
    def get_wn_syn_pos(cls, nlp, synonym):
        doc = nlp(synonym)
        return doc[0].tag_
        
    @classmethod
    def get_synonyms(cls, nlp, word: str, wpos:str, num_synonyms=Macros.max_num_synonyms):
        synonyms = list()
        if wpos is None: return synonyms
        for syn in cls.get_synsets(nlp, word):
            for lm in syn.lemmas():
                synonym = lm.name()
                wn_spos = cls.get_wn_syn_pos(nlp, synonym)
                if wpos==wn_spos and synonym.lower().strip()!=word.lower().strip():
                    synonyms.append(' '.join(synonym.lower().split('_')))
                # end if
            # end for
        # end for
        return list(set(synonyms))[:num_synonyms]

    

# if __name__=="__main__":
#     word = "traditional"
#     wpos = "VB"
#     syns = Synonyms.get_synonyms(word, wpos)
#     print(f"{word}: {syns}")
