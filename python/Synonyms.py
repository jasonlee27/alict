# This script extract synonyms given a word

from typing import *

import re, os
import sys
import json
import random

from nltk.corpus import wordnet
from CFG import BeneparCFG


class Synonyms:

    wordnet_tag_map = {
        'n': 'NN',
        's': 'JJ',
        'a': 'JJ',
        'r': 'RB',
        'v': 'VB'
    }
    
    @classmethod
    def get_synsets(cls, word: str):
        return wordnet.synsets(word)

    @classmethod
    def get_word_pos(cls, word: str):
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
    def get_wn_syn_pos(cls, synset):
        return cls.wordnet_tag_map[synset.pos()]
        
    @classmethod
    def get_synonyms(cls, word: str, wpos:str):
        synonyms = list()
        for syn in cls.get_synsets(word):
            # spos = cls.get_wn_syn_pos(syn)
            for lm in syn.lemmas():
                _, spos = cls.get_word_pos(lm.name())
                if lm.name()!=word and wpos==spos:
                    synonyms.append(lm.name())
                # end if
            # end for
        # end for
        return list(set(synonyms))

    

# if __name__=="__main__":
#     word = "enjoy"
#     wpos = "VB"
#     syns = Synonyms.get_synonyms(word, wpos)
#     print(f"{word}: {syns}")
