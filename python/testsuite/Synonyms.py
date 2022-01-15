# This script extract synonyms given a word

from typing import *

import re, os
import sys
import json
import random

# from nltk.corpus import wordnet
from .cfg.CFG import BeneparCFG
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
    def get_words_pos(cls, words):
        trees= BeneparCFG.get_words_pos(words)
        results = list()
        for tr_i, tree in enumerate(trees):
            try:
                parse_string = tree._.parse_string
                pattern = r"\(([^\:|^\(|^\)]+)\s"+words[tr_i]+r"\)"
                search = re.search(pattern, parse_string)
                if search:
                    results.append((parse_string, search.group(1)))
                else:
                    results.append((None, None))
                # end if
            except IndexError:
                print(f"IndexError: {words[tr_i]}")
                results.append((None, None))
            # end try
        # end for
        return results
        
    @classmethod
    def get_wn_syn_pos(cls, synset):
        return cls.wordnet_tag_map[synset.pos()]
        
    @classmethod
    def get_synonyms(cls, nlp, word: str, wpos:str, num_synonyms=Macros.max_num_synonyms):
        synonyms = list()
        if wpos is None:
            return synonyms
        # end if
        for syn in cls.get_synsets(nlp, word):
            wn_spos = cls.get_wn_syn_pos(syn)
            syns = [lm.name() for lm in syn.lemmas()]
            if wpos.startswith(wn_spos):
                sposs = cls.get_words_pos(syns)
                for sword, spos in zip(syns,sposs):
                    if sword.lower().strip()!=word.lower().strip() and wpos==spos[1]:
                        synonyms.append(sword.lower())
                    # end if
                # end for
            # end if
        # end for
        return list(set(synonyms))[:num_synonyms]

    

# if __name__=="__main__":
#     word = "traditional"
#     wpos = "VB"
#     syns = Synonyms.get_synonyms(word, wpos)
#     print(f"{word}: {syns}")
