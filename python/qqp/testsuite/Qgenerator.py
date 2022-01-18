# This script perturb and transform inputs in datasets that meet requirements

from typing import *

import re, os
import sys
import json
import spacy
import random
import string
import checklist
import numpy as np

from checklist.editor import Editor
from checklist.expect import Expect
from pathlib import Path

from ..utils.Macros import Macros
from ..utils.Utils import Utils
# from .sentiwordnet.Sentiwordnet import Sentiwordnet
from .Synonyms import Synonyms
from .Suggest import Suggest

random.seed(27)


class Qgenerator:

    def __init__(self, seed, cfg_seed, new_input_results, requirement):
        # new_inputs: List of
        # (masked new input, exp_cfg_from, exp_cfg_to, exp_cfg_pos, exp_word, input with exp_word)
        # if no expansion, then it will be empty
        self.new_inputs = new_input_results
        self.editor = Editor()
        self.seed = seed
        self.cfg_seed = cfg_seed        
        self.transform_action, self.transform_props = requirement['transform'].split()
    
    def generate_question(self):
        new_sent_dict = None
        if self.transform_action=='add':
            new_sent_dict = cls.add()
        elif self.transform_action=='replace':
            new_sent_dict = cls.replace()
        elif self.transform_action=='remove':
            new_sent_dict = cls.remove()
        # end if
        return new_sent

    def _add_adj(self, sent):
        # find noun and sample one
        nlp = spacy.load('en_core_web_md')
        doc = nlp(sent)
        tokens = [str(t) for t in doc]
        nouns = [t_i for t_i, t in enumerate(doc) if t.pos_=='NOUN']
        random.shuffle(nouns)
        
        # insert masked token before the selected noun
        masked_tokens = tokens.insert(nouns[0], Macros.MASK)

        # genereate masked sentence
        masked_sent = Utils.detokenize(masked_tokens)

        # get the suggested word for the masked word
        word_suggestions = Suggest.get_word_suggestion(self.editor, masked_sent, mask_pos=None)
        new_sents = list()
        for w in list(set(word_suggestions)):
            doc = nlp(w)
            if doc[0].pos_=='ADJ':
                new_sent = masked_sent.replace(Macros.MASK, w)
                new_sents.append(new_sent)
            # end if
        # end for
        return new_sents
        
    def add(self):
        # generate seed question pair
        if self.transform_props=='adj':
            results = dict()
            # generating new question by adding word of specified tag of pos in question
            results[self.seed] = self._add_adj(self.seed)
            
            # generating new question by adding word of specified tag of pos in question
            results['exp_inputs'] = dict()
            for s in self.new_inputs:
                results['exp_inputs'][s[5]] = self._add_adj(s[5])
            # end for
            return results
        else:
            # generating new question by adding word in question
            pass
        # end if
        return
    

    def replace(self):
        # generate seed question pair
        
        # generate expanded input question pair
        return new_sents

    def remove(self):
        # generate seed question pair

        # generate expanded input question pair
        return new_sents
