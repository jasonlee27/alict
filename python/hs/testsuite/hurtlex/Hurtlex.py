# This script extracts hateful words
# based on the Hurtlex lexicon dataset
# Link: https://github.com/valeriobasile/hurtlex

from typing import *

import re, os
import sys
import json
import random

from pathlib import Path
from ...utils.Macros import Macros
from ...utils.Utils import Utils

class Hurtlex:

    @classmethod
    def read_raw_data(cls) -> Dict:
        data = dict()
        raw_data = Utils.read_sv(
            Macros.hurtlex_data_file,
            delimeter='	',
            is_first_attributes=True
        )
        for d in raw_data['lines']:
            w_id, w_pos, w_cat, w_streo, w_lem, w_lvl = d[0], d[1], d[2], d[3], d[4], d[5]
            data[w_id] = {
                raw_data['attributes'][1]: w_pos,
                raw_data['attributes'][2]: w_cat,
                raw_data['attributes'][3]: w_streo,
                raw_data['attributes'][4]: w_lem,
                raw_data['attributes'][5]: w_lvl
            }
        # end for
        return data

    @classmethod
    def split_by_pos(cls, raw_data: Dict) -> Dict:
        poss = ['n', 'a', 'v'] # n: noun, a: adj, v: verb
        data = {
            'n': list(),
            'a': list(),
            'v': list()
        }
        for d in raw_data.keys():
            if d['pos']==poss[0]: # n
                data[poss[0]].append({
                    
                })
        
        
        
        


# if __name__=="__main__":
#     Sentiwordnet.get_sent_words()
