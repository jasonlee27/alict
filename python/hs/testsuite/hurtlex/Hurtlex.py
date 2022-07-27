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
    def get_target_pos_words(cls, raw_data: Dict, target_pos) -> Dict:
        # poss = ['n', 'a', 'v'] # n: noun, a: adj, v: verb
        data = list()
        for d in raw_data.keys():
            if raw_data[d]['pos']==target_pos: # n
                data.append({
                    'id': d,
                    'cat': raw_data[d]['category'],
                    'stereotype': raw_data[d]['stereotype'],
                    'lemma': raw_data[d]['lemma'],
                    'level': raw_data[d]['level']
                })
            # end if
        # end for
        return data

    @classmethod
    def get_target_cat_words(cls, raw_data: Dict, target_cat) -> Dict:
        poss = ['n', 'a', 'v'] # n: noun, a: adj, v: verb
        data = list()
        for d in raw_data.keys():
            if raw_data[d]['category']==target_cat: # n
                data.append({
                    'id': d,
                    'pos': raw_data[d]['pos'],
                    'stereotype': raw_data[d]['stereotype'],
                    'lemma': raw_data[d]['lemma'],
                    'level': raw_data[d]['level']
                })
            # end if
        # end for
        return data
