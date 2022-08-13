# This script extracts hateful words
# based on the Hurtlex lexicon dataset
# Link: https://github.com/valeriobasile/hurtlex

# Hurtlex Category Label Description
# PS	negative stereotypes ethnic slurs
# RCI	locations and demonyms
# PA	professions and occupations
# DDF	physical disabilities and diversity
# DDP	cognitive disabilities and diversity
# DMC	moral and behavioral defects
# IS	words related to social and economic disadvantage
# OR	plants
# AN	animals
# ASM	male genitalia
# ASF	female genitalia
# PR	words related to prostitution
# OM	words related to homosexuality
# QAS	with potential negative connotations
# CDS	derogatory words
# RE	felonies and words related to crime and immoral behavior
# SVP	words related to the seven deadly sins of the Christian tradition

from typing import *

import re, os
import sys
import json
import random

from pathlib import Path
from ...utils.Macros import Macros
from ...utils.Utils import Utils

class Hurtlex:

    POS = ['n', 'a', 'v']
    CAT = [
        'ps', 'rci', 'pa', 'ddf', 'ddp',
        'dmc', 'is', 'or', 'an', 'asm',
        'asf', 'pr', 'om', 'qas', 'cds'
        're', 'svp'
    ]

    @classmethod
    def read_raw_data(cls) -> List[Dict]:
        data = list()
        raw_data = Utils.read_sv(
            Macros.hurtlex_data_file,
            delimeter='	',
            is_first_attributes=True
        )
        for d in raw_data['lines']:
            w_id, w_pos, w_cat, w_streo, w_lem, w_lvl = d[0], d[1], d[2], d[3], d[4], d[5]
            data.append({
                raw_data['attributes'][0]: w_id,
                raw_data['attributes'][1]: w_pos,
                raw_data['attributes'][2]: w_cat,
                raw_data['attributes'][3]: w_streo,
                raw_data['attributes'][4]: w_lem,
                raw_data['attributes'][5]: w_lvl
            })
        # end for
        return data

    @classmethod
    def get_target_pos_words(cls, raw_data: List[Dict], target_pos) -> List[Dict]:
        target_pos = target_pos.strip('<>')
        poss = cls.POS # n: noun, a: adj, v: verb
        if target_pos not in poss:
            return raw_data
        # end if
        data = list()
        for d in raw_data:
            if d['pos']==target_pos: # n
                data.append(d)
            # end if
        # end for
        return data

    @classmethod
    def get_target_cat_words(cls, raw_data: List[Dict], target_cat) -> List[Dict]:
        target_cat = target_cat.strip('<>')
        data = list()
        for cat in target_cat.split('_'):
            for d in raw_data:
                if d['category']==cat and d not in data:
                    data.append(d)
                # end if
            # end for
        # end for
        return data
