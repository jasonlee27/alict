# this script extracts protected group words
# listed in hatecheck-data repo
# Link: https://github.com/paul-rottger/hatecheck-data

from typing import *

import re, os
import sys
import json
import random

from pathlib import Path
from ...utils.Macros import Macros
from ...utils.Utils import Utils


class Hatecheck:

    hatecheck_placeholder_file = Macros.hatecheck_data_file.parent / 'template_placeholders.csv'
    PROFANITY_WORDS = [
        'fuck', 'fucking', 'fucked', 'motherfucker', 'motherfuckers',
        'shit', 'shitty', 'shithead', 'bullshit', 'pissed off',
        'asshole', 'assholes', 'bastards', 'bastard', 'prick'
        'scumbag', 'scumbags', 'dumbass', 'badass', 'dickhead'   
    ]
    
    @classmethod
    def get_placeholder_values(cls) -> dict:
        raw_data = Utils.read_sv(cls.hatecheck_placeholder_file)
        attr = raw_data['attributes']
        data = raw_data['lines']
        tgt_of_interest = ['IDENTITY_S', 'IDENTITY_P', 'IDENTITY_A']
        res = {
            t: list()
            for t in tgt_of_interest
        }
        for d in data:
            ph_name = d[0].strip('[]')
            if ph_name in tgt_of_interest:
                values = ",".join(d[1:])
                res[ph_name] = values.strip('\"\"').split(',')
            # end if
        # end for
        return res

    @classmethod
    def get_slur_words(cls) -> dict:
        raw_data = Utils.read_sv(cls.hatecheck_placeholder_file)
        attr = raw_data['attributes']
        data = raw_data['lines']
        tgt_of_interest = ['SLUR_S', 'SLUR_P']
        res = {
            t: list()
            for t in tgt_of_interest
        }
        for d in data:
            ph_name = d[0].strip('[]')
            if ph_name in tgt_of_interest:
                values = ",".join(d[1:])
                res[ph_name] = values.strip('\"\"').split(',')
            # end if
        # end for
        return res

    @classmethod
    def get_profanity_words(cls) -> list:
        return cls.PROFANITY_WORDS
