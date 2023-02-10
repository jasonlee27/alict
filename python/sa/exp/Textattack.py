
import os
import re
import nltk
import time
import random
import multiprocessing

from tqdm import tqdm
from ..model.Result import Result
from .SelfBleu import read_our_seeds

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger


class Textattack:

    TEXTATTACK_DIR = Macros.download_dir / 'textattack'
    
    @classmethod
    def parse_results(cls, recipe_name):
        # recipe_name: [alzantot, bert-attack, pso]
        log_file = cls.TEXTATTACK_DIR / f"{recipe_name}-log.txt"
        lines = Utils.read_txt(log_file)
        l_i = 0
        result = dict()
        while(l_i<len(lines))
            if re.search(r'----- Result (\d+) -----', lines[l_i]):
                if_failed = re.search(r'\[\[\[SKIPPED\]\]\]|\[\[\[FAILED\]\]\]', lines[l_i+1])
                orig_sent = re.sub(
                    r'\[\[([a-zA-Z_][a-zA-Z_0-9]*)\]\]',
                    r'\1',
                    lines[l_i+3]
                )
                if not is_failed:
                    adv_sent = re.sub(
                        r'\[\[([a-zA-Z_][a-zA-Z_0-9]*)\]\]',
                        r'\1',
                        lines[l_i+5]
                    )
                    l_i += 5
                else:
                    adv_sent = None
                    l_i += 3
                # end if
                result[orig_sent] = adv_sent
            elif lines[l_i].startswith('Number of successful attacks:'):
                break
            # end if
            l_i += 1
        # end while
        return result

    @classmethod
    def get_s2lct_exp_fails(cls, adv_example_dict: Dict):
        texts_lcs = dict()
        seed_exp_map = dict()
        seed_dir = Macros.result_dir / f"templates_{task}_{search_dataset_name}_{selection_method}"
        seed_files = [
            f for f in os.listdir(str(seed_dir))
            if f.startswith('cfg_expanded_inputs_') and f.endswith('.json')
        ]
        for seed_file in seed_files:
            seed_dict = Utils.read_json(seed_dir / seed_file)
            lc = seed_dict['requirement']['description']
            if lc not in seed_exp_map.keys():
                seed_exp_map[lc] = dict()
            # end if
            exp_sents = list()
            for s in seed_dict['inputs'].keys():
                exp_sents_per_seed = [e[5] for e in seed_dict['inputs'][s]['exp_inputs']]
                if s not in seed_exp_map[lc].keys():
                    seed_exp_map[lc][s] = exp_sents_per_seed
                # end if
                exp_sents.extend(exp_sents_per_seed)
            # end for
            texts_lcs[lc] = exp_sents
        # end for
        return seed_exp_map, texts_lcs

            
