
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
    MODEL_UNDER_TEST = 'textattack/bert-base-uncased-SST-2'
    
    @classmethod
    def parse_results(cls, recipe_name):
        # recipe_name: [alzantot, bert-attack, pso]
        log_file = cls.TEXTATTACK_DIR / f"{recipe_name}-log.txt"
        lines = Utils.read_txt(log_file)
        l_i = 0
        result = dict()
        while(l_i<len(lines)):
            if re.search(r'----- Result (\d+) -----', lines[l_i]):
                is_failed = re.search(r'\[\[\[SKIPPED\]\]\]|\[\[\[FAILED\]\]\]', lines[l_i+1])
                orig_sent = re.sub(
                    r'\[\[([a-zA-Z_][a-zA-Z_0-9]*)\]\]',
                    r'\1',
                    lines[l_i+3].strip()
                )
                if not is_failed:
                    adv_sent = re.sub(
                        r'\[\[([a-zA-Z_][a-zA-Z_0-9]*)\]\]',
                        r'\1',
                        lines[l_i+5].strip()
                    )
                    l_i += 5
                else:
                    adv_sent = None
                    l_i += 3
                # end if
                result[orig_sent] = adv_sent if adv_sent is None else [adv_sent]
            elif lines[l_i].startswith('Number of successful attacks:'):
                break
            # end if
            l_i += 1
        # end while
        return result

    @classmethod
    def get_s2lct_exp_fails(cls,
                            adv_example_dict: dict,
                            task: str,
                            search_dataset_name: str,
                            selection_method: str):
        test_results_dir = Macros.result_dir / f"test_results_{task}_{search_dataset_name}_{selection_method}"
        test_results_file = test_results_dir / 'test_result_analysis.json'
        result_dict = Utils.read_json(test_results_file)
        result_dict_model = result_dict[cls.MODEL_UNDER_TEST]
        ptf_seeds_all = dict()
        for lc_i, res in enumerate(result_dict_model):
            lc = res['req']
            # ptfs = res['pass->fail']
            ptf_seeds_all[lc_i] = [
                r['from']['sent']
                for r in res['pass->fail']
            ]
            print(len(res['pass->fail']), sum([len(d['to']) for d in res['pass->fail']]))
        # end for
        
        seeds_used_for_adv = list(adv_example_dict.keys())
        result = dict()
        for s in seeds_used_for_adv:
            tokens = Utils.tokenize(s)
            _s = Utils.detokenize(tokens)
            result[s] = None
            for lc_i in ptf_seeds_all.keys():
                if s in ptf_seeds_all[lc_i]:
                    s_i = ptf_seeds_all[lc_i].index(s)
                    result[s] = [
                        t['sent']
                        for t in result_dict_model[lc_i]['pass->fail'][s_i]['to']
                    ]
                elif _s in ptf_seeds_all[lc_i]:
                    s_i = ptf_seeds_all[lc_i].index(_s)
                    result[s] = [
                        t['sent']
                        for t in result_dict_model[lc_i]['pass->fail'][s_i]['to']
                    ]
                # end if
            # end for
        # end for
        return result


def main(task: str,
         search_dataset_name: str,
         selection_method: str):
    # recipe_name: [alzantot, bert-attack, pso]
    recipe_name = ['alzantot', 'bert-attack', 'pso']
    for r in recipe_name:
        adv_example_dict = Textattack.parse_results(r)
        num_adv_ta = sum([len(adv_example_dict[s]) for s in adv_example_dict.keys() if adv_example_dict[s] is not None])
        s2lct_result = Textattack.get_s2lct_exp_fails(
            adv_example_dict,
            task,
            search_dataset_name,
            selection_method
        )
        num_adv_s2lct = sum([len(s2lct_result[s]) for s in s2lct_result.keys() if s2lct_result[s] is not None])
        print(num_adv_ta, num_adv_s2lct)
    # end 
    return
