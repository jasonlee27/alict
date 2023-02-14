
import os
import re
import nltk
import time
import random
import multiprocessing

from tqdm import tqdm
from ..model.Result import Result
from .SelfBleu import read_our_seeds

from .SelfBleu import SelfBleu

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger

random.seed(Macros.RAND_SEED[1])

class Textattack:
    # textattack/bert-base-uncased-SST-2
    # textattack/roberta-base-SST-2
    # distilbert-base-uncased-finetuned-sst-2-english
    MODEL_UNDER_TEST = 'textattack/bert-base-uncased-SST-2'
    TEXTATTACK_DIR = Macros.download_dir / 'textattack'
    MODEL_LOG_MAP = {
        'textattack/bert-base-uncased-SST-2': 'log.txt',
        'textattack/roberta-base-SST-2': '2-log.txt',
        'distilbert-base-uncased-finetuned-sst-2-english': '3-log.txt'
    }
    
    @classmethod
    def parse_results(cls, recipe_name):
        # recipe_name: [alzantot, bert-attack, pso]
        log_ext = cls.MODEL_LOG_MAP[cls.MODEL_UNDER_TEST]
        log_file = cls.TEXTATTACK_DIR / f"{recipe_name}-{log_ext}"
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
    def get_s2lct_exp_sents(cls,
                            adv_example_dict: dict,
                            task: str,
                            search_dataset_name: str,
                            selection_method: str):
        seed_exp_map = dict()
        seed_dir = Macros.result_dir / f"templates_{task}_{search_dataset_name}_{selection_method}"
        seed_files = [
            f for f in os.listdir(str(seed_dir))
            if f.startswith('cfg_expanded_inputs_') and f.endswith('.json')
        ]
        adv_seed_used = list(adv_example_dict.keys())
        for seed_file in seed_files:
            seed_dict = Utils.read_json(seed_dir / seed_file)
            lc = seed_dict['requirement']['description']
            for s in seed_dict['inputs'].keys():
                if s in adv_seed_used:
                    exp_sents_per_seed = [e[5] for e in seed_dict['inputs'][s]['exp_inputs']]
                    exp_sent = random.sample(exp_sents_per_seed, 1) if any(exp_sents_per_seed) else None
                    seed_exp_map[s] = [exp_sent]
                # end if
            # end for
        # end for
        return seed_exp_map

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
            # print(len(res['pass->fail']), sum([len(d['to']) for d in res['pass->fail']]))
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
    print(f"MODEL_UNDER_TEST: {Textattack.MODEL_UNDER_TEST}")
    # recipe_name: [alzantot, bert-attack, pso]
    recipe_name = ['alzantot', 'bert-attack', 'pso']
    s2lct_seed_exp_map = None
    s2lct_adv_result = None
    num_adv_s2lct = -1
    for r_i, r in enumerate(recipe_name):
        adv_example_dict = Textattack.parse_results(r)
        num_adv_ta = sum([
            len(adv_example_dict[s])
            for s in adv_example_dict.keys()
            if adv_example_dict[s] is not None
        ])
        if r_i==0:
            s2lct_seed_exp_map = Textattack.get_s2lct_exp_sents(
                adv_example_dict,
                task,
                search_dataset_name,
                selection_method
            )
            # s2lct_adv_result = Textattack.get_s2lct_exp_fails(
            #     adv_example_dict,
            #     task,
            #     search_dataset_name,
            #     selection_method
            # )
            # num_adv_s2lct = sum([len(s2lct_adv_result[s]) for s in s2lct_adv_result.keys() if s2lct_adv_result[s] is not None])
            
        # end if
        # print(f"RECIPE {r}: {num_adv_ta}, {num_adv_s2lct}")
    # end for
    
    # for s in s2lct_result.keys():
    #     if s2lct_result[s] is not None:
    #         print(s)
    #         for e in s2lct_result[s]:
    #             print('===== ', e)
    #             print()
    #         # end for
    #     # end if
    # # end for
    
    return
