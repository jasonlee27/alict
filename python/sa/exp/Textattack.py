
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
from .ProductionruleCoverage import ProductionruleCoverage

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger

random.seed(Macros.RAND_SEED[1])
NUM_PROCESSES = 1

def get_selfbleu_scores(adv_example_dict, s2lct_seed_exp_map):
    adv_examples = list()
    s2lct_exps = list()
    for s in adv_examples_dict.keys():
        adv_examples.extend(adv_examples_dict[s])
        s2lct_exps.extend(s2lct_seed_exp_map[s])
    # end for
    selfbleu_adv = SelfBleu(texts=adv_examples,
                            num_data=len(adv_examples),
                            logger=logger)
    score_adv = sbleu_seed.get_score_wo_sample()
    sbleu_exp = SelfBleu(texts=s2lct_exps,
                         num_data=len(s2lct_exps),
                         logger=logger)
    score_exp = sbleu_exp.get_score_wo_sample()
    return score_adv, score_exp

def get_cfg_rules_per_sent(sent):
    st = time.time()
    tree_dict = BeneparCFG.get_seed_cfg(sent)
    cfg_rules = tree_dict['rule']
    rules = list()
    for lhs in cfg_rules.keys():
        rhss = cfg_rules[lhs]
        for rhs in rhss:
            rhs_pos = rhs['pos']
            if rhs_pos!=rhs['word'][0]:
                rules.append(f"{lhs}->{rhs_pos}")
            # end if
        # end for
    # end for
    ft = time.time()
    # print(f"{sent}::{round(ft-st,2)}sec")
    return {
        'sent': sent,
        'cfg_rules': list(set(rules))
    }

def get_pdr_scores(adv_example_dict, exp_rules):
    adv_examples = list()
    adv_rules = list()
    for s in adv_examples_dict.keys():
        adv_examples.extend(adv_examples_dict[s])
    # end for

    args = [(s,) for s in adv_examples]
    pool = Pool(processes=NUM_PROCESSES)
    results = pool.starmap_async(get_cfg_rules_per_sent,
                                 args,
                                 chunksize=len(args) // NUM_PROCESSES).get()
    for r in results:
        adv_rules[r['sent']] = r['cfg_rules']
    # end for

    pdr_adv_obj = ProductionruleCoverage(our_cfg_rules=adv_rules)
    score_adv, _ = pdr_adv_obj.get_score()
    pdr_exp_obj = ProductionruleCoverage(our_cfg_rules=exp_rules)
    score_exp, _ = pdr_exp_obj.get_score()
    return score_adv, score_exp


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
        exp_rules = dict()
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
                    cfg_seed = cfg_res['inputs'][s]['cfg_seed']
                    pdr_seed = get_pdr_per_sent(cfg_seed)
                    # exp_sents_per_seed = [e[5] for e in seed_dict['inputs'][s]['exp_inputs']]
                    if any(seed_dict['inputs'][s]['exp_inputs']):
                        pdr_exp = pdr_seed.copy()
                        exp_obj = random.sample(seed_dict['inputs'][s]['exp_inputs'], 1)
                        cfg_from, cfg_to, exp_sent = exp_obj[1], exp_obj[2], exp_obj[5]
                        cfg_from = cfg_from.replace(f" -> ", '->')
                        lhs, rhs = cfg_from.split('->')
                        if len(eval(rhs))==1:
                            cfg_from = f"{lhs}->{eval(rhs)[0]}"
                        # end if
                        cfg_to = cfg_to.replace(f" -> ", '->')
                        pdr_exp.remove(cfg_from)
                        pdr_exp.append(cfg_to)
                        exp_rules[exp_sent] = pdr_exp
                        seed_exp_map[s] = exp_sent
                    # end if
                # end if
            # end for
        # end for
        return seed_exp_map, exp_rules

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
            s2lct_seed_exp_map, s2lct_exp_rules = Textattack.get_s2lct_exp_sents(
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
        
        pdr_cov_scores = get_pdr_scores(adv_example_dict, s2lct_exp_rules)
        
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
