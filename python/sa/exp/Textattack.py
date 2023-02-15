
import os
import re
import nltk
import time
import random
import multiprocessing


from tqdm import tqdm
from multiprocessing import Pool
from ..model.Result import Result

from ..synexp.cfg.CFG import BeneparCFG
from .SelfBleu import SelfBleu
from .ProductionruleCoverage import ProductionruleCoverage

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger

random.seed(Macros.RAND_SEED[1])
NUM_PROCESSES = 1
NUM_TRIALS = 5

def get_selfbleu_scores(adv_example_dict, s2lct_seed_exp_map, trial_i):
    adv_examples = list()
    s2lct_exps = list()
    for s in adv_example_dict.keys():
        if adv_example_dict[s] is not None:
            adv_examples.extend(adv_example_dict[s])
        # end if
        if any(s2lct_seed_exp_map[trial_i].get(s, list())):
            s2lct_exps.extend(s2lct_seed_exp_map[trial_i][s])
        # end if
    # end for
    sbleu_adv = SelfBleu(texts=adv_examples,
                         num_data=len(adv_examples))
    score_adv = sbleu_adv.get_score_wo_sample()
    sbleu_exp = SelfBleu(texts=s2lct_exps,
                         num_data=len(s2lct_exps))
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

def get_pdr_per_sent(cfg_seed):
    rules = list()
    for lhs in cfg_seed.keys():
        rhss = cfg_seed[lhs]
        for rhs in rhss:
            rhs_pos = rhs['pos']
            if type(rhs_pos)==str:
                if rhs_pos!=rhs['word'][0]:
                    rules.append(f"{lhs}->{rhs_pos}")
                # end if
            else:
                rules.append(f"{lhs}->{rhs_pos}")
            # end if
        # end for
    # end for
    return list(set(rules))

def get_pdr_scores(adv_example_dict, exp_rules, trial_i, adv_rules=dict()):
    if not any(adv_rules):
        adv_examples = list()
        adv_rules = dict()
        for s in adv_example_dict.keys():
            if adv_example_dict[s] is not None:
                adv_examples.extend(adv_example_dict[s])
            # end if
        # end for
        
        args = [(s,) for s in adv_examples]
        pool = Pool(processes=NUM_PROCESSES)
        results = pool.starmap_async(get_cfg_rules_per_sent,
                                     args,
                                     chunksize=len(args) // NUM_PROCESSES).get()
        for r in results:
            adv_rules[r['sent']] = r['cfg_rules']
        # end for
    # end if
    
    pdr_adv_obj = ProductionruleCoverage(our_cfg_rules=adv_rules)
    score_adv, _ = pdr_adv_obj.get_score()
    pdr_exp_obj = ProductionruleCoverage(our_cfg_rules=exp_rules[trial_i])
    score_exp, _ = pdr_exp_obj.get_score()
    return score_adv, score_exp, adv_rules


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
        seed_exp_map = [dict() for t in range(NUM_TRIALS)]
        exp_rules = [dict() for t in range(NUM_TRIALS)]
        seed_dir = Macros.result_dir / f"templates_{task}_{search_dataset_name}_{selection_method}"
        seed_files = [
            f for f in os.listdir(str(seed_dir))
            if f.startswith('cfg_expanded_inputs_') and f.endswith('.json')
        ]
        adv_seed_used = list(adv_example_dict.keys())
        num_exps_over_samples = 0
        for seed_file in seed_files:
            cfg_res = Utils.read_json(seed_dir / seed_file)
            lc = cfg_res['requirement']['description']
            for s in cfg_res['inputs'].keys():
                if s in adv_seed_used:
                    cfg_seed = cfg_res['inputs'][s]['cfg_seed']
                    pdr_seed = get_pdr_per_sent(cfg_seed)
                    # exp_sents_per_seed = [e[5] for e in cfg_res['inputs'][s]['exp_inputs']]
                    num_exps_over_samples += len(cfg_res['inputs'][s]['exp_inputs'])
                    if any(cfg_res['inputs'][s]['exp_inputs']):
                        for t in range(NUM_TRIALS):
                            random.seed(t)
                            exp_obj = random.sample(cfg_res['inputs'][s]['exp_inputs'], 1)[0]
                            pdr_exp = pdr_seed.copy()
                            cfg_from, cfg_to, exp_sent = exp_obj[1], exp_obj[2], exp_obj[5]
                            if exp_sent not in exp_rules[t].keys():
                                cfg_from = cfg_from.replace(f" -> ", '->')
                                lhs, rhs = cfg_from.split('->')
                                if len(eval(rhs))==1:
                                    cfg_from = f"{lhs}->{eval(rhs)[0]}"
                                # end if
                                cfg_to = cfg_to.replace(f" -> ", '->')
                                pdr_exp.remove(cfg_from)
                                pdr_exp.append(cfg_to)
                                exp_rules[t][exp_sent] = pdr_exp
                                seed_exp_map[t][s] = [exp_sent]
                            # end if
                        # end for
                    # end if
                # end if
            # end for
        # end for
        return seed_exp_map, exp_rules, num_exps_over_samples

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
    res_dir = Macros.result_dir / 'textattack' / f"{task}_{search_dataset_name}_{selection_method}"
    # recipe_name: [alzantot, bert-attack, pso]
    recipe_name = ['alzantot', 'bert-attack', 'pso']
    s2lct_seed_exp_map = None
    s2lct_adv_result = None
    num_adv_s2lct = -1
    scores = {
        r: {
            'num_data': None,
            'num_samples': list(),
            'selfbleu_scores': list(),
            'pdrcov_scores': list()
        }
        for r in recipe_name+['ours_exp']
    }
    scores['model_under_test'] = Textattack.MODEL_UNDER_TEST
    for r_i, r in enumerate(recipe_name):
        adv_example_dict = Textattack.parse_results(r)
        num_adv_ta = sum([
            len(adv_example_dict[s])
            for s in adv_example_dict.keys()
            if adv_example_dict[s] is not None
        ])
        if r_i==0:
            s2lct_seed_exp_map, s2lct_exp_rules, num_all_exps = Textattack.get_s2lct_exp_sents(
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
            # num_adv_s2lct = sum([
            #     len(s2lct_adv_result[s])
            #     for s in s2lct_adv_result.keys()
            #     if s2lct_adv_result[s] is not None
            # ])
            
        # end if
        adv_rules = list()
        
        scores[r]['num_data'] = num_adv_ta
        scores['ours_exp']['num_data'] = num_all_exps
        
        for t in range(NUM_TRIALS):
            num_exp_s2lct = sum([
                len(s2lct_seed_exp_map[t][s])
                for s in s2lct_seed_exp_map[t].keys()
                if s2lct_seed_exp_map[t][s] is not None
            ])
            scores[r]['num_samples'].append(num_adv_ta)
            selfbleu_adv, selfbleu_exp = get_selfbleu_scores(adv_example_dict, s2lct_seed_exp_map, t)
            pdr_cov_adv, pdr_cov_exp, adv_rules = get_pdr_scores(adv_example_dict, s2lct_exp_rules, t, adv_rules=adv_rules)
            scores[r]['selfbleu_scores'].append(selfbleu_adv)
            scores[r]['pdrcov_scores'].append(pdr_cov_adv)
            if r_i==0:
                scores['ours_exp']['num_samples'].append(num_exp_s2lct)
                scores['ours_exp']['selfbleu_scores'].append(selfbleu_exp)
                scores['ours_exp']['pdrcov_scores'].append(pdr_cov_exp)
            # end if
            print(f"TRIAL: {t+1}")
            print(f"{r}:: NUM_ADV: {num_adv_ta}")
            print(f"NUM_EXPS: {num_all_exps}")
            print(f"NUM_EXP_SAMPLES: {num_exp_s2lct}")
            print(f"SELF-BLEU: ADV: {selfbleu_adv}, OURS: {selfbleu_exp}")
            print(f"PDR-COV: ADV: {pdr_cov_adv}, OURS: {pdr_cov_exp}\n")
        # end for
        Utils.write_json(scores, res_dir / f"diversity_scores.json", pretty_format=True)
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
