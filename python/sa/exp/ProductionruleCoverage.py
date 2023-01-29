import re
import os
import math
import nltk
import time
import random
import multiprocessing

from tqdm import tqdm
from multiprocessing import Pool
# from functools import partial
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.preprocessing import normalize

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger
from ..synexp.cfg.CFG import BeneparCFG
from ..seed.Search import ChecklistTestsuite

import torch.multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

NUM_PROCESSES = 2 # Macros.num_processes

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
    return  list(set(rules))


class ProductionruleCoverage:
    
    def __init__(self,
                 lc: str,
                 our_cfg_rules: dict=None,
                 bl_cfg_rules: dict=None):
        # the json file used for retraining sa models
        self.lc = lc
        self.our_cfg_rules = our_cfg_rules
        self.bl_cfg_rules = bl_cfg_rules
        self.our_num_data = len(self.our_cfg_rules.keys())
        self.bl_num_data = len(self.bl_cfg_rules.keys()) if bl_cfg_rules is not None else None
        self.scores = list()

    def get_score(self):
        our_rule_set = list()
        for r in self.our_cfg_rules.values():
            our_rule_set.extend(r)
        # end for
        our_rule_set = list(set(our_rule_set))

        bl_rule_set = list()
        if self.bl_cfg_rules is not None:
            for r in self.bl_cfg_rules.values():
                bl_rule_set.extend(r)
            # end for
            bl_rule_set = list(set(bl_rule_set))
        # end if
        return len(our_rule_set), len(bl_rule_set)
    
    @classmethod
    def get_our_seed_cfg_rules(cls,
                               task,
                               search_dataset_name,
                               selection_method,
                               parse_all_sents=False,
                               logger=None):
        seed_rules_over_lcs = dict()
        seed_dir = Macros.result_dir / f"templates_{task}_{search_dataset_name}_{selection_method}"
        req_dir = Macros.result_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            lc_desc = l.strip().split('::')[0]
            lc_cksum = Utils.get_cksum(lc_desc)
            _lc_cksum = Utils.get_cksum(lc_desc.lower())
            seed_file = seed_dir / f"cfg_expanded_inputs_{lc_cksum}.json"
            res_file = Macros.pdr_cov_result_dir / f"seed_{task}_{search_dataset_name}_{selection_method}_pdr_{lc_cksum}.json"
            if os.path.exists(seed_dir / f"cfg_expanded_inputs_{_lc_cksum}.json") and \
               not os.path.exists(seed_dir / f"cfg_expanded_inputs_{lc_cksum}.json"):
                seed_file = seed_dir / f"cfg_expanded_inputs_{_lc_cksum}.json"
                res_file = Macros.pdr_cov_result_dir / f"seed_{task}_{search_dataset_name}_{selection_method}_pdr_{_lc_cksum}.json"
                lc_desc = lc_desc.lower()
            # end if

            seed_rules = dict()
            if os.path.exists(res_file):
                try:
                    seed_rules = Utils.read_json(res_file)
                    seed_rules_over_lcs[lc_desc] = seed_rules
                except:
                    seed_rules = dict()
                # end try
            # end if
            if not any(seed_rules):
                print(f"OUR_SEED::{lc_desc}")
                cfg_res = Utils.read_json(seed_dir / seed_file)
                if not parse_all_sents:
                    if logger is not None:
                        logger.print(f"OUR_SEED::{lc_desc}")
                    # end if

                    for seed in cfg_res['inputs'].keys():
                        if seed not in seed_rules.keys():
                            cfg_seed = cfg_res['inputs'][seed]['cfg_seed']
                            seed_rules[seed] = get_pdr_per_sent(cfg_seed)
                        # end if
                    # end for
                else:
                    if logger is not None:
                        logger.print(f"OUR_SEED_FOR_PDR::{lc_desc}::")
                    # end if
                    args = [(s[1],) for s in seed['seeds'] if s[1] not in seed_rules.keys()]
                    if any(args):
                        pool = Pool(processes=NUM_PROCESSES)
                        results = pool.starmap_async(get_cfg_rules_per_sent,
                                                     args,
                                                     chunksize=len(args) // NUM_PROCESSES).get()
                        for r in results:
                            seed_rules[r['sent']] = r['cfg_rules']
                        # end for

                    # end if
                # end if
                Utils.write_json(seed_rules, res_file, pretty_format=True)
                seed_rules_over_lcs[lc_desc] = seed_rules
            # end if
        # end for
        return seed_rules_over_lcs

    @classmethod
    def get_our_exp_cfg_rules(cls,
                              task,
                              search_dataset_name,
                              selection_method,
                              logger=None):
        exp_rules_over_lcs = dict()
        seed_dir = Macros.result_dir / f"templates_{task}_{search_dataset_name}_{selection_method}"
        req_dir = Macros.result_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            lc_desc = l.strip().split('::')[0]
            lc_cksum = Utils.get_cksum(lc_desc)
            _lc_cksum = Utils.get_cksum(lc_desc.lower())
            seed_file = seed_dir / f"cfg_expanded_inputs_{lc_cksum}.json"
            res_file = Macros.pdr_cov_result_dir / f"exp_{task}_{search_dataset_name}_{selection_method}_pdr_{lc_cksum}.json"
            if os.path.exists(seed_dir / f"cfg_expanded_inputs_{_lc_cksum}.json") and \
               not os.path.exists(seed_dir / f"cfg_expanded_inputs_{lc_cksum}.json"):
                seed_file = seed_dir / f"cfg_expanded_inputs_{_lc_cksum}.json"
                res_file = Macros.pdr_cov_result_dir / f"exp_{task}_{search_dataset_name}_{selection_method}_pdr_{_lc_cksum}.json"
                lc_desc = lc_desc.lower()
            # end if
            
            exp_rules = dict()
            if os.path.exists(res_file):
                try:
                    exp_rules = Utils.read_json(res_file)
                    exp_rules_over_lcs[lc_desc] = exp_rules
                except:
                    exp_rules = dict()
                # end try
            # end if
            if not any(exp_rules):
                cfg_res = Utils.read_json(seed_file)
                print(f"OUR_EXP_FOR_PDR::{lc_desc}")
                if logger is not None:
                    logger.print(f"OUR_EXP_FOR_PDR::{lc_desc}")
                # end if
                for seed in cfg_res['inputs'].keys():
                    exp_rules[seed] = list()
                    cfg_seed = cfg_res['inputs'][seed]['cfg_seed']
                    pdr_seed = get_pdr_per_sent(cfg_seed)
                    for exp in cfg_res['inputs'][seed]['exp_inputs']:
                        pdr_exp = pdr_seed.copy()
                        cfg_from, cfg_to, exp_sent = exp[1], exp[2], exp[5]
                        if exp_sent not in exp_rules.keys():
                            cfg_from = cfg_from.replace(f" -> ", '->')
                            lhs, rhs = cfg_from.split('->')
                            if len(eval(rhs))==1:
                                cfg_from = f"{lhs}->{eval(rhs)[0]}"
                            # end if
                            cfg_to = cfg_to.replace(f" -> ", '->')
                            pdr_exp.remove(cfg_from)
                            pdr_exp.append(cfg_to)
                            exp_rules[exp_sent] = pdr_exp
                        # end if
                    # end for
                # end for
                Utils.write_json(exp_rules, res_file, pretty_format=True)
                exp_rules_over_lcs[lc_desc] = exp_rules
            # end if
        # end for
        return exp_rules_over_lcs
    
    @classmethod
    def get_bl_cfg_rules(cls,
                         task,
                         search_dataset_name,
                         selection_method,
                         logger=None):
        bl_rules_over_lcs = dict()
        res_file = Macros.pdr_cov_result_dir / f"bl_cfg_rules_{task}_checklist.json"
        seed_dir = Macros.result_dir / f"templates_{task}_{search_dataset_name}_{selection_method}"
        seed_files = [
            f for f in os.listdir(str(seed_dir))
            if f.startswith('cfg_expanded_inputs_') and f.endswith('.json')
        ]
        if os.path.exists(res_file):
            bl_rules = Utils.read_json(res_file)
        else:
            bl_rules = dict()
        # end if
        for seed_file in seed_files:
            cfg_res = Utils.read_json(seed_dir / seed_file)
            lc_desc = cfg_res['requirement']['description']
            lc_cksum = Utils.get_cksum(lc_desc)
            if lc_desc not in bl_rules.keys():
                bl_rules[lc_desc] = dict()
            # end if
                
            if logger is not None:
                logger.print(f"BL_FOR_PDR::{lc_desc}")
            # end if
            sents = ChecklistTestsuite.get_sents(
                Macros.checklist_sa_dataset_file,
                lc_desc
            )
            args = [s[1] for s in sents if s[1] not in bl_rules[lc_desc].keys()]
            if any(args):
                pool = Pool(processes=NUM_PROCESSES)
                results = pool.map_async(get_cfg_rules_per_sent,
                                         args,
                                         chunksize=len(args) // NUM_PROCESSES).get()
                for r in results:
                    bl_rules[lc_desc][r['sent']] = r['cfg_rules']
                # end for
            # end if
            Utils.write_json(bl_rules, res_file, pretty_format=True)
        # end for
        return bl_rules

    @classmethod
    def get_mt_cfg_rules(cls,
                         task,
                         search_dataset_name,
                         selection_method,
                         baseline_name,
                         mt_sents,
                         logger=None):
        mt_rules = dict()
        res_file = Macros.pdr_cov_result_dir / f"mt_cfg_rules_{task}_{baseline_name}.json"
        mt_rules_prev = dict()
        if os.path.exists(str(res_file)):
            mt_rules_prev = Utils.read_json(res_file)
        # end if
        args = list()
        for s in mt_sents:
            if s in mt_rules_prev.keys():
                mt_rules[s] = mt_rules_prev[s]
            else:
                args.append(s)
            # end if
        # end for
        pool = Pool(processes=NUM_PROCESSES)
        results = pool.map_async(get_cfg_rules_per_sent,
                                 args,
                                 chunksize=len(args) // NUM_PROCESSES).get()
        for r in results:
            mt_rules[r['sent']] = r['cfg_rules']
        # end for
        Utils.write_json(bl_rules, res_file, pretty_format=True)
        return mt_rules
    
    

def main_sample(task,
                search_dataset_name,
                selection_method):
    st = time.time()
    num_trials = 10
    logger_file = Macros.log_dir / f"seed_{task}_{search_dataset_name}_{selection_method}_pdrcov.log"
    result_file = Macros.pdr_cov_result_dir / f"seed_exp_bl_sample_{task}_{search_dataset_name}_{selection_method}_pdrcov.json"
    Macros.pdr_cov_result_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(logger_file=logger_file,
                    logger_name='seed_exp_bl_sample_pdrcov_log')
    seed_rules = ProductionruleCoverage.get_our_seed_cfg_rules(
        task,
        search_dataset_name,
        selection_method,
        parse_all_sents=False,
        logger=logger
    )
    exp_rules = ProductionruleCoverage.get_our_exp_cfg_rules(
        task,
        search_dataset_name,
        selection_method,
        logger=logger
    )
    checklist_rules = ProductionruleCoverage.get_bl_cfg_rules(
        task,
        search_dataset_name,
        selection_method,
        logger=logger
    )
    
    scores = dict()
    for lc in tqdm(seed_rules.keys()):
        if lc not in scores.keys():
            len_seed_exp = len(list(seed_rules[lc].keys())+list(exp_rules[lc].keys()))
            max_num_samples = int(1e5) # min(int(100*math.ceil(len_seed_exp/100.)), int(2e4))
            num_samples = list(range(500, max_num_samples+500, 500))
            logger.print(f"OURS_PDR_SAMPLE::{lc}")
            our_sents, bl_sents = list(), list()
            scores[lc] = {
                'ours_seed': {
                    f"{num_sample}sample": {
                        'coverage_scores': list()
                    }
                    for num_sample in num_samples
                },
                'ours_seed_exp': {
                    f"{num_sample}sample": {
                        'coverage_scores': list()
                    }
                    for num_sample in num_samples 
                },
                'bl': {
                    f"{num_sample}sample": {
                        'coverage_scores': list()
                    }
                    for num_sample in num_samples
                }
            }
            all_seed_exp_sents = list(seed_rules[lc].keys())+list(exp_rules[lc].keys())
            for num_sample in num_samples:
                for num_trial in range(num_trials):
                    random.seed(num_trial)
                    seed_sents = random.sample(list(seed_rules[lc].keys()),
                                               min(len(seed_rules[lc]), num_sample))
                    seed_exp_sents = random.sample(all_seed_exp_sents,
                                                   min(len(all_seed_exp_sents), num_sample))
                    bl_sents = random.sample(list(checklist_rules[lc].keys()),
                                             min(len(checklist_rules[lc]), num_sample))
                    pdr1 = {
                        s: seed_rules[lc][s]
                        for s in seed_sents
                    }
                    pdr2 = {
                        s: seed_rules[lc][s] 
                        for s in seed_exp_sents
                        if s in seed_rules[lc].keys()
                    }
                    for e in seed_exp_sents:
                        if e not in pdr2.keys():
                            pdr2[e] = exp_rules[lc][e]
                        # end if
                    # end for
                    pdr3 = {
                        s: checklist_rules[lc][s]
                        for s in bl_sents
                    }
                    pdr_obj1 = ProductionruleCoverage(lc=lc,
                                                      our_cfg_rules=pdr1)
                    pdr_obj2 = ProductionruleCoverage(lc=lc,
                                                      our_cfg_rules=pdr2)
                    pdr_obj3 = ProductionruleCoverage(lc=lc,
                                                      our_cfg_rules=pdr3)
                    cov_score_seed, _ = pdr_obj1.get_score()
                    cov_score_seed_exp, _ = pdr_obj2.get_score()
                    cov_score_bl, _ = pdr_obj3.get_score()
                    scores[lc]['ours_seed'][f"{num_sample}sample"]['coverage_scores'].append(cov_score_seed)
                    scores[lc]['ours_seed_exp'][f"{num_sample}sample"]['coverage_scores'].append(cov_score_seed_exp)
                    scores[lc]['bl'][f"{num_sample}sample"]['coverage_scores'].append(cov_score_bl)
                # end for
                scores[lc]['ours_seed'][f"{num_sample}sample"]['avg_score'] = Utils.avg(scores[lc]['ours_seed'][f"{num_sample}sample"]['coverage_scores'])
                scores[lc]['ours_seed'][f"{num_sample}sample"]['med_score'] = Utils.median(scores[lc]['ours_seed'][f"{num_sample}sample"]['coverage_scores'])
                scores[lc]['ours_seed'][f"{num_sample}sample"]['std_score'] = Utils.stdev(scores[lc]['ours_seed'][f"{num_sample}sample"]['coverage_scores'])
                scores[lc]['ours_seed_exp'][f"{num_sample}sample"]['avg_score'] = Utils.avg(scores[lc]['ours_seed_exp'][f"{num_sample}sample"]['coverage_scores'])
                scores[lc]['ours_seed_exp'][f"{num_sample}sample"]['med_score'] = Utils.median(scores[lc]['ours_seed_exp'][f"{num_sample}sample"]['coverage_scores'])
                scores[lc]['ours_seed_exp'][f"{num_sample}sample"]['std_score'] = Utils.stdev(scores[lc]['ours_seed_exp'][f"{num_sample}sample"]['coverage_scores'])
                scores[lc]['bl'][f"{num_sample}sample"]['avg_score'] = Utils.avg(scores[lc]['bl'][f"{num_sample}sample"]['coverage_scores'])
                scores[lc]['bl'][f"{num_sample}sample"]['med_score'] = Utils.median(scores[lc]['bl'][f"{num_sample}sample"]['coverage_scores'])
                scores[lc]['bl'][f"{num_sample}sample"]['std_score'] = Utils.stdev(scores[lc]['bl'][f"{num_sample}sample"]['coverage_scores'])
            # end for
            Utils.write_json(scores, result_file, pretty_format=True)
        # end if
    # end for
    return

def main_all(task,
             search_dataset_name,
             selection_method):
    st = time.time()
    logger_file = Macros.log_dir / f"seeds_exps_all_{task}_{search_dataset_name}_{selection_method}_pdrcov.log"
    result_file = Macros.pdr_cov_result_dir / f"seed_exp_bl_all_{task}_{search_dataset_name}_{selection_method}_pdrcov.json"
    Macros.pdr_cov_result_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(logger_file=logger_file,
                    logger_name='seed_exp_all_pdrcov_log')
    seed_rules = ProductionruleCoverage.get_our_seed_cfg_rules(
        task,
        search_dataset_name,
        selection_method,
        parse_all_sents=False,
        logger=logger
    )
    exp_rules = ProductionruleCoverage.get_our_exp_cfg_rules(
        task,
        search_dataset_name,
        selection_method,
        logger=logger
    )
    checklist_rules = ProductionruleCoverage.get_bl_cfg_rules(
        task,
        search_dataset_name,
        selection_method,
        logger=logger
    )
    
    scores = dict()
    for lc in tqdm(seed_rules.keys()):
        if lc not in scores.keys():
            logger.print(f"OURS_PDR_SEED_EXP_ALL::{lc}", end='::')
            our_sents, bl_sents = list(), list()
            scores[lc] = {
                'ours_seed': {
                    'coverage_scores': None
                },
                'ours_seed_exp': {
                    'coverage_scores': None
                },
                'bl': {
                    'coverage_scores': None
                }
            }
            seed_sents = list(seed_rules[lc].keys())
            exp_sents = list(exp_rules[lc].keys())
            bl_sents = list(checklist_rules[lc].keys())
            pdr1 = {
                s: seed_rules[lc][s]
                for s in seed_sents
            }
            pdr2 = {
                s: seed_rules[lc][s]
                for s in seed_sents
            }
            for e in exp_sents:
                if e not in pdr2.keys():
                    pdr2[e] = exp_rules[lc][e]
                # end if
            # end for
            pdr3 = {
                e: checklist_rules[lc][e]
                for e in bl_sents
            }
            pdr1_obj = ProductionruleCoverage(lc=lc,
                                              our_cfg_rules=pdr1)
            cov_score_ours_seed, _ = pdr1_obj.get_score()
            pdr2_obj = ProductionruleCoverage(lc=lc,
                                              our_cfg_rules=pdr2)
            cov_score_ours_seed_exp, _ = pdr2_obj.get_score()
            pdr3_obj = ProductionruleCoverage(lc=lc,
                                              our_cfg_rules=pdr3)
            cov_score_bl, _ = pdr3_obj.get_score()
            scores[lc]['ours_seed']['coverage_scores'] = cov_score_ours_seed
            scores[lc]['ours_seed_exp']['coverage_scores'] = cov_score_ours_seed_exp
            scores[lc]['bl']['coverage_scores'] = cov_score_bl
        # end if
        Utils.write_json(scores, result_file, pretty_format=True)
    # end for
    return


def main_mtnlp(task,
               search_dataset_name,
               selection_method):
    st = time.time()
    num_trials = 10
    logger_file = Macros.log_dir / f"mtnlp_{task}_{search_dataset_name}_{selection_method}_pdrcov.log"
    seed_dir = Macros.result_dir / f"templates_{task}_{search_dataset_name}_{selection_method}"
    mtnlp_dir = Macros.download_dir / 'MT-NLP'
    mtnlp_res_dir =  Macros.result_dir / 'mtnlp' / f"{task}_{search_dataset_name}_{selection_method}_sample"
    mtnlp_file = mtnlp_res_dir / 'mutations_s2lct_seed_samples.json'
    result_file = Macros.pdr_cov_result_dir / f"mtnlp_sample_{task}_{search_dataset_name}_{selection_method}_pdrcov.json"
    logger = Logger(logger_file=logger_file,
                    logger_name='mtnlt_mutation_log')
    
    # _seed_rules = ProductionruleCoverage.get_our_seed_cfg_rules(
    #     task,
    #     search_dataset_name,
    #     selection_method,
    #     parse_all_sents=False,
    #     logger=logger
    # )

    _exp_rules = ProductionruleCoverage.get_our_exp_cfg_rules(
        task,
        search_dataset_name,
        selection_method,
        logger=logger
    )
    
    mt_res = Utils.read_json(mtnlp_file)
    sample_file = mtnlp_dir / mt_res['sample_file']
    file_ind = re.search('raw_file(\d)\.txt', mt_res['sample_file']).group(1)
    seed_sents = list(mt_res['mutations'].keys())
    # seed_rules = dict()
    seed_lcs = dict()
    search_lns = [
        l.strip()
        for l in Utils.read_txt(mtnlp_dir / f"{task}_seed_samples_raw_file{file_ind}.txt")
        if l.strip()!=''
    ]
    for s in seed_sents:
        for l in search_lns:
            if l.split('::')[0].strip()==s:
                seed_lcs[s] = l.split('::')[-1].strip()
            # end if
        # end for
    # end for
        
    # end for
    
    # get exp_sents and their rules
    # exp_sents = [
    #     l.split('::')[0]
    #     for l in Utils.read_txt(mtnlp_dir / f"{task}_exp_samples_raw_file{file_ind}.txt")
    #     if l.strip()!=''
    # ]
    exp_sents = dict()
    req_dir = Macros.result_dir / 'reqs'
    req_file = req_dir / 'requirements_desc.txt'
    for s in seed_lcs.keys():
        lc_desc = seed_lcs[s].strip()
        lc_cksum = Utils.get_cksum(lc_desc)
        # _lc_cksum = Utils.get_cksum(lc_desc.lower())
        seed_file = seed_dir / f"cfg_expanded_inputs_{lc_cksum}.json"
        exp_sents[s] = list()
        # if os.path.exists(seed_dir / f"cfg_expanded_inputs_{_lc_cksum}.json") and \
        #    not os.path.exists(seed_dir / f"cfg_expanded_inputs_{lc_cksum}.json"):
        #     seed_file = seed_dir / f"cfg_expanded_inputs_{_lc_cksum}.json"    
        # # end if
        cfg_res = Utils.read_json(seed_file)
        print(seed_file)
        tokens = Utils.tokenize(s)
        _s = Utils.detokenize(tokens)
        for exp in cfg_res['inputs'][_s]['exp_inputs']:
            exp_sent = exp[5]
            exp_sents[s].append(exp_sent)
        # end for
    # end for
    logger.print(f"OURS_PDR_SAMPLE::mtnlp::")
    scores = {
        'ours_exp': list(),
        'mtnlp': list()
    }
    for t in tqdm(range(num_trials)):
        sample_exp_sents = list()
        for s in exp_sents.keys():
            exp_sent = random.sample(exp_sents[s], 1)
            sample_exp_sents.append(exp_sent)
        # end for

        sample_mt_sents = list()
        for s in seed_sents:
            ana_mt_sents = mt_res['mutations'][s]['ana']
            act_mt_sents = mt_res['mutations'][s]['act']
            if any(ana_mt_sents+act_mt_sents):
                mt_sent = random.sample(ana_mt_sents+act_mt_sents, 1)
                sample_mt_sents.append(mt_sent)
            # end if
        # end for

        sample_exp_rules = dict()
        for e in sample_exp_sents:
            seed_sent = [s for s in exp_sents.keys() if e in exp_sents[s]][0]
            lc = seed_lcs[seed_sent]
            sample_exp_rules[e] = _exp_rules[lc][e]
        # end for
    
        mt_rules = ProductionruleCoverage.get_mt_cfg_rules(
            task,
            search_dataset_name,
            selection_method,
            'mtnlp',
            sample_mt_sents,
            logger=logger
        )
        pdr1 = {
            s: sample_exp_rules[s]
            for s in sample_exp_sents
        }
        pdr2 = {
            s: mt_rules[s] 
            for s in sample_mt_sents
        }
        
        pdr_obj1 = ProductionruleCoverage(lc=None,
                                          our_cfg_rules=pdr1)
        pdr_obj2 = ProductionruleCoverage(lc=None,
                                          our_cfg_rules=pdr2)
        cov_score_exp, _ = pdr_obj1.get_score()
        cov_score_mt, _ = pdr_obj2.get_score()
        scores['ours_exp'].append(cov_score_exp)
        scores['mtnlp'].append(cov_score_mt)
    # end for
    Utils.write_json(scores, result_file, pretty_format=True)
    return
