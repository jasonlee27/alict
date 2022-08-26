import os
import nltk
import time
import multiprocessing

from tqdm import tqdm
from multiprocessing import Pool
# from functools import partial
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.preprocessing import normalize

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger
from ..testsuite.cfg.CFG import BeneparCFG
from ..testsuite.Search import ChecklistTestsuite

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
        self.bl_num_data = len(self.bl_cfg_rules.keys())
        self.scores = list()

    def get_score(self):
        our_rule_set = list()
        for r in self.our_cfg_rules.values():
            our_rule_set.extend(r)
        # end for
        our_rule_set = list(set(our_rule_set))

        bl_rule_set = list()
        for r in self.bl_cfg_rules.values():
            bl_rule_set.extend(r)
        # end for
        bl_rule_set = list(set(bl_rule_set))
        return len(our_rule_set), len(bl_rule_set)
    
    @classmethod
    def get_our_seed_cfg_rules(cls,
                               task,
                               dataset_name,
                               selection_method,
                               num_seeds,
                               num_trials,
                               logger=None):
        if num_seeds<0:
            cfg_res_file = Macros.result_dir / f"cfg_expanded_inputs{num_trials}_{task}_{dataset_name}_{selection_method}.json"
            res_file = Macros.pdr_cov_result_dir / f"seed_cfg_rules{num_trials}_{task}_{dataset_name}.json"
        else:
            cfg_res_file = Macros.result_dir / f"cfg_expanded_inputs{num_trials}_{task}_{dataset_name}_{selection_method}_{num_seeds}seeds.json"
            res_file = Macros.pdr_cov_result_dir / f"seed_cfg_rules{num_trials}_{task}_{dataset_name}.json"
        # end if
        if os.path.exists(cfg_res_file):
            seed_rules = dict()
            cfg_results = Utils.read_json(cfg_res_file)
            for cfg_res in cfg_results:
                lc = cfg_res['requirement']['description']
                seed_inputs = list()
                seed_rules[lc] = dict()
                if logger is not None:
                    logger.print(f"OUR_SEED::{lc}::")
                # end if
                
                for seed in cfg_res['inputs'].keys():
                    cfg_seed = cfg_res['inputs'][seed]['cfg_seed']
                    seed_rules[lc][seed] = get_pdr_per_sent(cfg_seed)
                # end for
                Utils.write_json(seed_rules, res_file, pretty_format=True)
            # end for
        else:
            if num_seeds<0:
                data_file = Macros.result_dir / f"seed_inputs{num_trials}_{task}_{dataset_name}.json"
                res_file = Macros.result_dir / f"seed_cfg_rules{num_trials}_{task}_{dataset_name}.json"
            else:
                data_file = Macros.result_dir / f"seed_inputs{num_trials}_{task}_{dataset_name}_{num_seeds}seeds.json"
                res_file = Macros.result_dir / f"seed_cfg_rules{num_trials}_{task}_{dataset_name}_{num_seeds}seeds.json"
            # end if
            seed_dicts = Utils.read_json(data_file)
            if os.path.exists(res_file):
                seed_rules = Utils.read_json(res_file)
            else:
                seed_rules = dict()
            # end if
            for seed in seed_dicts:
                lc = seed['requirement']['description']
                if lc not in seed_rules.keys():
                    if logger is not None:
                        logger.print(f"OUR_SEED::{lc}::")
                    # end if
                    seed_rules[lc] = dict()
                    args = [(s[1],) for s in seed['seeds']]
                    pool = Pool(processes=NUM_PROCESSES)
                    results = pool.starmap_async(get_cfg_rules_per_sent,
                                                 args,
                                                 chunksize=len(args) // NUM_PROCESSES).get()
                    for r in results:
                        seed_rules[lc][r['sent']] = r['cfg_rules']
                    # end for
                    Utils.write_json(seed_rules, res_file, pretty_format=True)
                # end if
            # end for
        # end if
        
        # if num_seeds<0:
        #     data_file = Macros.result_dir / f"seed_inputs_{task}_{dataset_name}.json"
        #     res_file = Macros.result_dir / f"seed_cfg_rules_{task}_{dataset_name}.json"
        # else:
        #     data_file = Macros.result_dir / f"seed_inputs_{task}_{dataset_name}_{num_seeds}seeds.json"
        #     res_file = Macros.result_dir / f"seed_cfg_rules_{task}_{dataset_name}_{num_seeds}seeds.json"
        # # end if
        # seed_dicts = Utils.read_json(data_file)
        # if os.path.exists(res_file):
        #     seed_rules = Utils.read_json(res_file)
        # else:
        #     seed_rules = dict()
        # # end if
        # for seed in seed_dicts:
        #     lc = seed['requirement']['description']
        #     if lc not in seed_rules.keys():
        #         if logger is not None:
        #             logger.print(f"OUR_SEED::{lc}::")
        #         # end if
        #         seed_rules[lc] = dict()
        #         args = [(s[1],) for s in seed['seeds']]
        #         pool = Pool(processes=NUM_PROCESSES)
        #         results = pool.starmap_async(get_cfg_rules_per_sent,
        #                                      args,
        #                                      chunksize=len(args) // NUM_PROCESSES).get()
        #         for r in results:
        #             seed_rules[lc][r['sent']] = r['cfg_rules']
        #         # end for
        #         Utils.write_json(seed_rules, res_file, pretty_format=True)
        #     # end if
        # # end for
        return seed_rules

    @classmethod
    def get_our_exp_cfg_rules(cls,
                              task,
                              dataset_name,
                              selection_method,
                              num_seeds,
                              num_trials,
                              logger=None):
        if num_seeds<0:
            data_file = Macros.result_dir / f"cfg_expanded_inputs{num_trials}_{task}_{dataset_name}_{selection_method}.json"
            res_file = Macros.pdr_cov_result_dir / f"exp_cfg_rules{num_trials}_{task}_{dataset_name}_{selection_method}.json"
        else:
            data_file = Macros.result_dir / f"cfg_expanded_inputs{num_trials}_{task}_{dataset_name}_{selection_method}_{num_seeds}seeds.json"
            res_file = Macros.pdr_cov_result_dir / f"exp_cfg_rules{num_trials}_{task}_{dataset_name}_{selection_method}_{num_seeds}seeds.json"
        # end if
        exp_dicts = Utils.read_json(data_file)
        if os.path.exists(str(res_file)):
            exp_rules = Utils.read_json(res_file)
        else:
            exp_rules = dict()
        # end if
        for exp in exp_dicts:
            lc = exp['requirement']['description']
            if logger is not None:
                logger.print(f"OUR_EXP::{lc}::")
            # end if
            if lc not in exp_rules.keys():
                exp_rules[lc] = dict()
                for seed in exp['inputs'].keys():
                    cfg_seed = exp['inputs'][seed]
                    pdr_seed = get_pdr_per_sent(cfg_seed)
                    pdrs = list()
                    for exp in exp['inputs'][seed]['exp_inputs']:
                        pdr_exp = pdr_seed.copy()
                        cfg_from, cfg_to, exp_sent = exp[1], exp[2], exp[5]
                        pdr_exp.remove(cfg_from)
                        pdr_exp.append(cfg_to)
                        pdrs.append(pdr_exp)
                    # end for
                    exp_rules[lc][seed] = pdrs
                # end for
                Utils.write_json(exp_rules, res_file, pretty_format=True)
            # end if
        # end for
        return exp_rules
    
    @classmethod
    def get_bl_cfg_rules(cls,
                         task,
                         dataset_name,
                         selection_method,
                         num_seeds,
                         num_trials,
                         logger=None):
        if num_seeds<0:
            data_file = Macros.result_dir / f"cfg_expanded_inputs{num_trials}_{task}_{dataset_name}_{selection_method}.json"
            res_file = Macros.pdr_cov_result_dir / f"bl_cfg_rules_{task}_checklist.json"
        else:
            data_file = Macros.result_dir / f"cfg_expanded_inputs{num_trials}_{task}_{dataset_name}_{selection_method}_{num_seeds}seeds.json"
            res_file = Macros.pdr_cov_result_dir / f"bl_cfg_rules_{task}_checklist.json"
        # end if
    
        seed_dicts = Utils.read_json(data_file)
        if os.path.exists(str(res_file)):
            bl_rules = Utils.read_json(res_file)
        else:
            bl_rules = dict()
        # end if
        for seed in seed_dicts:
            lc = seed['requirement']['description']
            if lc not in bl_rules.keys():
                if logger is not None:
                    logger.print(f"BL::{lc}::")
                # end if
                bl_rules[lc] = dict()
                sents = ChecklistTestsuite.get_sents(
                    Macros.checklist_sa_dataset_file,
                    lc
                )
                args = [s[1] for s in sents]
                pool = Pool(processes=NUM_PROCESSES)
                results = pool.map_async(get_cfg_rules_per_sent,
                                         args,
                                         chunksize=len(args) // NUM_PROCESSES).get()
                for r in results:
                    bl_rules[lc][r['sent']] = r['cfg_rules']
                # end for
                Utils.write_json(bl_rules, res_file, pretty_format=True)
            # end if
        # end for
        return bl_graph_dict
    

def main_seed(task,
              search_dataset_name,
              selection_method,
              num_seeds,
              num_trials):
    st = time.time()
    num_trials = '' if num_trials==1 else str(num_trials)
    if num_seeds<0:
        logger_file = Macros.log_dir / f"seeds{num_trials}_{task}_{search_dataset_name}_pdrcov.log"
        result_file = Macros.pdr_cov_result_dir / f"seeds{num_trials}_{task}_{search_dataset_name}_pdrcov.json"
    else:
        logger_file = Macros.log_dir / f"seeds{num_trials}_{task}_{search_dataset_name}_{num_seeds}seeds_pdrcov.log"
        result_file = Macros.pdr_cov_result_dir / f"seeds{num_trials}_{task}_{search_dataset_name}_{num_seeds}seeds_pdrcov.json"
    # end if
    Macros.pdr_cov_result_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(logger_file=logger_file,
                    logger_name='seed_pdrcov_log')
    seed_rules = ProductionruleCoverage.get_our_seed_cfg_rules(
        task,
        search_dataset_name,
        selection_method,
        num_seeds,
        num_trials,
        logger=logger
    )
    checklist_rules = ProductionruleCoverage.get_bl_cfg_rules(
        task,
        search_dataset_name,
        selection_method,
        num_seeds,
        num_trials,
        logger=logger
    )
    if os.path.exists(str(result_file)):
        scores = Utils.read_json(result_file)
    else:
        scores = dict()
    # end if
    
    for lc in tqdm(seed_rules.keys()):
        if lc not in scores.keys():
            logger.print(f"OURS::{lc}", end='::')
            pdr_obj = ProductionruleCoverage(lc=lc,
                                             our_cfg_rules=seed_rules[lc],
                                             bl_cfg_rules=checklist_rules[lc])
            scores[lc] = {
                'our_num_data': pdr_obj.our_num_data,
                'bl_num_data': pdr_obj.bl_num_data,
                'coverage_score': pdr_obj.get_score()
            }
            Utils.write_json(scores, result_file, pretty_format=True)
        # end if
    # end for
    ft = time.time()
    logger.print(f"ProductionruleCoverage.main_seed::{round(ft-st,3)}sec")
    return
