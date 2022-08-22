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

NUM_PROCESSES = 15 # Macros.num_processes

def get_cfg_rules_per_sent(sent):
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
    return {
        'sent': sent,
        'cfg_rules': list(set(rules))
    }

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
                               logger=None):
        data_file = Macros.result_dir / f"seed_inputs_{task}_{dataset_name}.json"
        res_file = Macros.result_dir / f"seed_cfg_rules_{task}_{dataset_name}.json"
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
        return seed_rules

    @classmethod
    def get_our_exp_cfg_rules(cls,
                              task,
                              dataset_name,
                              selection_method,
                              logger=None):
        data_file = Macros.result_dir / f"cfg_expanded_inputs_{task}_{dataset_name}_{selection_method}.json"
        res_file = Macros.result_dir / f"exp_cfg_rules_{task}_{dataset_name}_{selection_method}.json"
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
                args = list()
                for seed in exp['inputs'].keys():
                    exp_inputs = [(e[5],) for e in exp['inputs'][seed]['exp_inputs']]
                    exp_inputs.append((seed,))
                    args.extend(exp_inputs)
                # end for
                pool = Pool(processes=NUM_PROCESSES)
                results = pool.starmap_async(get_cfg_rules_per_sent,
                                             args,
                                             chunksize=len(args) // NUM_PROCESSES).get()
                for r in results:
                    exp_rules[lc][r['sent']] = r['cfg_rules']
                # end for
                Utils.write_json(exp_rules, res_file, pretty_format=True)
            # end if
        # end for
        return exp_rules
    
    @classmethod
    def get_bl_cfg_rules(cls,
                         task,
                         dataset_name,
                         logger=None):
        data_file = Macros.result_dir / f"seed_inputs_{task}_{dataset_name}.json"
        res_file = Macros.result_dir / f"bl_cfg_rules_{task}_checklist.json"
        seed_dicts = Utils.read_json(data_file)

        if os.path.exists(str(res_file)):
            bl_rules = Utils.read_json(res_file)
        else:
            bl_rules = dict()
        # end if
        for seed in seed_dicts:
            lc = exp['requirement']['description']
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
                results = pool.map_async(cls.get_cfg_rules_per_sent,
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
              selection_method):
    st = time.time()
    logger_file = Macros.log_dir / f"seeds_{task}_{search_dataset_name}_pdrcov.log"
    logger = Logger(logger_file=logger_file,
                    logger_name='seed_pdrcov_log')
    Macros.selfbleu_result_dir.mkdir(parents=True, exist_ok=True)
    result_file = Macros.pdr_cov_result_dir / f"seeds_{task}_{search_dataset_name}_pdr_coverage.json"
    seed_rules = ProductionruleCoverage.get_our_seed_cfg_rules(
        task,
        search_dataset_name,
        logger=logger
    )
    checklist_rules = ProductionruleCoverage.get_bl_cfg_rules(
        task,
        search_dataset_name,
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
