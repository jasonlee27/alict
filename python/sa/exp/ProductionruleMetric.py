import os
import nltk
import time
import multiprocessing

from multiprocessing import Pool
# from functools import partial
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.preprocessing import normalize

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger
from ..testsuite.cfg.CFG import BeneparCFG
from ..testsuite.Search import ChecklistTestsuite

NUM_PROCESSES_IN_USE = 3 # os.cpu_count()

class ProductionruleMetric:
    
    def __init__(self, cfgs=None):
        # the json file used for retraining sa models
        self.cfgs = cfgs
        self.reference = None
        self.num_data = len(self.cfgs)
        self.scores = list()

    def depDistance(self, hyp_graph, ref_graphs):
        # count occurences of each type of relationship
        # reference: https://github.com/RobustNLP/TestTranslation/tree/master/code/SIT
        res = list()
        hyp_counts = dict()
        for rule in hyp_graph:
            hyp_counts[rule] = hyp_counts.get(rule, 0) + 1
        # end for
        for ref_graph in ref_graphs:
            ref_counts = dict()
            for rule in ref_graph:
                ref_counts[rule] = ref_counts.get(rule, 0) + 1
            # end for
            all_deps = set(list(hyp_counts.keys()) + list(ref_counts.keys()))
            diffs = 0
            for dep in all_deps:
                diffs += abs(hyp_counts.get(dep,0) + ref_counts.get(dep,0))
            # end for
            res.append(diffs)
        # end for
        return res

    def get_score(self):
        pool = Pool(processes=NUM_PROCESSES_IN_USE)
        result = list()
        reference = self.cfgs
        self.scores = list()
        def callback(res):
            self.scores.append(res)
            return
        for d_i in range(self.num_data):
            hypothesis = reference[d_i]
            other = reference[:d_i] + reference[d_i+1:]
            result.append(pool.apply_async(self.depDistance,
                                           args=(hypothesis, other),
                                           callback=callback))
        # end for
        # scores = list()
        cnt = 1
        # for r in result:
        #     scores.extend(r.get())
        # # end for
        # cnt = len(self.scores)
        scores = list(normalize([self.scores])[0])
        pool.close()
        pool.join()
        return float("{:.3f}".format(sum(self.scores) / len(self.scores)))

    @classmethod
    def get_seed_graph(cls, cfg_seed):
        seed_graph = list()
        for lhs, rhss in cfg_seed.items():
            for rhs in rhss:
                if type(rhs['pos'])==list:
                    rhs_pos = rhs['pos']
                    pd_rule = f"{lhs} -> {str(rhs_pos)}"
                    seed_graph.append(pd_rule)
                elif type(rhs['pos'])==str and rhs['pos']!=rhs['word'][0]:
                    rhs_pos = [rhs['pos']]
                    pd_rule = f"{lhs} -> {str(rhs_pos)}"
                    seed_graph.append(pd_rule)
                # end if
            # end for
        # end for
        return seed_graph

    @classmethod
    def get_exp_graph(cls, seed_graph, exps):
        exp_graphs = list()
        for exp in exps:
            pd_rule_from = exp[1]
            pd_rule_to = exp[2]
            target_ind = seed_graph.index(pd_rule_from)
            exp_graph = seed_graph.copy()
            exp_graph[target_ind] = pd_rule_to
            exp_graphs.append(exp_graph)
        # end for
        return exp_graphs
    
    # @classmethod
    # def get_our_cfg_graph(cls,
    #                       task,
    #                       dataset_name,
    #                       selection_method):
    #     data_file = Macros.result_dir / f"cfg_expanded_inputs_{task}_{dataset_name}_{selection_method}.json"
    #     exp_dicts = Utils.read_json(data_file)
    #     seed_graph_dict = dict()
    #     exp_graph_dict = dict()
    #     for exp in exp_dicts:
    #         lc = exp['requirement']['description']
    #         seeds = exp['inputs']
    #         seed_graphs = list()
    #         exp_graphs = list()
    #         for seed in seeds.keys():
    #             cfg_seed = seeds[seed]['cfg_seed']
    #             exps = seeds[seed]['exp_inputs']
    #             seed_graph = cls.get_seed_graph(cfg_seed)
    #             seed_graphs.append(seed_graph)
    #             exp_graphs = cls.get_exp_graph(seed_graph, exps)
    #             exp_graphs.extend(exp_graphs)
    #         # end for
    #         seed_graph_dict[lc] = seed_graphs
    #         exp_graph_dict[lc] = exp_graphs
    #     # end for
    #     return seed_graph_dict, exp_graph_dict

    @classmethod
    def get_our_cfg_graph(cls,
                          task,
                          dataset_name,
                          selection_method):
        data_file = Macros.result_dir / f"seed_inputs_{task}_{dataset_name}.json"
        exp_dicts = Utils.read_json(data_file)
        seed_graph_dict = dict()
        exp_graph_dict = dict()
        for exp in exp_dicts:
            lc = exp['requirement']['description']
            seeds = exp['seeds']
            seed_graphs = list()
            exp_graphs = list()
            inputs = [Utils.tokenize(s[1]) for s in seeds]
            tree_dicts = BeneparCFG.get_seed_cfgs(inputs)
            for tree_dict in tree_dicts:
                cfg_seed = tree_dict['rule']
                seed_graph = cls.get_seed_graph(cfg_seed)
                seed_graphs.append(seed_graph)
            # end for
            # for seed in seeds:
            #     tree_dict = BeneparCFG.get_seed_cfg(seed[1])
            #     cfg_seed = tree_dict['rule']
            #     seed_graph = cls.get_seed_graph(cfg_seed)
            #     seed_graphs.append(seed_graph)
            # # end for
            seed_graph_dict[lc] = seed_graphs
        # end for
        return seed_graph_dict, exp_graph_dict
    
    @classmethod
    def get_bl_cfg_graph(cls,
                         task,
                         dataset_name,
                         selection_method):
        data_file = Macros.result_dir / f"cfg_expanded_inputs_{task}_{dataset_name}_{selection_method}.json"
        exp_dicts = Utils.read_json(data_file)
        bl_graph_dict = dict()
        for exp in exp_dicts:
            lc = exp['requirement']['description']
            bl_graphs = list()
            sents = ChecklistTestsuite.get_sents(
                Macros.checklist_sa_dataset_file,
                lc
            )
            for s in sents:
                tree_dict = BeneparCFG.get_seed_cfg(s[1])
                cfg_bl = tree_dict['rule']
                bl_graph = cls.get_seed_graph(cfg_bl)
                bl_graphs.append(bl_graph)
            # end for
            bl_graph_dict[lc] = bl_graphs
        # end for
        return bl_graph_dict
    

def main_seed(task,
              search_dataset_name,
              selection_method):
    logger_file = Macros.log_dir / f"seeds_{task}_{search_dataset_name}_pdrdiv.log"
    logger = Logger(logger_file=logger_file,
                    logger_name='seed_pdrdiv_log')
    Macros.selfbleu_result_dir.mkdir(parents=True, exist_ok=True)
    result_file = Macros.selfbleu_result_dir / f"seeds_{task}_{search_dataset_name}_selfbleu.json"
    seed_graphs, _ = ProductionruleMetric.get_our_cfg_graph(
        task,
        search_dataset_name,
        selection_method
    )
    result = dict()
    scores = dict()
    scores_baseline = dict()
    for lc in seed_graphs.keys():
        print(lc)
        st = time.time()
        logger.print(f"OURS::{lc}", end='::')
        pdr = ProductionruleMetric(cfgs=seed_graphs[lc])
        scores[lc] = {
            'num_data': pdr.num_data,
            'score': pdr.get_score()
        }
        result = {
            'ours': scores,
            'checklist': scores_baseline
        }
        Utils.write_json(result, result_file, pretty_format=True)
        ft = time.time()
        logger.print(f"{round(ft-st,2)}secs", end='::')
        logger.print(f"num_data:{scores[lc]['num_data']}::score:{scores[lc]['score']}")
    # end for

    checklist_graphs = ProductionruleMetric.get_bl_cfg_graph(
        task,
        search_dataset_name,
        selection_method
    )
    for lc in checklist_graphs.keys():
        print(lc)
        st = time.time()
        logger.print(f"BL::{lc}", end='::')
        pdr = ProductionruleMetric(cfgs=checklist_graphs[lc])
        scores_baseline[lc] = {
            'num_data': pdr.num_data,
            'score': pdr.get_score()
        }
        result = {
            'ours': scores,
            'checklist': scores_baseline
        }
        Utils.write_json(result, result_file, pretty_format=True)
        ft = time.time()
        logger.print(f"{round(ft-st,2)}secs", end='::')
        logger.print(f"num_data:{scores_baseline[lc]['num_data']}::score:{scores_baseline[lc]['score']}")
    # end for
    result = {
        'ours': scores,
        'checklist': scores_baseline
    }
    Utils.write_json(result, result_file, pretty_format=True)
    return
    
