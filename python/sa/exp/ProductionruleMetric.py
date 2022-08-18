import os
import nltk
import multiprocessing

from multiprocessing import Pool
# from functools import partial
from nltk.translate.bleu_score import SmoothingFunction

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..testsuite.cfg.CFG import BeneparCFG

class ProductionruleMetric:

    NUM_PROCESSES_IN_USE = 3 # os.cpu_count()
    
    def __init__(self, graphs=None):
        # the json file used for retraining sa models
        self.graphs = graphs
        self.reference = None
        self.num_data = len(self.graphs)
        
    def depDistance(self, graph1, graph2):
        # count occurences of each type of relationship
        # reference: https://github.com/RobustNLP/TestTranslation/tree/master/code/SIT
        res = list()
        for g in graph2:
            
	    counts1 = dict()
	    for i in graph1:
	        counts1[i[1]] = counts1.get(i[1], 0) + 1
            # end for
            counts2 = dict()
	    for i in graph2:
	        counts2[i[1]] = counts2.get(i[1], 0) + 1
            # end for
	    all_deps = set(list(counts1.keys()) + list(counts2.keys()))
	    diffs = 0
	    for dep in all_deps:
	        diffs += abs(counts1.get(dep,0) - counts2.get(dep,0))
            # end for
            res.append(diffs)
        # end for
        return res

    @classmethod
    def get_score(self):
        pool = Pool(processes=cls.NUM_PROCESSES_IN_USE)
        result = list()
        reference = self.graphs
        for d_i in range(self.num_data):
            hypothesis = reference[d_i]
            other = reference[:d_i] + reference[d_i+1:]
            result.append(pool.apply_async(cls.depDistance, args=(hypothesis, other)))
        # end for
        score = 0.
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        # end for
        pool.close()
        pool.join()
        return float("{:.3f}".format(score / cnt))

    @classmethod
    def get_seed_graph(cls, cfg_seed):
        seed_graph = list()
        for lhs, rhss in cfg_seed.keys():
            for rhs in rhss:
                rhs_pos = rhs['pos'] if type(rhs['pos'])==list else [rhs['pos']]
                pd_rule = f"{lhs} -> {str(rhs_pos)}"
                seed_graph.append(pd_rule)
            # end for
        # end for
        return seed_graph

    @classmethod
    def get_exp_graph(cls, seed_graph, exps):
        exp_graphs = list()
        for exp in exps:
            pd_rule_from = exp[1]
            target_ind = seed_graph.index(pd_rule_from)
            exp_graph = seed_graph.copy()
            pd_rule_to = exp[2]
            exp_graph[target_ind] = pd_rule_to
            exp_graphs.append(exp_graph)
        # end for
        return exp_graphs
    
    @classmethod
    def get_our_cfg_graph(cls,
                          task,
                          dataset_name,
                          selection_method):
        data_file = Macros.result_dir / f"cfg_expanded_inputs_{task}_{dataset_name}_{selection_method}.json"
        exp_dicts = Utils.read_json(data_file)
        seed_graph_dict = dict()
        exp_graph_dict = dict()
        for exp in exp_dicts:
            lc = exp['requirement']['description']
            seeds = exp['inputs']
            seed_graphs = list()
            exp_graphs = list()
            for seed in seeds.keys():
                cfg_seed = seeds[seed]['cfg_seed']
                exps = seeds[seed]['exp_inputs']
                seed_graph = cls.get_seed_graph(cfg_seed)
                seed_graphs.append(seed_graph)
                exp_graphs = cls.get_exp_graph(cfg_seed, exps)
                exp_graphs.extend(exp_graph)
            # end for
            seed_graph_dict[lc] = seed_graphs
            exp_graph_dict[lc] = exp_graphs
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
            lc = seeds['requirement']['description']
            bl_graphs = list()
            sents = ChecklistTestsuite.get_sents(
                testsuite_file,
                lc
            )
            for s in sents:
                tree_dict = BeneparCFG.get_seed_cfg(s[1])
                cfg_bl = tree_dict['rule']
                bl_graph = cls.get_seed_graph(cfg_bl)
                bl_graphs.append(bl_seed)
            # end for
            bl_graph_dict[lc] = bl_graphs
        # end for
        return bl_graph_dict
    

def main_seed(task,
              dataset_name,
              selection_method):
    logger_file = Macros.log_dir / f"seeds_{task}_{search_dataset_name}_pdrdiv.log"
    logger = Logger(logger_file=logger_file,
                    logger_name='seed_pdrdiv_log')
    
    seed_graphs, _ = read_our_cfg_graph(task,
                                        search_dataset_name,
                                        selection_method)
    scores = dict()
    for lc in seed_graphs.keys():
        st = time.time()
        logger.print(f"OURS::{lc}", end='::')
        sbleu = SelfBleu(texts=texts_ours[lc])
        scores[lc] = {
            'num_data': sbleu.num_data,
            'score': sbleu.get_score()
        }
        ft = time.time()
        logger.print(f"{round(ft-st,2)}secs", end='::')
        logger.print(f"num_data:{scores[lc]['num_data']}::score:{scores[lc]['score']}")
    # end for

    _, texts_checklist = read_checklist_testcases(task,
                                                  search_dataset_name,
                                                  selection_method)
    scores_baseline = dict()
    for lc in texts_checklist.keys():
        st = time.time()
        logger.print(f"BL::{lc}", end='::')
        sbleu = SelfBleu(texts=texts_checklist[lc])
        scores_baseline[lc] = {
            'num_data': sbleu.num_data,
            'score': sbleu.get_score()
        }
        ft = time.time()
        logger.print(f"{round(ft-st,2)}secs", end='::')
        logger.print(f"num_data:{scores_baseline[lc]['num_data']}::score:{scores_baseline[lc]['score']}")
    # end for

    Macros.selfbleu_result_dir.mkdir(parents=True, exist_ok=True)
    result_file = Macros.selfbleu_result_dir / f"seeds_{task}_{search_dataset_name}_selfbleu.json"
    result = {
        'ours': scores,
        'checklist': scores_baseline
    }
    Utils.write_json(result, result_file, pretty_format=True)
    return
    
