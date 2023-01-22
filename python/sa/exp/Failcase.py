import os
import nltk
import time
import random
import multiprocessing

from tqdm import tqdm
from .SelfBleu import read_our_seeds, read_our_exps
from ..model.Result import Result

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger

def main_p2f_f2p(task,
                 search_dataset_name,
                 selection_method):
    num_trials = 10
    num_samples = [50, 100, 150, 200]
    logger_file = Macros.log_dir / f"p2f_f2p_{task}_{search_dataset_name}_{selection_method}.log"
    result_file = Macros.result_dir / f"p2f_f2p_{task}_{search_dataset_name}_{selection_method}.json"
    test_result_file = Macros.result_dir / f"test_results_{task}_{search_dataset_name}_{selection_method}" / "test_result_analysis.json"
    test_result = Utils.read_json(test_result_file)
    model_names = list(test_result.keys())
    _, texts_seed_ours = read_our_seeds(task,
                                        search_dataset_name,
                                        selection_method)
    scores = dict()
    for m in test_result.keys():
        scores[m] = dict()
        test_result_model = test_result[m]
        for lc in texts_seed_ours.keys():
            test_result_model_lc = [l for l in test_result_model if l['req']==lc or l['req']==lc.lower()][0]
            p2f_cases = dict()
            f2p_cases = dict()
            for pf in test_result_model_lc['pass->fail']:
                if pf['from']['sent'] not in p2f_cases.keys():
                    p2f_cases[pf['from']['sent']] = len(pf['to'])
                else:
                    p2f_cases[pf['from']['sent']] += len(pf['to'])
                # end if
            # end for
            for pf in test_result_model_lc['fail->pass']:
                if pf['from']['sent'] not in p2f_cases.keys():
                    f2p_cases[pf['from']['sent']] = len(pf['to'])
                else:
                    f2p_cases[pf['from']['sent']] += len(pf['to'])
                # end if
            # end for
            scores[m][lc] = {
                f"{num_sample}sample": {
                    'p2f': list(),
                    'f2p': list()
                }
                for num_sample in num_samples
            }
            for num_sample in num_samples:
                for num_trial in range(num_trials):
                    random.seed(num_trial)
                    our_sents = random.sample(texts_seed_ours[lc], min(len(texts_seed_ours[lc]), num_sample))
                    our_sents = [
                        Utils.detokenize(Utils.tokenize(s))
                        for s in our_sents
                    ]
                    scores[m][lc][f"{num_sample}sample"]['p2f'].append(sum([p2f_cases[s] if s in p2f_cases.keys() else 0 for s in our_sents]))
                    scores[m][lc][f"{num_sample}sample"]['f2p'].append(sum([f2p_cases[s] if s in f2p_cases.keys() else 0 for s in our_sents]))
                # end for
            # end for
        # end for
        Utils.write_json(scores, result_file, pretty_format=True)
    # end for
    return

def main_fail(task,
              search_dataset_name,
              selection_method):
    num_trials = 10
    num_samples = [50, 100, 150, 200]
    logger_file = Macros.log_dir / f"failcases_{task}_{search_dataset_name}_{selection_method}.log"
    result_file = Macros.result_dir / f"failcases_{task}_{search_dataset_name}_{selection_method}.json"
    result_dir = Macros.result_dir / f"test_results_{task}_{search_dataset_name}_{selection_method}" 
    # test_result_file = result_dir / 'test_result_analysis.json'
    # test_result = Utils.read_json(test_result_file)
    raw_test_result_file = result_dir / 'test_results.txt'
    raw_test_result = Result.parse_results(raw_test_result_file, Macros.sa_models_file)
    _, texts_seed_ours = read_our_seeds(task,
                                        search_dataset_name,
                                        selection_method)
    _, texts_exp_ours = read_our_exps(task,
                                      search_dataset_name,
                                      selection_method)
    
    # reqs = Requirements.get_requirements(nlp_task)
    scores = dict()
    for model in Utils.read_txt(Macros.sa_models_file):
        model = model.strip()
        scores[model] = dict()
        model_results = result_dict[model]
        reqs = sorted(set([r['req'] for r in model_results]))
        for r in reqs:
            lc = r['description']
            scores[m][lc] = {
                f"{num_sample}sample": 0
                for num_sample in num_samples
            }
            seeds_fail = [mr for mr in model_results if mr['req']==r and mr['sent_type']=='SEED'][0]['fail']
            exps_fail = [mr for mr in model_results if mr['req']==r and mr['sent_type']=='EXP'][0]['fail']
            for num_sample in num_samples:
                for num_trial in range(num_trials):
                    random.seed(num_trial)
                    our_sents = random.sample(texts_seed_ours[lc]+texts_exp_ours[lc], min(len(texts_seed_ours[lc]), num_sample))
                    our_sents = [
                        Utils.detokenize(Utils.tokenize(s))
                        for s in our_sents
                    ]
                    num_fails = len([s for s in ours_sents if s in seeds_fail or s in exps_fail])
                    scores[m][lc][f"{num_sample}sample"] = num_fails
                # end for
            # end for
        # end for
        Utils.write_json(scores, result_file, pretty_format=True)
    # end for
    return
