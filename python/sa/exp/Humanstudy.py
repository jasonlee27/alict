# This script is to sample sentences
# from seed/exp sentences for pilot study
# Incorrect input (Ground Truth): # of l{i}_model != l{i}_human
# Reported bugs (Approach): # of l{i}_ours != l{i}_model


from typing import *

import re, os
import nltk
import copy
import random
import numpy
import spacy

from pathlib import Path
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger

random.seed(27)

class Humanstudy:

    # SENTIMENT_MAP = {
    #     1: 'strong_neg',
    #     2: 'weak_neg',
    #     3: 'strong_neutral',
    #     4: 'weak_neg',
    #     5: 'strong_pos'
    # }

    SENTIMENT_MAP_FROM_STR = {
        'negative': [1,2],
        'neutral': [3],
        'positive': [4,5],
        "['positive', 'neutral']": [3,4,5]
    }
    SENTIMENT_MAP_FROM_SCORE = {
        '0': [1,2],
        '1': [3],
        '2': [4,5],
        "None": [3,4,5]
    }

    
    @classmethod
    def read_sentences(cls, json_file: Path, include_label=False):
        inputs = Utils.read_json(json_file)
        results = dict()
        for inp in inputs:
            req = inp['requirement']['description']
            seeds, exps = list(), list()
            for seed in inp['inputs'].keys():
                if include_label:
                    label = inp['inputs'][seed]['label']
                    seeds.append((seed, cls.SENTIMENT_MAP_FROM_STR[str(label)]))
                    exps.extend([(e[5], cls.SENTIMENT_MAP_FROM_STR[str(label)]) for e in inp['inputs'][seed]['exp_inputs']])
                else:
                    seeds.append(seed)
                    exps.extend([e[5] for e in inp['inputs'][seed]['exp_inputs']])
                # end if
            # end for
            results[req] = {
                'seed': seeds,
                'exp': exps
            }
        # end for
        return results

    @classmethod
    def sample_sents(cls, sent_dict: Dict, num_samples=10):
        sample_results = dict()
        for req in sent_dict.keys():
            num_seed = len(sent_dict[req]['seed'])
            num_exp = len(sent_dict[req]['exp'])
            seed_ids = list(range(num_seed))
            exp_ids = list(range(num_exp))
            random.shuffle(seed_ids)
            random.shuffle(exp_ids)
            sample_results[req] = {
                'seed': [(sent_dict[req]['seed'][idx], req) for idx in seed_ids[:num_samples]],
                'exp': [(sent_dict[req]['exp'][idx], req) for idx in exp_ids[:num_samples]]
            }
        # end for
        return sample_results
    
    @classmethod
    def write_samples(cls, sample_dict: Dict):
        seeds, exps = list(), list()
        for req in sample_dict.keys():
            seeds.extend(sample_dict[req]['seed'])
            exps.extend(sample_dict[req]['exp'])
        # end for
        seed_res = ""
        exp_res = ""
        random.shuffle(seeds)
        random.shuffle(exps)
        for s, r in seeds:
            seed_res += f"{s} :: {r}\n\n\n"
        # end for
        for e, r in exps:
            exp_res += f"{e} :: {r}\n\n\n"
        # end for
        res_dir = Macros.result_dir / 'human_study'
        res_dir.mkdir(parents=True, exist_ok=True)
        Utils.write_txt(seed_res, res_dir / "seed_samples_raw.txt")
        Utils.write_txt(exp_res, res_dir / "exp_samples_raw.txt")
        print(f"num_seed_samples: {len(seeds)}\nnum_exp_samples: {len(exps)}")
        return
    
    @classmethod
    def read_results(cls, res_file: Path, num_samples: int=100):
        res_lines = Utils.read_txt(res_file)
        res = dict()
        num_sents = 0
        for l in res_lines:
            if re.search("\:\:\d$", l) and num_sents<num_samples:
                l_split = l.strip().split('::')
                tokens = Utils.tokenize(l_split[0])
                sent = Utils.detokenize(tokens)
                res[sent] = l_split[1]
                num_sents += 1
            # end if
        # end for
        return res

    @classmethod
    def get_target_results(cls, target_file, seed_results, exp_results=None):
        sent_dict = cls.read_sentences(target_file, include_label=True)
        res = dict()
        seed_sents = list(seed_results.keys())
        exp_sents = list(exp_results.keys())
        for r in sent_dict.keys():
            for s in sent_dict[r]['seed']:
                if s[0] is not None:
                    sent = s[0]
                    tokens = Utils.tokenize(s[0])
                    _sent = Utils.detokenize(tokens)
                    if sent in seed_sents or _sent in seed_sents:
                        res[_sent] = s[1]
                    # end if
                # end if
            # end for
            for s in sent_dict[r]['exp']:
                if s[0] is not None:
                    sent = s[0]
                    tokens = Utils.tokenize(s[0])
                    _sent = Utils.detokenize(tokens)
                    if sent in exp_sents or _sent in exp_sents:
                        res[_sent] = s[1]
                    # end if
                # end if
            # end for
        # end for
        return res

    @classmethod
    def read_result_file(cls, result_file):
        line = None
        with open(result_file, "r") as f:
            line = f.read()
        # end with
        return line

    @classmethod
    def get_requirement_from_string(cls, model_result_str, task):
        req_search = re.search(f"Running {task}::([A-Z]+)::(.*)", model_result_str)
        if req_search:
            sent_type = req_search.group(1).strip()
            lc = req_search.group(2).strip()
            return sent_type, lc
        # end if
        return None, None

    @classmethod
    def get_pass_sents_from_model_string(cls, model_result_str, seed_sents, exp_sents):
        result = list()
        for l in model_result_str.splitlines():
            sent_search = re.search(r"DATA::PASS::(\d*\.?\d* \d*\.?\d* \d*\.?\d*)::(\d)::(\d?|None?)::(.*)", l)
            if sent_search:
                sent = sent_search.group(4)
                tokens = Utils.tokenize(sent)
                _sent = Utils.detokenize(tokens)
                if sent in seed_sents or \
                   _sent in seed_sents or \
                   sent in exp_sents or \
                   _sent in exp_sents:
                    result.append({
                        'pred': cls.SENTIMENT_MAP_FROM_SCORE[str(sent_search.group(2))],
                        'label': cls.SENTIMENT_MAP_FROM_SCORE[str(sent_search.group(3))],
                        'sent': _sent,
                    })
                # end if
            # end if
        # end for
        return result

    @classmethod
    def get_fail_sents_from_model_string(cls, model_result_str, seed_sents, exp_sents):
        result = list()
        for l in model_result_str.splitlines():
            sent_search = re.search(r"DATA::FAIL::(\d*\.?\d* \d*\.?\d* \d*\.?\d*)::(\d)::(\d?|None?)::(.*)", l)
            if sent_search:
                sent = sent_search.group(4)
                tokens = Utils.tokenize(sent)
                _sent = Utils.detokenize(tokens)
                if sent in seed_sents or \
                   _sent in seed_sents or \
                   sent in exp_sents or \
                   _sent in exp_sents:
                    result.append({
                        'conf': sent_search.group(1),
                        'pred': cls.SENTIMENT_MAP_FROM_SCORE[str(sent_search.group(2))],
                        'label': cls.SENTIMENT_MAP_FROM_SCORE[str(sent_search.group(3))],
                        'sent': _sent,
                    })
                # end if
            # end if
        # end for
        return result
    
    @classmethod
    def get_ours_results_per_requirement_from_string(cls, result_str, task, model_name, seed_sents, exp_sents):
        pattern = f">>>>> MODEL: {model_name}\n(.*?)?\n<<<<< MODEL: {model_name}"
        # end if
        p = re.compile(pattern, re.DOTALL)
        model_results = p.findall(result_str)
        sents = {'pass': list(), 'fail': list()}
        for r_i, r in enumerate(model_results):
            # sent_type, lc = cls.get_requirement_from_string(r, task)
            sents['pass'].extend(cls.get_pass_sents_from_model_string(r, seed_sents, exp_sents))
            sents['fail'].extend(cls.get_fail_sents_from_model_string(r, seed_sents, exp_sents))
        # end for
        return sents
    
    @classmethod
    def get_predict_results(cls,
                            nlp_task,
                            search_dataset_name,
                            selection_method,
                            model_name,
                            seed_results,
                            exp_results):
        pred_res_file = Macros.result_dir / f"test_results_{nlp_task}_{search_dataset_name}_{selection_method}" / "test_results.txt"
        seed_sents = list(seed_results.keys())
        exp_sents = list(exp_results.keys())
        result_str = cls.read_result_file(pred_res_file)
        model_res_per_reqs = cls.get_ours_results_per_requirement_from_string(result_str,
                                                                              nlp_task,
                                                                              model_name,
                                                                              seed_sents,
                                                                              exp_sents)
        return model_res_per_reqs
    
    @classmethod
    def get_label_inconsistency(cls,
                                tgt_results,
                                seed_results,
                                exp_results):
        # Label inconsistency: # of l{i}_ours != l{i}_human
        num_seed_corr, num_seed_incorr = 0, 0
        num_exp_corr, num_exp_incorr = 0, 0
        seed_sents = list(seed_results.keys())
        exp_sents = list(exp_results.keys())
        for s_i, sent in enumerate(tgt_results.keys()):
            label = tgt_results[sent]
            if sent in seed_sents:
                label_h = seed_results[sent]
                is_same_label = [l for l in label if str(l)==str(label_h)]
                if any(is_same_label):
                    num_seed_corr += 1
                else:
                    num_seed_incorr += 1
                # end if
            elif sent in exp_sents:
                label_h = exp_results[sent]
                is_same_label = [l for l in label if str(l)==str(label_h)]
                if any(is_same_label):
                    num_exp_corr += 1
                else:
                    num_exp_incorr += 1
                # end if
            # end if
        # end for
        return num_seed_incorr, num_exp_incorr


    @classmethod
    def get_reported_bugs(cls,
                          pred_results,
                          tgt_results,
                          seed_results,
                          exp_results):
        # Reported bugs (Approach): # of l{i}_ours != l{i}_model
        seed_sents = list(seed_results.keys())
        exp_sents = list(exp_results.keys())
        seed_rep_bugs = len([s for s in pred_results['fail'] if s['sent'] in seed_sents])
        exp_rep_bugs = len([s for s in pred_results['fail'] if s['sent'] in exp_sents])
        return seed_rep_bugs, exp_rep_bugs
    
    @classmethod
    def get_incorrect_inputs(cls,
                             pred_results,
                             seed_results,
                             exp_results):
        # Incorrect input (Ground Truth): # of l{i}_model != l{i}_human
        num_seed_corr, num_seed_incorr = 0, 0
        num_exp_corr, num_exp_incorr = 0, 0
        # seed_sents = list(seed_results.keys())
        # exp_sents = list(exp_results.keys())
        for r in pred_results['pass']:
            sent = r['sent']
            pred = r['pred']
            if sent in seed_results.keys():
                label_h = seed_results[sent]
                is_same_label = [l for l in pred if str(l)==str(label_h)]
                if any(is_same_label):
                    num_seed_corr += 1
                else:
                    num_seed_incorr += 1
                # end if
            elif sent in exp_results.keys():
                label_h = exp_results[sent]
                is_same_label = [l for l in pred if str(l)==str(label_h)]
                if any(is_same_label):
                    num_exp_corr += 1
                else:
                    num_exp_incorr += 1
                # end if
            # end if
        # end for
        
        for r in pred_results['fail']:
            sent = r['sent']
            pred = r['pred']
            if sent in seed_results.keys():
                label_h = seed_results[sent]
                is_same_label = [l for l in pred if str(l)==str(label_h)]
                if any(is_same_label):
                    num_seed_corr += 1
                else:
                    num_seed_incorr += 1
                # end if
            elif sent in exp_results.keys():
                label_h = exp_results[sent]
                is_same_label = [l for l in pred if str(l)==str(label_h)]
                if any(is_same_label):
                    num_exp_corr += 1
                else:
                    num_exp_incorr += 1
                # end if
            # end if            
        # end for
        return num_seed_incorr, num_exp_incorr
    
    @classmethod
    def get_results(cls,
                    nlp_task: str,
                    search_dataset_name: str,
                    selection_method: str,
                    model_name: str,
                    res_dir: Path,
                    target_file: Path,
                    num_samples:int=100):
        # model_name="textattack/bert-base-uncased-SST-2"
        seed_res_files = sorted([
            f for f in os.listdir(str(res_dir))
            if os.path.isfile(os.path.join(str(res_dir), f)) and \
            f.startswith('seed_samples_subject') and \
            f.endswith('.txt')
        ])
        res = dict()
        seed_rep_bugs_subjs = list()
        seed_incorr_inps_subjs = list()
        seed_label_incons_subjs = list()
        exp_rep_bugs_subjs = list()
        exp_incorr_inps_subjs = list()
        exp_label_incons_subjs = list()
        labels = dict()
        for seed_i, seed_res_file in enumerate(seed_res_files):
            subject_i = int(re.search(r"seed_samples_subject(\d+)\.txt", seed_res_file).group(1))
            seed_res_file = res_dir / seed_res_file
            exp_res_file = res_dir / f"exp_samples_subject{subject_i}.txt"
            
            seed_res = cls.read_results(seed_res_file, num_samples=num_samples)
            exp_res = cls.read_results(exp_res_file, num_samples=num_samples)
            tgt_res = cls.get_target_results(target_file, seed_res, exp_results=exp_res)
            pred_res = cls.get_predict_results(nlp_task,
                                               search_dataset_name,
                                               selection_method,
                                               model_name,
                                               seed_res,
                                               exp_res)
            seed_rep_bugs, exp_rep_bugs = cls.get_reported_bugs(
                pred_res, tgt_res, seed_res, exp_res
            )
            seed_incorr_inps, exp_incorr_inps = cls.get_incorrect_inputs(
                pred_res, seed_res, exp_res
            )
            seed_label_incons, exp_label_incons = cls.get_label_inconsistency(
                tgt_res, seed_res, exp_res
            )
            res[f"subject_{subject_i}"] = {
                'seed': {
                    'reported_bugs': seed_rep_bugs,
                    'incorrect_inputs': seed_incorr_inps,
                    'label_inconsistency': seed_label_incons
                },
                'exp': {
                    'reported_bugs': exp_rep_bugs,
                    'incorrect_inputs': exp_incorr_inps,
                    'label_inconsistency': exp_label_incons
                }
            }
            seed_rep_bugs_subjs.append(seed_rep_bugs)
            seed_incorr_inps_subjs.append(seed_incorr_inps)
            seed_label_incons_subjs.append(seed_label_incons)
            exp_rep_bugs_subjs.append(exp_rep_bugs)
            exp_incorr_inps_subjs.append(exp_incorr_inps)
            exp_label_incons_subjs.append(exp_label_incons)
        # end for
        
        res['agg'] = {
            'num_subjects': len(seed_rep_bugs_subjs),
            'seed': {
                'avg_reported_bugs': sum(seed_rep_bugs_subjs)/len(seed_rep_bugs_subjs),
                'avg_incorrect_inputs': sum(seed_incorr_inps_subjs)/len(seed_incorr_inps_subjs),
                'avg_label_inconsistency': sum(seed_label_incons_subjs)/len(seed_label_incons_subjs)
            },
            'exp': {
                'avg_reported_bugs': sum(exp_rep_bugs_subjs)/len(exp_rep_bugs_subjs),
                'avg_incorrect_inputs': sum(exp_incorr_inps_subjs)/len(exp_incorr_inps_subjs),
                'avg_label_inconsistency': sum(exp_label_incons_subjs)/len(exp_label_incons_subjs)
            }
        }
        print(res)
        return res
    
    @classmethod
    def main_sample(cls,
                    nlp_task,
                    search_dataset_name,
                    selection_method):
        target_file = Macros.result_dir / f"cfg_expanded_inputs_{nlp_task}_{search_dataset_name}_{selection_method}.json"
        sent_dict = cls.read_sentences(target_file)
        sample_dict = cls.sample_sents(sent_dict, num_samples=5)
        cls.write_samples(sample_dict)
        return

    @classmethod
    def main_result(cls,
                    nlp_task,
                    search_dataset_name,
                    selection_method,
                    model_name,
                    num_samples):
        target_file = Macros.result_dir / f"cfg_expanded_inputs_{nlp_task}_{search_dataset_name}_{selection_method}.json"
        res_dir = Macros.result_dir / 'human_study'
        # model_name = "textattack/bert-base-uncased-SST-2"
        result = cls.get_results(
            nlp_task,
            search_dataset_name,
            selection_method,
            model_name,
            res_dir,
            target_file,
            num_samples=num_samples
        )
        Utils.write_json(result, res_dir / "human_study_results.json", pretty_format=True)
        return
