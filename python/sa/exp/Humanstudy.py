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
            # seeds, exps = list(), list()
            seed_dict = dict()
            for seed in inp['inputs'].keys():
                if include_label:
                    label = inp['inputs'][seed]['label']
                    seed_dict[seed] = {
                        'label': label,
                        'exp': [
                            (e[5], cls.SENTIMENT_MAP_FROM_STR[str(label)])
                            for e in inp['inputs'][seed]['exp_inputs'] if e[5] is not None
                        ]
                    }
                else:
                    seed_dict[seed] = {
                        'exp': [
                            e[5]
                            for e in inp['inputs'][seed]['exp_inputs'] if e[5] is not None
                        ]
                    }
                # end if
            # end for
            results[req] = seed_dict
        # end for
        return results
        
    @classmethod
    def sample_sents(cls, sent_dict: Dict, num_files=3, num_samples=5):
        sample_results = dict()
        for f_i in range(num_files):
            sample_results[f"file{f_i+1}"] = dict()
        # end for
        
        for req in sent_dict.keys():
            seed_sents = list(sent_dict[req].keys())
            random.shuffle(seed_sents)
            s_i = 0
            sample_seed_sents = list()
            sample_exp_sents = list()
            for f_i in range(num_files):
                seed_sents_per_step = list()
                exp_sents_per_step = list()
                if s_i<len(seed_sents):
                    for _s_i, s in enumerate(seed_sents[s_i:]):
                        if len(sent_dict[req][s]['exp'])>0 and \
                           s not in sample_seed_sents:
                            seed_sents_per_step.append((s, req))
                            sample_seed_sents.append(s)
                            s_i = _s_i+1
                        # end if
                        if len(seed_sents_per_step)==num_samples:
                            break
                        # end if
                    # end for
                    for s, req in seed_sents_per_step:
                        exps = sent_dict[req][s]['exp']
                        random.shuffle(exps)
                        for e in exps:
                            if e not in sample_exp_sents:
                                exp_sents_per_step.append((e, req))
                                sample_exp_sents.append(e)
                                break
                            # end if
                        # end for
                    # end for
                    sample_results[f"file{f_i+1}"][req] = {
                        'seed': seed_sents_per_step,
                        'exp': exp_sents_per_step
                    }
                # end if
            # end for
        # end for
        return sample_results
    
    @classmethod
    def write_samples(cls, sample_dict: Dict, res_dir: Path):
        for f_i in sample_dict.keys():
            seeds, exps = list(), list()
            seed_res = ""
            exp_res = ""
            for req in sample_dict[f_i].keys():
                seeds.extend(sample_dict[f_i][req]['seed'])
                exps.extend(sample_dict[f_i][req]['exp'])
            # end for
            for s, r in seeds:
                seed_res += f"{s} :: {r}\n\n\n"
            # end for
            for e, r in exps:
                exp_res += f"{e} :: {r}\n\n\n"
            # end for
            Utils.write_txt(seed_res, res_dir / f"seed_samples_raw_{f_i}.txt")
            Utils.write_txt(exp_res, res_dir / f"exp_samples_raw_{f_i}.txt")
            print(f"{f_i}:\nnum_seed_samples: {len(seeds)}\nnum_exp_samples: {len(exps)}")
        # end for
        return
    
    @classmethod
    def read_sample_sentences(cls, sample_sent_file: Path, num_samples: int=100):
        res_lines = [
            l.split('::') for l in Utils.read_txt(sample_sent_file) if l.strip()!=''
        ]
        res = list()
        num_sents = 0
        for l in res_lines:
            if num_sents<num_samples:
                tokens = Utils.tokenize(l[0].strip())
                sent = Utils.detokenize(tokens)
                # res[sent] = l[1].strip()
                res.append(sent)
                num_sents += 1
            # end if
        # end for
        return res

    @classmethod
    def read_sample_scores(cls, resp_files: List[Path], sents: List[str], num_samples: int=100):
        res = dict()
        for resp_i, resp_f in enumerate(resp_files):                
            res_lines = Utils.read_txt(resp_f)
            num_sents = 0
            for l_i, l in enumerate(res_lines):
                if num_sents<num_samples:
                    l_split = l.strip().split()
                    sent_score, lc_score = int(l_split[0]), int(l_split[1])
                    if resp_i==0:
                        res[sents[l_i]] = {
                            'sent_score': [sent_score],
                            'lc_score': [lc_score]
                        }
                    else:
                        res[sents[l_i]]['sent_score'].append(sent_score)
                        res[sents[l_i]]['lc_score'].append(lc_score)
                    # end if
                    num_sents += 1
                # end if
            # end for
        # end for
        return res

    @classmethod
    def get_target_results(cls, target_file, seed_human_results, exp_human_results):
        sent_dict = cls.read_sentences(target_file, include_label=True)
        res, res_lc = dict(), dict()
        seed_sents = list(seed_human_results.keys())
        exp_sents = list(exp_human_results.keys())
        for r in sent_dict.keys():
            for s in sent_dict[r].keys(): # seed sents
                if any(sent_dict[r][s]['exp']):
                    sent = s
                    tokens = Utils.tokenize(sent)
                    _sent = Utils.detokenize(tokens)
                    if sent in seed_sents or _sent in seed_sents:
                        res[_sent] = cls.SENTIMENT_MAP_FROM_STR[str(sent_dict[r][s]['label'])]
                        res_lc[_sent] = r
                    # end if
                    for e in sent_dict[r][s]['exp']:
                        sent = e[0]
                        tokens = Utils.tokenize(sent)
                        _sent = Utils.detokenize(tokens)
                        if sent in exp_sents or _sent in exp_sents:
                            res[_sent] = e[1]
                            res_lc[_sent] = r
                        # end if
                    # end for
                # end if
            # end for
        # end for
        return res, res_lc

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
    def get_label_consistency(cls,
                              tgt_results,
                              seed_human_results,
                              exp_human_results):
        # Label inconsistency: # of l{i}_ours != l{i}_human
        num_seed_corr, num_seed_incorr = 0, 0
        num_exp_corr, num_exp_incorr = 0, 0
        seed_sents = list(seed_human_results.keys())
        exp_sents = list(exp_human_results.keys())
        res = {
            'seed': dict(),
            'exp': dict()
        }
        for s_i, sent in enumerate(tgt_results.keys()):
            label = tgt_results[sent]
            if sent in seed_sents:
                labels_h = seed_human_results[sent]['sent_score']
                label_consistency = [1 if l in label else 0 for l in labels_h]
                res['seed'][sent] = sum(label_consistency)/len(label_consistency)
            elif sent in exp_sents:
                labels_h = exp_human_results[sent]['sent_score']
                label_consistency = [1 if l in label else 0 for l in labels_h]
                res['exp'][sent] = sum(label_consistency)/len(label_consistency)
            # end if
        # end for
        return res

    @classmethod
    def get_lc_relevancy(cls,
                         tgt_lc_results,
                         seed_human_results,
                         exp_human_results):
        # Label inconsistency: # of l{i}_ours != l{i}_human
        num_seed_corr, num_seed_incorr = 0, 0
        num_exp_corr, num_exp_incorr = 0, 0
        seed_sents = list(seed_human_results.keys())
        exp_sents = list(exp_human_results.keys())
        res = {
            'seed': dict(),
            'exp': dict()
        }
        for s_i, sent in enumerate(tgt_lc_results.keys()):
            lc = tgt_lc_results[sent]
            if sent in seed_sents:
                lc_scores_h = seed_human_results[sent]['lc_score']
                res['seed'][sent] = sum(lc_scores_h)/len(lc_scores_h)
            elif sent in exp_sents:
                lc_scores_h = exp_human_results[sent]['lc_score']
                res['exp'][sent] = sum(lc_scores_h)/len(lc_scores_h)
            # end if
        # end for
        return res

    @classmethod
    def norm_lc_relevancy(cls, scores, min_val=1, max_val=5):
        return [(s-min_val)/(max_val-min_val) for s in scores]

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
        seed_sent_files = sorted([
            f for f in os.listdir(str(res_dir))
            if os.path.isfile(os.path.join(str(res_dir), f)) and \
            re.search(r"seed_samples_raw_file(\d+)\.txt", f)
        ])
        res = dict()
        seed_rep_bugs_subjs = list()
        seed_incorr_inps_subjs = list()
        seed_label_incons_subjs = list()
        exp_rep_bugs_subjs = list()
        exp_incorr_inps_subjs = list()
        exp_label_incons_subjs = list()
        labels = dict()
        for seed_f_i, seed_sent_file in enumerate(seed_sent_files):
            subject_i = int(re.search(r"seed_samples_raw_file(\d+)\.txt", seed_sent_file).group(1))
            exp_sent_file = res_dir / f"exp_samples_raw_file{subject_i}.txt"
            seed_resp_files = sorted([
                res_dir / resp_f for resp_f in os.listdir(str(res_dir))
                if os.path.isfile(os.path.join(str(res_dir), resp_f)) and \
                re.search(f"seed_samples_raw_file{subject_i}_resp(\d+)\.txt", resp_f)
            ])
            exp_resp_files = sorted([
                res_dir / resp_f for resp_f in os.listdir(str(res_dir))
                if os.path.isfile(os.path.join(str(res_dir), resp_f)) and \
                re.search(f"exp_samples_raw_file{subject_i}_resp(\d+)\.txt", resp_f)
            ])
            
            seed_sents = cls.read_sample_sentences(res_dir / seed_sent_file)
            seed_human_res = cls.read_sample_scores(seed_resp_files, seed_sents)
            
            exp_sents = cls.read_sample_sentences(exp_sent_file)
            exp_human_res = cls.read_sample_scores(exp_resp_files, exp_sents)
            
            tgt_res, tgt_res_lc = cls.get_target_results(target_file,
                                                         seed_human_res,
                                                         exp_human_res)
            # pred_res = cls.get_predict_results(nlp_task,
            #                                    search_dataset_name,
            #                                    selection_method,
            #                                    model_name,
            #                                    seed_res,
            #                                    exp_res)            
            # seed_rep_bugs, exp_rep_bugs = cls.get_reported_bugs(
            #     pred_res, tgt_res, seed_res, exp_res
            # )
            # seed_incorr_inps, exp_incorr_inps = cls.get_incorrect_inputs(
            #     pred_res, seed_res, exp_res
            # )
            res[f"file{subject_i}"] = {
                'label_scores': cls.get_label_consistency(tgt_res, seed_human_res, exp_human_res),
                'lc_scores': cls.get_lc_relevancy(tgt_res_lc, seed_human_res, exp_human_res)
            }
        # end for

        agg_seed_label_scores = list()
        agg_seed_lc_scores = list()
        agg_exp_label_scores = list()
        agg_exp_lc_scores = list()
        
        for f_i in res.keys():
            agg_seed_label_scores.extend(list(res[f_i]['label_scores']['seed'].values()))
            agg_seed_lc_scores.extend(list(res[f_i]['lc_scores']['seed'].values()))
            agg_exp_label_scores.extend(list(res[f_i]['label_scores']['exp'].values()))
            agg_exp_lc_scores.extend(list(res[f_i]['lc_scores']['exp'].values()))
        # end for
        agg_seed_lc_scores = cls.norm_lc_relevancy(agg_seed_lc_scores)
        agg_exp_lc_scores = cls.norm_lc_relevancy(agg_exp_lc_scores)
        
        res['agg'] = {
            'seed': {
                'num_sents': len(agg_seed_label_scores),
                'label_scores': agg_seed_label_scores,
                'lc_scores': agg_seed_lc_scores,
                'avg_label_score': sum(agg_seed_label_scores)/len(agg_seed_label_scores),
                'avg_lc_score': sum(agg_seed_lc_scores)/len(agg_seed_lc_scores),
            },
            'exp': {
                'num_sents': len(agg_exp_label_scores),
                'label_scores': agg_exp_label_scores,
                'lc_scores': agg_exp_lc_scores,
                'avg_label_score': sum(agg_exp_label_scores)/len(agg_exp_label_scores),
                'avg_lc_score': sum(agg_exp_lc_scores)/len(agg_exp_lc_scores),
            }
        }
        print(res['agg'])
        return res
    
    @classmethod
    def main_sample(cls,
                    nlp_task,
                    search_dataset_name,
                    selection_method):
        target_file = Macros.result_dir / f"cfg_expanded_inputs_{nlp_task}_{search_dataset_name}_{selection_method}.json"
        res_dir = Macros.result_dir / 'human_study' / f"{nlp_task}_{search_dataset_name}_{selection_method}"
        res_dir.mkdir(parents=True, exist_ok=True)
        sent_dict = cls.read_sentences(target_file)
        sample_dict = cls.sample_sents(sent_dict, num_files=3, num_samples=5)
        cls.write_samples(sample_dict, res_dir)
        return

    @classmethod
    def main_result(cls,
                    nlp_task,
                    search_dataset_name,
                    selection_method,
                    model_name,
                    num_samples):
        target_file = Macros.result_dir / f"cfg_expanded_inputs_{nlp_task}_{search_dataset_name}_{selection_method}.json"
        res_dir = Macros.result_dir / 'human_study' / f"{nlp_task}_{search_dataset_name}_{selection_method}"
        res_dir.mkdir(parents=True, exist_ok=True)
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
        Utils.write_json(result, res_dir / f"human_study_results.json", pretty_format=True)
        return
