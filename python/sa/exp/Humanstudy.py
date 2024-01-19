# This script is to sample sentences
# from seed/exp sentences for pilot study
# Incorrect input (Ground Truth): # of l{i}_model != l{i}_human
# Reported bugs (Approach): # of l{i}_ours != l{i}_model


import re, os
# import nltk
# import copy
import random
import numpy

from typing import *
from pathlib import Path

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
    def read_sentences(cls, json_dir: Path, include_label=False):
        template_files = [
            f for f in os.listdir(str(json_dir))
            if f.startswith('cfg_expanded_inputs_') and f.endswith('.json')
        ]
        results = dict()
        for template_file in template_files:
            inp = Utils.read_json(json_dir / template_file)
            lc_desc = inp['requirement']['description']
            # seeds, exps = list(), list()
            seed_dict = dict()
            for seed in inp['inputs'].keys():
                if include_label:
                    label = inp['inputs'][seed]['label']
                    seed_dict[seed] = {
                        'label': label,
                        'exp': [
                            (e[5], label)
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
            results[lc_desc] = seed_dict
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
    def sample_sents_tosem(
        cls, 
        sent_dict: Dict,
        num_files=10
    ):
        sample_results = dict()
        sample_size = 384
        seed_sents_over_lcs = list()
        lcs = list()
        for lc_i, lc in enumerate(sent_dict.keys()):
            seed_sents = [
                (s, lc) for s in sent_dict[lc].keys()
                if any(sent_dict[lc][s]['exp'])
            ]
            seed_sents_over_lcs.extend(seed_sents)
        # end for
        sample_seeds = random.sample(
            seed_sents_over_lcs, 
            sample_size
        )

        num_samples_per_step = len(sample_seeds)//num_files
        sample_exp_sents = list()
        for f_i in range(num_files):
            seed_sents_per_step = sample_seeds[num_samples_per_step*(f_i):num_samples_per_step*(f_i+1)]
            exp_sents_per_step = list()
            for seed_s, lc in seed_sents_per_step:
                exps = sent_dict[lc][seed_s]['exp']
                random.shuffle(exps)
                for e in exps:
                    if e not in sample_exp_sents:
                        exp_sents_per_step.append((e, lc))
                        sample_exp_sents.append(e)
                        break
                    # end if
                # end for
            # end for
            sample_results[f"file{f_i+1}"] = {
                'seed': seed_sents_per_step,
                'exp': exp_sents_per_step
            }
        # end for

        rem_num_samples_per_step = len(sample_seeds)%num_files
        selected_file_for_rem_samples = random.sample(
            list(sample_results.keys()), 
            rem_num_samples_per_step
        )

        for key_i, key in enumerate(selected_file_for_rem_samples):
            seed_sent = sample_seeds[sample_seeds*num_files+key_i]
            sample_results[key]['seed'].append((seed_sent, lc))
            # sample_results[key][lc]['exp'].append(seed_sent)

            exps = sent_dict[lc][seed_sent]['exp']
            random.shuffle(exps)
            for e in exps:
                if e not in sample_results[key][lc]['exp'] and \
                    e not in sample_exp_sents:
                    sample_results[key]['exp'].append((e, lc))
                    sample_exp_sents.append(e)
                    break
                # end if
            # end for
        # end for

        '''
        sample_size_over_lcs = {
            'Short sentences with sentiment-laden adjectives': 43,
            'Short sentences with neutral adjectives and nouns': 15,
            'Sentiment change over time, present should prevail': 43,
            'Negated negative should be positive or neutral': 43,
            'Negated neutral should still be neutral': 24,
            'Negated positive with neutral content in the middle': 43,
            'Negation of negative at the end, should be positive or neutral': 43,
            'Author sentiment is more important than of others': 44,
            'parsing sentiment in (question, yes) form': 43,
            'Parsing sentiment in (question, no) form': 43,
        } # statistically significant sample size calculated: 384

        for lc_i, lc in enumerate(sent_dict.keys()):
            seed_sents = [
                s for s in sent_dict[lc].keys()
                if any(sent_dict[lc][s]['exp'])
            ]
            random.shuffle(seed_sents)
            s_i = 0
            sample_size_per_lc = sample_size_over_lcs[lc]
            sample_seeds = random.sample(
                seed_sents, 
                sample_size_per_lc
            )
            num_samples_per_step = len(sample_seeds)//num_files
            sample_exp_sents = list()
            for f_i in range(num_files):
                seed_sents_per_step = sample_seeds[num_samples_per_step*(f_i):num_samples_per_step*(f_i+1)]
                seed_sents_per_step = [
                    (s, lc) for s in seed_sents_per_step
                ]
                exp_sents_per_step = list()
                for seed_s, _ in seed_sents_per_step:
                    exps = sent_dict[lc][seed_s]['exp']
                    random.shuffle(exps)
                    for e in exps:
                        if e not in sample_exp_sents:
                            exp_sents_per_step.append((e, lc))
                            sample_exp_sents.append(e)
                            break
                        # end if
                    # end for
                # end for
                sample_results[f"file{f_i+1}"][lc] = {
                    'seed': seed_sents_per_step,
                    'exp': exp_sents_per_step
                }
            # end for

            rem_num_samples_per_step = len(sample_seeds)%num_files
            selected_file_for_rem_samples = random.sample(
                list(sample_results.keys()), 
                rem_num_samples_per_step
            )

            for key_i, key in enumerate(selected_file_for_rem_samples):
                seed_sent = sample_seeds[num_samples_per_step*num_files+key_i]
                sample_results[key][lc]['seed'].append((seed_sent, lc))
                # sample_results[key][lc]['exp'].append(seed_sent)

                exps = sent_dict[lc][seed_sent]['exp']
                random.shuffle(exps)
                for e in exps:
                    if e not in sample_results[key][lc]['exp'] and \
                       e not in sample_exp_sents:
                        sample_results[key][lc]['exp'].append((e, lc))
                        sample_exp_sents.append(e)
                        break
                    # end if
                # end for
            # end for
        # end for
        '''
        return sample_results
    
    @classmethod
    def write_samples(cls, sample_dict: Dict, res_dir: Path, num_samples_per_file=50):
        for f_i in sample_dict.keys():
            random.seed(f_i)
            seeds, exps = list(), list()
            seed_res = ""
            exp_res = ""
            for req in sample_dict[f_i].keys():
                seeds.extend(sample_dict[f_i][req]['seed'])
                exps.extend(sample_dict[f_i][req]['exp'])
            # end for

            seed_inds = list(range(len(seeds)))
            ids_out = list()
            if len(seeds)>num_samples_per_file:
                num_samples_out = len(seeds) - num_samples_per_file
                random.shuffle(seed_inds)
                ids_out = seed_inds[:num_samples_out]
            # end if
            for s_i in seed_inds:
                if s_i not in ids_out:
                    s, r = seeds[s_i]
                    e, _r = exps[s_i]
                    seed_res += f"{s} :: {r}\n\n\n"
                    exp_res += f"{e} :: {_r}\n\n\n"
                # end if
            # end for
            Utils.write_txt(seed_res, res_dir / f"seed_samples_raw_{f_i}.txt")
            Utils.write_txt(exp_res, res_dir / f"exp_samples_raw_{f_i}.txt")
            print(f"{f_i}:\nnum_seed_samples: {len(seeds)}\nnum_exp_samples: {len(exps)}")
        # end for
        return

    @classmethod
    def write_samples_tosem(
        cls,
        sample_dict: Dict, 
        res_dir: Path
    ):
        for f_i in sample_dict.keys():
            seeds = sample_dict[f_i]['seed']
            exps = sample_dict[f_i]['exp']
            seed_res = ""
            exp_res = ""

            for s_i, s in enumerate(seeds):
                s, r = seeds[s_i]
                e, _r = exps[s_i]
                seed_res += f"{s} :: {r}\n\n\n"
                exp_res += f"{e} :: {_r}\n\n\n"
            # end for
            Utils.write_txt(seed_res, res_dir / f"seed_samples_raw_{f_i}.txt")
            Utils.write_txt(exp_res, res_dir / f"exp_samples_raw_{f_i}.txt")
            print(f"{f_i}:\nnum_seed_samples: {len(seeds)}\nnum_exp_samples: {len(exps)}")
        # end for

        '''
        for f_i in sample_dict.keys():
            random.seed(f_i)
            seeds, exps = list(), list()
            seed_res = ""
            exp_res = ""
            for lc in sample_dict[f_i].keys():
                seeds.extend(sample_dict[f_i][lc]['seed'])
                exps.extend(sample_dict[f_i][lc]['exp'])
            # end for

            for s_i, s in enumerate(seeds):
                s, r = seeds[s_i]
                e, _r = exps[s_i]
                seed_res += f"{s} :: {r}\n\n\n"
                exp_res += f"{e} :: {_r}\n\n\n"
            # end for
            Utils.write_txt(seed_res, res_dir / f"seed_samples_raw_{f_i}.txt")
            Utils.write_txt(exp_res, res_dir / f"exp_samples_raw_{f_i}.txt")
            print(f"{f_i}:\nnum_seed_samples: {len(seeds)}\nnum_exp_samples: {len(exps)}")
        # end for
        '''
        return 

    @classmethod
    def read_sample_sentences(cls, sample_sent_file: Path):
        res_lines = [
            l.split('::') for l in Utils.read_txt(sample_sent_file) if l.strip()!=''
        ]
        res = list()
        for l in res_lines:
            # tokens = Utils.tokenize(l[0].strip())
            # sent = Utils.detokenize(tokens)
            # res.append(sent)
            res.append(l[0].strip())
        # end for
        return res

    @classmethod
    def read_sample_scores(cls, resp_files: List[Path], sents: List[str]):
        res = dict()
        for resp_i, resp_f in enumerate(resp_files):
            res_lines = Utils.read_txt(resp_f)
            num_sents = 0
            for l_i, l in enumerate(res_lines):
                l_split = l.strip().split()
                sent_score, lc_score = int(l_split[0]), int(l_split[1])
                val_score = None
                if len(l_split)>2:
                    val_score = int(l_split[2])
                # end if
                if sents[l_i] not in res.keys():
                    res[sents[l_i]] = {
                        'sent_score': [sent_score],
                        'lc_score': [lc_score],
                        'val_score': None if val_score is None else [val_score]
                    }
                else:
                    res[sents[l_i]]['sent_score'].append(sent_score)
                    res[sents[l_i]]['lc_score'].append(lc_score)
                    if len(l_split)>2:
                        res[sents[l_i]]['val_score'].append(val_score)
                    # end if
                # end if
            # end for
        # end for
        return res

    @classmethod
    def read_sample_scores_tosem(
        cls, 
        resp_file: Path, 
        seed_sents: List[str],
        exp_sents: List[str]
    ):
        res_seed = dict()
        res_exp = dict()
        
        # {
        #     'attributes': [att.strip() for att in lines[0].split(delimeter)],
        #     'lines': [l.strip().split(delimeter) for l in lines[1:]]
        # }
        # attributes: ,seed_sent_label_relevancy,seed_sent_lc_relevancy,exp_sent_label_relevancy,exp_sent_lc_relevancy,seed_exp_validity
        res_lines = Utils.read_sv(
            resp_file, 
            delimeter=',', 
            is_first_attributes=True
        )

        attributes = res_lines['attributes']
        res_lines = res_lines['lines']
        
        num_sents = 0
        for l_i, l in enumerate(res_lines):
            seed_sent = seed_sents[l_i]
            exp_sent = exp_sents[l_i]

            if res_seed.get(seed_sent, None) is None:
                res_seed[seed_sent] = {
                    'sent_score': [float(l[1])],
                    'lc_score': [float(l[2])],
                }
            else:
                res_seed[seed_sent]['sent_score'].append(float(l[1]))
                res_seed[seed_sent]['lc_score'].append(float(l[2]))
            # end if

            if res_exp.get(exp_sent, None) is None:
                res_exp[exp_sent] = {
                    'sent_score': [float(l[3])],
                    'lc_score': [float(l[4])],
                    'val_score': [float(l[5])]
                }
            else:
                res_exp[exp_sent]['sent_score'].append(float(l[3]))
                res_exp[exp_sent]['lc_score'].append(float(l[4]))
                res_exp[exp_sent]['val_score'].append(float(l[5]))
            # end if
        # end for
        return res_seed, res_exp

    @classmethod
    def get_target_results(cls, seed_cfg_dir, resps, res_dir):
        res = Utils.read_json(res_dir / 'sent_sample_labels.json')
        res_lc = Utils.read_json(res_dir / 'sent_sample_lcs.json')
        if res is not None and res_lc is not None:
            return res, res_lc
        # end if
        
        sent_dict = cls.read_sentences(seed_cfg_dir, include_label=True)
        res, res_lc = dict(), dict()
        seed_sents = list()
        exp_sents = list()
        for f_key in resps.keys():
            seed_sents.extend(list(resps[f_key]['seed'].keys()))
            exp_sents.extend(list(resps[f_key]['exp'].keys()))
        # end for
        # seed_sents = list(seed_human_results.keys())
        # exp_sents = list(exp_human_results.keys())
        for lc in sent_dict.keys():
            for s in sent_dict[lc].keys(): # seed sents
                if any(sent_dict[lc][s]['exp']):
                    sent = s
                    tokens = Utils.tokenize(sent)
                    _sent = Utils.detokenize(tokens)
                    if sent in seed_sents and sent not in res.keys():
                        res[sent] = cls.SENTIMENT_MAP_FROM_STR[str(sent_dict[lc][s]['label'])]
                        # res[sent] = sent_dict[r][s]['label']
                        res_lc[sent] = lc
                    elif _sent in seed_sents and _sent not in res.keys():
                        res[_sent] = cls.SENTIMENT_MAP_FROM_STR[str(sent_dict[lc][s]['label'])]
                        # res[_sent] = sent_dict[r][s]['label']
                        res_lc[_sent] = lc
                    # end if
                    for e in sent_dict[lc][s]['exp']:
                        e_sent = e[0]
                        tokens = Utils.tokenize(sent)
                        _e_sent = Utils.detokenize(tokens)
                        if e_sent in exp_sents and e_sent not in res.keys():
                            res[e_sent] = cls.SENTIMENT_MAP_FROM_STR[str(e[1])]
                            res_lc[e_sent] = lc
                        elif _e_sent in exp_sents and _e_sent not in res.keys():
                            res[_e_sent] = cls.SENTIMENT_MAP_FROM_STR[str(e[1])]
                            res_lc[_e_sent] = lc
                        # end if
                    # end for
                # end if
            # end for
        # end for
        Utils.write_json(res, res_dir / 'sent_sample_labels.json')
        Utils.write_json(res_lc, res_dir / 'sent_sample_lcs.json')
        print(len(seed_sents), len(exp_sents), len(res.keys()), len(res_lc.keys()))
        return res, res_lc

    @classmethod
    def get_target_results_tosem(
        cls, 
        seed_cfg_dir: Path,
        resps: Dict,
        res_dir: Path
    ):
        res = Utils.read_json(res_dir / 'sent_sample_labels.json')
        res_lc = Utils.read_json(res_dir / 'sent_sample_lcs.json')
        if res is not None and res_lc is not None:
            return res, res_lc
        # end if
        
        sent_dict = cls.read_sentences(seed_cfg_dir, include_label=True)
        res, res_lc = dict(), dict()
        seed_sents = list()
        exp_sents = list()
        for f_key in resps.keys():
            seed_sents.extend(list(resps[f_key]['seed'].keys()))
            exp_sents.extend(list(resps[f_key]['exp'].keys()))
        # end for
        # seed_sents = list(seed_human_results.keys())
        # exp_sents = list(exp_human_results.keys())
        for lc in sent_dict.keys():
            for s in sent_dict[lc].keys(): # seed sents
                if any(sent_dict[lc][s]['exp']):
                    sent = s
                    tokens = Utils.tokenize(sent)
                    _sent = Utils.detokenize(tokens)
                    if sent in seed_sents and sent not in res.keys():
                        res[sent] = cls.SENTIMENT_MAP_FROM_STR[str(sent_dict[lc][s]['label'])]
                        # res[sent] = sent_dict[r][s]['label']
                        res_lc[sent] = lc
                    elif _sent in seed_sents and _sent not in res.keys():
                        res[_sent] = cls.SENTIMENT_MAP_FROM_STR[str(sent_dict[lc][s]['label'])]
                        # res[_sent] = sent_dict[r][s]['label']
                        res_lc[_sent] = lc
                    # end if
                    for e in sent_dict[lc][s]['exp']:
                        e_sent = e[0]
                        tokens = Utils.tokenize(sent)
                        _e_sent = Utils.detokenize(tokens)
                        if e_sent in exp_sents and e_sent not in res.keys():
                            res[e_sent] = cls.SENTIMENT_MAP_FROM_STR[str(e[1])]
                            res_lc[e_sent] = lc
                        elif _e_sent in exp_sents and _e_sent not in res.keys():
                            res[_e_sent] = cls.SENTIMENT_MAP_FROM_STR[str(e[1])]
                            res_lc[_e_sent] = lc
                        # end if
                    # end for
                # end if
            # end for
        # end for
        Utils.write_json(res, res_dir / 'sent_sample_labels.json')
        Utils.write_json(res_lc, res_dir / 'sent_sample_lcs.json')
        print(len(seed_sents), len(exp_sents), len(res.keys()), len(res_lc.keys()))
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
        num_seed_data = 0
        num_exp_data = 0
        num_tgt_data = 0
        for s in seed_sents:
            tokens = Utils.tokenize(s)
            _s = Utils.detokenize(tokens)
            num_tgt_data += 1
            if s in tgt_results.keys():
                num_seed_data += 1
                label = tgt_results[s]
                labels_h = seed_human_results[s]['sent_score']
                label_consistency = [1 if l in label else 0 for l in labels_h]
                res['seed'][s] = sum(label_consistency)/len(label_consistency)
            elif _s in tgt_results.keys():
                num_seed_data += 1
                label = tgt_results[_s]
                labels_h = seed_human_results[s]['sent_score']
                label_consistency = [1 if l in label else 0 for l in labels_h]
                res['seed'][s] = sum(label_consistency)/len(label_consistency)
            else:
                print('get_label_consistency: seed: ', s)
            # end if
        # end for

        for s in exp_sents:
            tokens = Utils.tokenize(s)
            _s = Utils.detokenize(tokens)
            num_tgt_data += 1
            if s in tgt_results.keys():
                num_exp_data += 1
                label = tgt_results[s]
                labels_h = exp_human_results[s]['sent_score']
                label_consistency = [1 if l in label else 0 for l in labels_h]
                res['exp'][s] = sum(label_consistency)/len(label_consistency)
            elif _s in tgt_results.keys():
                num_exp_data += 1
                label = tgt_results[_s]
                labels_h = exp_human_results[s]['sent_score']
                label_consistency = [1 if l in label else 0 for l in labels_h]
                res['exp'][s] = sum(label_consistency)/len(label_consistency)
            else:
                print('get_label_consistency: exp: ', s)
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
        num_tgt_data = 0
        num_seed_data = 0
        num_exp_data = 0

        for s in seed_sents:
            tokens = Utils.tokenize(s)
            _s = Utils.detokenize(tokens)
            num_tgt_data += 1
            if s in tgt_lc_results.keys():
                num_seed_data += 1
                lc_scores_h = seed_human_results[s]['lc_score']
                res['seed'][s] = sum(lc_scores_h)/len(lc_scores_h)
            elif _s in tgt_lc_results.keys():
                num_seed_data += 1
                lc_scores_h = seed_human_results[s]['lc_score']
                res['seed'][s] = sum(lc_scores_h)/len(lc_scores_h)
            else:
                print('get_lc_consistency: seed: ', s)
            # end if
        # end for

        for s in exp_sents:
            tokens = Utils.tokenize(s)
            _s = Utils.detokenize(tokens)
            num_tgt_data += 1
            if s in tgt_lc_results.keys():
                num_seed_data += 1
                lc_scores_h = exp_human_results[s]['lc_score']
                res['exp'][s] = sum(lc_scores_h)/len(lc_scores_h)
            elif _s in tgt_lc_results.keys():
                num_seed_data += 1
                lc_scores_h = exp_human_results[s]['lc_score']
                res['exp'][s] = sum(lc_scores_h)/len(lc_scores_h)
            else:
                print('get_lc_consistency: exp: ', s)
            # end if
        # end for
        return res

    @classmethod
    def norm_lc_relevancy(cls, scores, min_val=1, max_val=5):
        return [(s-min_val)/(max_val-min_val) for s in scores]

    @classmethod
    def get_exp_validity(cls,
                         exp_human_results):
        # Label inconsistency: # of l{i}_ours != l{i}_human
        num_seed_corr, num_seed_incorr = 0, 0
        num_exp_corr, num_exp_incorr = 0, 0
        exp_sents = list(exp_human_results.keys())
        res = {
            'exp': dict()
        }
        num_tgt_data = 0
        num_seed_data = 0
        num_exp_data = 0

        for e in exp_sents:
            val_scores = exp_human_results[e]['val_score']
            if val_scores is not None:
                res['exp'][e] = sum(val_scores)/len(val_scores)
            else:
                return
            # end if
        # end for
        return res

    @classmethod
    def norm_exp_validity(cls, scores, min_val=1, max_val=5):
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
                    seed_cfg_dir: Path):
        # model_name="textattack/bert-base-uncased-SST-2"
        seed_sent_files = sorted([
            f for f in os.listdir(str(res_dir))
            if os.path.isfile(os.path.join(str(res_dir), f)) and \
            re.search(r"seed_samples_raw_file(\d+)\.txt", f)
        ])
        res = dict()
        # seed_rep_bugs_subjs = list()
        # seed_incorr_inps_subjs = list()
        # seed_label_incons_subjs = list()
        # exp_rep_bugs_subjs = list()
        # exp_incorr_inps_subjs = list()
        # exp_label_incons_subjs = list()
        # labels = dict()
        resps = dict()
        for seed_f_i, seed_sent_file in enumerate(seed_sent_files):
            file_i = int(re.search(r"^seed_samples_raw_file(\d+)\.txt", seed_sent_file).group(1))
            exp_sent_file =  f"exp_samples_raw_file{file_i}.txt"
            seed_resp_files = sorted([
                res_dir / resp_f for resp_f in os.listdir(str(res_dir))
                if os.path.isfile(os.path.join(str(res_dir), resp_f)) and \
                re.search(f"^seed_samples_raw_file{file_i}_resp(\d+)?\.txt", resp_f)
            ])
            exp_resp_files = sorted([
                res_dir / resp_f for resp_f in os.listdir(str(res_dir))
                if os.path.isfile(os.path.join(str(res_dir), resp_f)) and \
                re.search(f"^exp_samples_raw_file{file_i}_resp(\d+)?\.txt", resp_f)
            ])
            if any(seed_resp_files) and any(exp_resp_files):
                seed_sents = cls.read_sample_sentences(res_dir / seed_sent_file)
                seed_human_res = cls.read_sample_scores(seed_resp_files, seed_sents)
                exp_sents = cls.read_sample_sentences(res_dir / exp_sent_file)
                exp_human_res = cls.read_sample_scores(exp_resp_files, exp_sents)
                resps[file_i] = {
                    'seed': seed_human_res,
                    'exp': exp_human_res
                }
            # end if
        # end for

        tgt_res, tgt_res_lc = cls.get_target_results(seed_cfg_dir,
                                                     resps,
                                                     res_dir)

        for f_i in resps.keys():
            seed_human_res = resps[f_i]['seed']
            exp_human_res = resps[f_i]['exp']
            res[f_i] = {
                'label_scores': cls.get_label_consistency(tgt_res, seed_human_res, exp_human_res),
                'lc_scores': cls.get_lc_relevancy(tgt_res_lc, seed_human_res, exp_human_res),
                'val_scores': cls.get_exp_validity(exp_human_res)
            }
        # end for

        agg_seed_label_scores = list()
        agg_seed_lc_scores = list()
        agg_exp_label_scores = list()
        agg_exp_lc_scores = list()
        agg_exp_val_scores = list()
        
        for f_i in res.keys():
            agg_seed_label_scores.extend(list(res[f_i]['label_scores']['seed'].values()))
            agg_seed_lc_scores.extend(list(res[f_i]['lc_scores']['seed'].values()))
            agg_exp_label_scores.extend(list(res[f_i]['label_scores']['exp'].values()))
            agg_exp_lc_scores.extend(list(res[f_i]['lc_scores']['exp'].values()))
            agg_exp_val_scores.extend(list(res[f_i]['val_scores']['exp'].values()))
        # end for
        agg_seed_lc_scores = cls.norm_lc_relevancy(agg_seed_lc_scores)
        agg_exp_lc_scores = cls.norm_lc_relevancy(agg_exp_lc_scores)
        agg_exp_val_scores = cls.norm_exp_validity(agg_exp_val_scores)
        
        res['agg'] = {
            'seed': {
                'num_sents': len(agg_seed_label_scores),
                'label_scores': agg_seed_label_scores,
                'lc_scores': agg_seed_lc_scores,
                'avg_label_score': float(Utils.avg(agg_seed_label_scores)),
                'med_label_score': float(Utils.median(agg_seed_label_scores)),
                'std_label_score': float(Utils.stdev(agg_seed_label_scores)),
                'avg_lc_score': float(Utils.avg(agg_seed_lc_scores)),
                'med_lc_score': float(Utils.median(agg_seed_lc_scores)),
                'std_lc_score': float(Utils.stdev(agg_seed_lc_scores))
            },
            'exp': {
                'num_sents': len(agg_exp_label_scores),
                'label_scores': agg_exp_label_scores,
                'lc_scores': agg_exp_lc_scores,
                'avg_label_score': float(Utils.avg(agg_exp_label_scores)),
                'med_label_score': float(Utils.median(agg_exp_label_scores)),
                'std_label_score': float(Utils.stdev(agg_exp_label_scores)),
                'avg_lc_score': float(Utils.avg(agg_exp_lc_scores)),
                'med_lc_score': float(Utils.median(agg_exp_lc_scores)),
                'std_lc_score': float(Utils.stdev(agg_exp_lc_scores)),
                'avg_val_score': float(Utils.avg(agg_exp_val_scores)),
                'med_val_score': float(Utils.median(agg_exp_val_scores)),
                'std_val_score': float(Utils.stdev(agg_exp_val_scores))
            }
        }
        return res

    @classmethod
    def get_results_tosem(
        cls,
        nlp_task: str,
        search_dataset_name: str,
        selection_method: str,
        res_dir: Path,
        seed_cfg_dir: Path
    ):
        # model_name="textattack/bert-base-uncased-SST-2"
        score_files = sorted([
            f for f in os.listdir(str(res_dir))
            if os.path.isfile(os.path.join(str(res_dir), f)) and \
            re.search(r"alict_humanstudy_sa_scores_file(\d+)\.csv", f)
        ])
        res = dict()
        # seed_rep_bugs_subjs = list()
        # seed_incorr_inps_subjs = list()
        # seed_label_incons_subjs = list()
        # exp_rep_bugs_subjs = list()
        # exp_incorr_inps_subjs = list()
        # exp_label_incons_subjs = list()
        # labels = dict()
        resps = dict()
        if not any(score_files):
            raise()
        # end if

        for f_i, score_file in enumerate(score_files):
            file_i = int(re.search(r"^alict_humanstudy_sa_scores_file(\d+)\.csv", score_file).group(1))

            sent_dir = res_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}"

            exp_sent_file =  sent_dir / f"exp_samples_raw_file{file_i}.txt"
            seed_sent_file = sent_dir / f"seed_samples_raw_file{file_i}.txt"

            seed_sents = cls.read_sample_sentences(res_dir / seed_sent_file)
            exp_sents = cls.read_sample_sentences(res_dir / exp_sent_file)

            seed_human_res, exp_human_res = cls.read_sample_scores_tosem(
                res_dir / score_file,
                seed_sents,
                exp_sents
            )
            resps[file_i] = {
                'seed': seed_human_res,
                'exp': exp_human_res
            }
        # end for

        tgt_res, tgt_res_lc = cls.get_target_results_tosem(
            seed_cfg_dir,                                         
            resps,
            sent_dir
        )

        for f_i in resps.keys():
            seed_human_res = resps[f_i]['seed']
            exp_human_res = resps[f_i]['exp']
            res[f_i] = {
                'label_scores': cls.get_label_consistency(tgt_res, seed_human_res, exp_human_res),
                'lc_scores': cls.get_lc_relevancy(tgt_res_lc, seed_human_res, exp_human_res),
                'val_scores': cls.get_exp_validity(exp_human_res)
            }
        # end for

        agg_seed_label_scores = list()
        agg_seed_lc_scores = list()
        agg_exp_label_scores = list()
        agg_exp_lc_scores = list()
        agg_exp_val_scores = list()
        
        for f_i in res.keys():
            agg_seed_label_scores.extend(list(res[f_i]['label_scores']['seed'].values()))
            agg_seed_lc_scores.extend(list(res[f_i]['lc_scores']['seed'].values()))
            agg_exp_label_scores.extend(list(res[f_i]['label_scores']['exp'].values()))
            agg_exp_lc_scores.extend(list(res[f_i]['lc_scores']['exp'].values()))
            agg_exp_val_scores.extend(list(res[f_i]['val_scores']['exp'].values()))
        # end for
        agg_seed_lc_scores = cls.norm_lc_relevancy(agg_seed_lc_scores)
        agg_exp_lc_scores = cls.norm_lc_relevancy(agg_exp_lc_scores)
        agg_exp_val_scores = cls.norm_exp_validity(agg_exp_val_scores)
        
        res['agg'] = {
            'seed': {
                'num_sents': len(agg_seed_label_scores),
                'label_scores': agg_seed_label_scores,
                'lc_scores': agg_seed_lc_scores,
                'avg_label_score': float(Utils.avg(agg_seed_label_scores)),
                'med_label_score': float(Utils.median(agg_seed_label_scores)),
                'std_label_score': float(Utils.stdev(agg_seed_label_scores)),
                'avg_lc_score': float(Utils.avg(agg_seed_lc_scores)),
                'med_lc_score': float(Utils.median(agg_seed_lc_scores)),
                'std_lc_score': float(Utils.stdev(agg_seed_lc_scores))
            },
            'exp': {
                'num_sents': len(agg_exp_label_scores),
                'label_scores': agg_exp_label_scores,
                'lc_scores': agg_exp_lc_scores,
                'avg_label_score': float(Utils.avg(agg_exp_label_scores)),
                'med_label_score': float(Utils.median(agg_exp_label_scores)),
                'std_label_score': float(Utils.stdev(agg_exp_label_scores)),
                'avg_lc_score': float(Utils.avg(agg_exp_lc_scores)),
                'med_lc_score': float(Utils.median(agg_exp_lc_scores)),
                'std_lc_score': float(Utils.stdev(agg_exp_lc_scores)),
                'avg_val_score': float(Utils.avg(agg_exp_val_scores)),
                'med_val_score': float(Utils.median(agg_exp_val_scores)),
                'std_val_score': float(Utils.stdev(agg_exp_val_scores))
            }
        }
        return res
    
    @classmethod
    def main_sample(cls,
                    nlp_task,
                    search_dataset_name,
                    selection_method):
        seed_dir = Macros.result_dir / f"templates_{nlp_task}_{search_dataset_name}_{selection_method}"
        res_dir = Macros.result_dir / 'human_study' / f"{nlp_task}_{search_dataset_name}_{selection_method}"
        res_dir.mkdir(parents=True, exist_ok=True)
        sent_dict = cls.read_sentences(seed_dir)
        sample_dict = cls.sample_sents(sent_dict, num_files=3, num_samples=5)
        cls.write_samples(sample_dict, res_dir)
        return

    @classmethod
    def main_result(cls,
                    nlp_task,
                    search_dataset_name,
                    selection_method,
                    model_name):
        seed_cfg_dir = Macros.result_dir / f"templates_{nlp_task}_{search_dataset_name}_{selection_method}"
        res_dir = Macros.result_dir / 'human_study' / f"{nlp_task}_{search_dataset_name}_{selection_method}"
        res_dir.mkdir(parents=True, exist_ok=True)
        # target_file = Macros.result_dir / f"cfg_expanded_inputs_{nlp_task}_{search_dataset_name}_{selection_method}.json"
        # model_name = "textattack/bert-base-uncased-SST-2"
        result = cls.get_results(
            nlp_task,
            search_dataset_name,
            selection_method,
            model_name,
            res_dir,
            seed_cfg_dir,
        )
        print(res_dir / f"human_study_results.json")
        Utils.write_json(result, res_dir / f"human_study_results.json", pretty_format=True)
        return

    @classmethod
    def main_label(cls,
                   nlp_task,
                   search_dataset_name,
                   selection_method):
        seed_dir = Macros.result_dir / f"templates_{nlp_task}_{search_dataset_name}_{selection_method}"
        res_dir = Macros.result_dir / 'human_study' / f"{nlp_task}_{search_dataset_name}_{selection_method}"
        sent_dict = cls.read_sentences(seed_dir, include_label=True)
        seed_sent_files = sorted([
            f for f in os.listdir(str(res_dir))
            if os.path.isfile(os.path.join(str(res_dir), f)) and \
            re.search(r"seed_samples_raw_file(\d+)\.txt", f)
        ])
        for seed_f_i, seed_sent_file in enumerate(seed_sent_files):
            res_str = ''

            inp_sents = [
                l.split('::')[0].strip()
                for l in Utils.read_txt(res_dir / seed_sent_file)
                if l.strip()!=''
            ]
            for lc in sent_dict.keys():
                for s in sent_dict[lc].keys(): # seed sents
                    tokens = Utils.tokenize(s)
                    _s = Utils.detokenize(tokens)
                    label = None
                    if s in inp_sents:
                        label = sent_dict[lc][s]['label']
                    elif _s in inp_sents:
                        label = sent_dict[lc][s]['label']
                    # end if
                    if label is not None:
                        res_str += f"{s} :: {label}\n"
                        label = None
                    # end if
                # end for
            # end for
            Utils.write_txt(res_str, res_dir / f"{seed_sent_file.split('.txt')[0]}_labels.txt")
        # end for

    @classmethod
    def main_sample_tosem(
        cls,
        nlp_task,
        search_dataset_name,
        selection_method
    ):
        seed_dir = Macros.result_dir / f"templates_{nlp_task}_{search_dataset_name}_{selection_method}"
        res_dir = Macros.result_dir / 'human_study_tosem' / f"{nlp_task}_{search_dataset_name}_{selection_method}"
        res_dir.mkdir(parents=True, exist_ok=True)
        sent_dict = cls.read_sentences(seed_dir)
        sample_dict = cls.sample_sents_tosem(sent_dict, num_files=10)
        cls.write_samples_tosem(sample_dict, res_dir)
        return

    @classmethod
    def main_result_tosem(
        cls,
        nlp_task,
        search_dataset_name,
        selection_method
    ):
        seed_cfg_dir = Macros.result_dir / f"templates_{nlp_task}_{search_dataset_name}_{selection_method}"
        res_dir = Macros.result_dir / 'human_study_tosem' # / f"{nlp_task}_{search_dataset_name}_{selection_method}"
        res_dir.mkdir(parents=True, exist_ok=True)
        # target_file = Macros.result_dir / f"cfg_expanded_inputs_{nlp_task}_{search_dataset_name}_{selection_method}.json"
        # model_name = "textattack/bert-base-uncased-SST-2"
        result = cls.get_results_tosem(
            nlp_task,
            search_dataset_name,
            selection_method,
            res_dir,
            seed_cfg_dir,
        )
        saveto = res_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}" / f"human_study_results_tosem.json"
        print(saveto)
        Utils.write_json(
            result, 
            saveto, 
            pretty_format=True
            )
        return




                


# class Mtnlp:

#     MTNLP_MUTATION_RES_DIR = Macros.result_dir / 'mt_nlp_mutation_results'

#     @classmethod
#     def get_sample_label_per_sent(cls, seed_sent, template_dict):
#         for lc in template_dict.keys():
#             for seed in template_dict[lc].keys():
#                 if seed==seed_sent:
#                     return template_dict[lc][seed]['label']
#                 # end if
#             # end for
#         # end for
#         return

#     @classmethod
#     def get_mutated_sents(sent_index, seed_txt_filename):
#         mutated_sent_txt_file = cls.MTNLP_MUTATION_RES_DIR / f"{seed_txt_filename}_seed{sent_index}.txt"
#         return [l for l in Utils.read_txt(mutated_sent_txt_file) if l!='\n']

#     @classmethod
#     def get_sample_labels(cls,
#                           nlp_task,
#                           search_dataset_name,
#                           selection_method,
#                           sample_sent_txt_file):
#         template_dir = Macros.result_dir / f"templates_{nlp_task}_{search_dataset_name}_{selection_method}"
#         template_dict = Humanstudy.read_sentences(template_dir)
#         # sample_sent_txt_filename = os.path.splittxt(os.path.basename(sample_sent_txt_file))
#         sample_sents = [l for l in Utils.read_txt(sample_sent_txt_file) if l!='\n']
#         # mutation_res = list()
#         res = dict()
#         for s_i, s in enumerate(sample_sents):
#             label = cls.get_sample_label_per_sent(s, template_dict)
#             res[s] = label
#             # mutated_sents = cls.get_mutated_sents(s_i, sample_sent_txt_filename)
#             # if any(mutated_sents):
#             #     mutation_res.append(mutated_sents)
#             # # end if
#         # end for
#         return res

#     @classmethod
#     def get_mutations_from_samples(cls, sample_sent_txt_file):
#         sample_sent_txt_filename = os.path.split(os.path.basename(str(sample_sent_txt_file)))
#         sample_sents = [l for l in Utils.read_txt(str(sample_sent_txt_file)) if l!='\n']
#         mutation_res = dict()
#         for s_i, s in enumerate(sample_sents):
#             # label = cls.get_sample_label_per_sent(s, template_dict)
#             # res[s] = label
#             mutated_sents = cls.get_mutated_sents(s_i, sample_sent_txt_filename)
#             if any(mutated_sents):
#                 mutation_resp[s] = mutated_sents
#             else:
#                 mutation_resp[s] = None
#             # end if
#         # end for
#         return mutation_res

#     # @classmethod
#     # def sample_mutations(cls, mutation_res, num_sample):
#     #     mutations_flatten = list()
#     #     for s in mutation_res.keys():
#     #         mutations = mutation_res[s]
#     #         if mutations is not None:
#     #             mutations_flatten.extend(mutations)
#     #         # end if
#     #     # end for
#     #     samples = random.sample(mutations_flatten, min(len(mutations_flatten), num_sample))
#     #     return samples

#     @classmethod
#     def main_mutation(cls, 
#                       nlp_task,
#                       search_dataset_name,
#                       selection_method):
#         sample_sent_txt_file = Macros.result_dir / 'human_study' / f"{nlp_task}_{search_dataset_name}_{selection_method}" / 'seed_samples_raw_file2.txt'
#         sample_labels = cls.get_sample_labels(nlp_task,
#                                               search_dataset_name,
#                                               selection_method,
#                                               sample_sent_txt_file)
#         num_sample = len(sample_labels.keys())
#         mutations = cls.get_mutations_from_samples(sample_sent_txt_file)
#         res = list()
#         for s in mutations.keys():
#             ms = mutations[s]
#             if ms is not None:
#                 ms_label = sample_labels[s]
#                 for _ms in ms:
#                     res.append(f"{_ms} :: {ms_label}\n")
#                 # end for
#             # end if
#         # end for
#         samples = random.sample(res, min(len(res), num_sample))
#         sample_sent_txt_filename = os.path.split(os.path.basename(sample_sent_txt_file))
#         Utils.write_txt('\n'.join(samples), cls.MTNLP_MUTATION_RES_DIR / f"mtnlp_mutation_samples_{sample_sent_txt_filename}.txt")
#         return
