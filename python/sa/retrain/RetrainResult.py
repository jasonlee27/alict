# This script is to analyze the 
# model test results

from typing import *
from pathlib import Path
from scipy.stats import entropy

# from checklist.test_suite import TestSuite as suite

from .Retrain import ChecklistTestcases, Retrain
from ..utils.Macros import Macros
from ..utils.Utils import Utils

import os, re


class RetrainResult:
    
    CHECKLIST_LC_LIST = [
        Macros.CHECKLIST_LC_LIST[0],
        Macros.CHECKLIST_LC_LIST[1],
        Macros.CHECKLIST_LC_LIST[2],
        Macros.CHECKLIST_LC_LIST[4],
        Macros.CHECKLIST_LC_LIST[5],
        Macros.CHECKLIST_LC_LIST[7],
        Macros.CHECKLIST_LC_LIST[8],
        Macros.CHECKLIST_LC_LIST[9]
    ]

    OUR_LC_LIST = [
        Macros.OUR_LC_LIST[0],
        Macros.OUR_LC_LIST[1],
        Macros.OUR_LC_LIST[2],
        Macros.OUR_LC_LIST[4],
        Macros.OUR_LC_LIST[5],
        Macros.OUR_LC_LIST[7],
        Macros.OUR_LC_LIST[8]
    ]

    OUR_TO_CH_MAP = {
        OUR_LC_LIST[0]: CHECKLIST_LC_LIST[0],
        OUR_LC_LIST[1]: CHECKLIST_LC_LIST[1],
        OUR_LC_LIST[2]: CHECKLIST_LC_LIST[2],
        OUR_LC_LIST[3]: CHECKLIST_LC_LIST[3],
        OUR_LC_LIST[4]: CHECKLIST_LC_LIST[4],
        OUR_LC_LIST[5]: CHECKLIST_LC_LIST[5],
        OUR_LC_LIST[6]: [CHECKLIST_LC_LIST[6],
                         CHECKLIST_LC_LIST[7]]
    }
    
    CH_TO_OUR_MAP = {
        CHECKLIST_LC_LIST[0]: OUR_LC_LIST[0],
        CHECKLIST_LC_LIST[1]: OUR_LC_LIST[1],
        CHECKLIST_LC_LIST[2]: OUR_LC_LIST[2],
        CHECKLIST_LC_LIST[3]: OUR_LC_LIST[3],
        CHECKLIST_LC_LIST[4]: OUR_LC_LIST[4],
        CHECKLIST_LC_LIST[5]: OUR_LC_LIST[5],
        str(CHECKLIST_LC_LIST[6:8]): OUR_LC_LIST[6]
    }
    
    # OUR_TO_CH_MAP = {
    #     OUR_LC_LIST[0].lower(): CHECKLIST_LC_LIST[0].lower(),
    #     OUR_LC_LIST[1].lower(): CHECKLIST_LC_LIST[1].lower(),
    #     OUR_LC_LIST[2].lower(): CHECKLIST_LC_LIST[2].lower(),
    #     OUR_LC_LIST[3].lower(): CHECKLIST_LC_LIST[3].lower(),
    #     OUR_LC_LIST[4].lower(): CHECKLIST_LC_LIST[4].lower(),
    #     OUR_LC_LIST[5].lower(): CHECKLIST_LC_LIST[5].lower(),
    #     OUR_LC_LIST[6].lower(): str(CHECKLIST_LC_LIST[6:8]).lower()
    # }
    
    # CH_TO_OUR_MAP = {
    #     CHECKLIST_LC_LIST[0].lower(): OUR_LC_LIST[0].lower(),
    #     CHECKLIST_LC_LIST[1].lower(): OUR_LC_LIST[1].lower(),
    #     CHECKLIST_LC_LIST[2].lower(): OUR_LC_LIST[2].lower(),
    #     CHECKLIST_LC_LIST[3].lower(): OUR_LC_LIST[3].lower(),
    #     CHECKLIST_LC_LIST[4].lower(): OUR_LC_LIST[4].lower(),
    #     CHECKLIST_LC_LIST[5].lower(): OUR_LC_LIST[5].lower(),
    #     str(CHECKLIST_LC_LIST[6:8]).lower(): OUR_LC_LIST[6].lower()
    # }
    
    @classmethod
    def get_model_results_from_string(cls, result_str, model_name, is_retrained_model=False):
        pattern = f">>>>> MODEL: {model_name}\n(.*?)?\n<<<<< MODEL: {model_name}"
        if is_retrained_model:
            pattern = f">>>>> RETRAINED MODEL: {model_name}\n(.*?)?\n<<<<< RETRAINED MODEL: {model_name}"
        # end if
        p = re.compile(pattern, re.DOTALL)
        model_results = [m.strip() for m in p.findall(result_str)]
        return model_results

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
    def get_ours_results_per_requirement_from_string(cls, result_str, task, model_name, is_retrained_model=False):
        pattern = f">>>>> MODEL: {model_name}\n(.*?)?\n<<<<< MODEL: {model_name}"
        if is_retrained_model:
            pattern = f">>>>> RETRAINED MODEL: {model_name}\n(.*?)?\n<<<<< RETRAINED MODEL: {model_name}"
        # end if
        p = re.compile(pattern, re.DOTALL)
        model_results = p.findall(result_str)
        model_results_per_reqs = list()
        for r_i, r in enumerate(model_results):
            sent_type, lc = cls.get_requirement_from_string(r, task)
            pass_sents, fail_sents = list(), list()
            cksum_vals = list()
            for p in cls.get_pass_sents_from_model_string(r):
                cksum = Utils.get_cksum(p['sent']+p['label'])
                pass_sents.append(p)
                if cksum not in cksum_vals:
                    cksum_vals.append(cksum)
                    # pass_sents.append(p)
                # end if
            # end for
            cksum_vals = list()
            for f in cls.get_fail_sents_from_model_string(r):
                cksum = Utils.get_cksum(f['sent']+f['label'])
                fail_sents.append(f)
                if cksum not in cksum_vals:
                    cksum_vals.append(cksum)
                    # fail_sents.append(f)
                # end if
            # end for
            model_results_per_reqs.append({
                'lc': lc,
                'sent_type': sent_type,
                'pass': pass_sents,
                'fail': fail_sents
            })
        # end for
        return model_results_per_reqs
    
    @classmethod
    def get_checklist_results_per_requirement_from_string(cls,
                                                          result_str,
                                                          retrained_model_name,
                                                          is_retrained_model=False):
        model_result_str = cls.get_model_results_from_string(result_str,
                                                             retrained_model_name,
                                                             is_retrained_model=is_retrained_model)
        model_results_per_reqs = list()
        for m in model_result_str[0].split('\n\n\n'):
            pattern = '(.*?)?\nTest cases\:'
            p = re.compile(pattern, re.DOTALL)
            req_search = p.search(m)
            lc = req_search.group(1).splitlines()[-1]
            model_results_per_reqs.append({
                'lc': lc,
                'pass': cls.get_pass_sents_from_model_string(m),
                'fail': cls.get_fail_sents_from_model_string(m)
            })
        # end for
        return model_results_per_reqs
        
    @classmethod
    def get_pass_sents_from_model_string(cls, model_result_str):
        result = list()
        for l in model_result_str.splitlines():
            sent_search = re.search(r"DATA::PASS::(\d*\.?\d* \d*\.?\d* \d*\.?\d*)::(\d)::(\d?|None?)::(.*)", l)
            if sent_search:
                sent = sent_search.group(4)
                tokens = Utils.tokenize(sent)
                sent = Utils.detokenize(tokens)
                result.append({
                    'conf': sent_search.group(1),
                    'pred': sent_search.group(2),
                    'label': sent_search.group(3),
                    'ent': str(round(entropy([float(p) for p in sent_search.group(1).split()], base=2), 5)),
                    'sent': sent,
                    'key': sent.replace(' ', '')
                })
            # end if
        # end for
        return result

    @classmethod
    def get_fail_sents_from_model_string(cls, model_result_str):
        result = list()
        for l in model_result_str.splitlines():
            sent_search = re.search(r"DATA::FAIL::(\d*\.?\d* \d*\.?\d* \d*\.?\d*)::(\d)::(\d?|None?)::(.*)", l)
            if sent_search:
                sent = sent_search.group(4)
                tokens = Utils.tokenize(sent)
                sent = Utils.detokenize(tokens)
                result.append({
                    'conf': sent_search.group(1),
                    'pred': sent_search.group(2),
                    'label': sent_search.group(3),
                    'ent': str(round(entropy([float(p) for p in sent_search.group(1).split()], base=2), 5)),
                    'sent': sent,
                    'key': sent.replace(' ', '')
                })
            # end if
        # end for
        return result

    @classmethod
    def get_task_from_result_str(cls, result_str):
        task_results = re.search(
            f"\*\*\*\*\* TASK: (.*) \*\*\*\*\*",
            result_str.splitlines()[0]
        )
        return task_results.group(1).strip()
            
    @classmethod
    def parse_model_results(cls, result_str, model_name, task, is_retrained_model=False):
        results = list()
        model_results = cls.get_model_results_from_string(
            result_str, model_name, is_retrained_model=is_retrained_model
        )
        for r in model_results:
            sent_type, lc = cls.get_requirement_from_string(r, task)
            pass_sents, fail_sents = list(), list()
            cksum_vals = list()
            for p in cls.get_pass_sents_from_model_string(r):
                cksum = Utils.get_cksum(p['sent']+p['label'])
                if cksum not in cksum_vals:
                    cksum_vals.append(cksum)
                    pass_sents.append(p)
                # end if
            # end for
            cksum_vals = list()
            for f in cls.get_fail_sents_from_model_string(r):
                cksum = Utils.get_cksum(f['sent']+f['label'])
                if cksum not in cksum_vals:
                    cksum_vals.append(cksum)
                    fail_sents.append(f)
                # end if
            # end for
            results.append({
                'sent_type': sent_type,
                'lc': lc,
                'pass': pass_sents,
                'fail': fail_sents
            })
        # end for
        return results

    @classmethod
    def read_result_file(cls, result_file):
        line = None
        with open(result_file, "r") as f:
            line = f.read()
        # end with
        return line

    @classmethod
    def escape_string(cls, str_in):
        string = str_in.translate(
            str.maketrans({
                "-":  r"\-",
                "&":  r"\&",
                ":":  r"\:",
                "(":  r"\(",
                ")":  r"\)",
                "[":  r"\[",
                "]":  r"\]",
                "^":  r"\^",
                "$":  r"\$",
                "*":  r"\*",
                ".":  r"\.",
                ",":  r"\,"
            })
        )
        return string
        
    @classmethod
    def read_result_file_by_lcs(cls, result_str, is_checklist=False):
        target_lcs = cls.OUR_LC_LIST
        if not is_checklist:
            _lcs = [lc for lc in cls.CHECKLIST_LC_LIST if not lc.startswith('Q & A: yes')]
            _lcs.append(str(cls.CHECKLIST_LC_LIST[6:8]))
            target_lcs = _lcs
            del _lcs
        # end if
        model_results = dict()
        for lc in target_lcs:
            _lc = cls.escape_string(lc)
            # pattern = f">>>>> MODEL: {model_name}(.*?)?\n<<<<< MODEL: {model_name}"
            pattern = f">>>>> Retrain\: LC<{_lc}>\n(.*)\n<<<<< Retrain\: LC<{_lc}>"
            p = re.compile(pattern, re.DOTALL)
            res_obtained = [m.strip() for m in p.findall(result_str)]
            if any(res_obtained):
                model_results[lc] = res_obtained[0]
            # end if
        # end for
        return model_results

    @classmethod
    def read_orig_checklist_result(cls,
                                   task,
                                   dataset_name,
                                   selection_method,
                                   retrained_model_name,
                                   is_retrained_by_lcs=True):
        # testing original models with checklist generated testcases
        test_result_dir = Macros.result_dir / f"test_results_{task}_{dataset_name}_{selection_method}"
        checklist_orig_result_file = test_result_dir / "test_results_checklist.txt"
        result_str = cls.read_result_file(checklist_orig_result_file)
        result_reqs = cls.get_checklist_results_per_requirement_from_string(result_str, retrained_model_name)
        checklist_testsuite_dict = {
            retrained_model_name: list()
        }
        temp_dict = dict()
        # result_lcs = cls.read_result_file_by_lcs(result_str, is_checklist=True)
        for r in result_reqs:
            lc = r['lc']
            if lc in cls.CHECKLIST_LC_LIST:
                if lc.startswith('Q & A: yes'):
                    temp_dict[lc] = r
                else:
                    checklist_testsuite_dict[retrained_model_name].append(r)
                # end if
            # end if
        # end for
        if any(temp_dict.keys()):
            # lc = '::'.join(temp_dict.keys())
            lc = str(list(temp_dict.keys()))
            checklist_testsuite_dict[retrained_model_name].append({
                'lc': lc, #.lower(),
                'pass': [s for key in temp_dict.keys() for s in temp_dict[key]['pass']],
                'fail': [s for key in temp_dict.keys() for s in temp_dict[key]['fail']]
            })
        # end if
        return checklist_testsuite_dict

    @classmethod
    def read_orig_ours_result(cls,
                              task,
                              dataset_name,
                              selection_method,
                              retrained_model_name):
        # testing original models with our generated testcases
        test_result_dir = Macros.result_dir / f"test_results_{task}_{dataset_name}_{selection_method}"
        ours_orig_result_file = test_result_dir / "test_results.txt"
        result_str = cls.read_result_file(ours_orig_result_file)
        # model_result_str = cls.parse_model_results(result_str, retrained_model_name, task)
        result_reqs = cls.get_ours_results_per_requirement_from_string(result_str, task, retrained_model_name)
        ours_testsuite_dict = {
            retrained_model_name: list()
        }
        for r in result_reqs:
            lc = r['lc']
            if lc in cls.OUR_LC_LIST:
                ours_testsuite_dict[retrained_model_name].append(r)
            # end if
        # end for
        return ours_testsuite_dict

    @classmethod
    def read_checklist_result_of_model_retrained_on_ours(cls,
                                                         task,
                                                         dataset_name,
                                                         selection_method,
                                                         retrained_model_name,
                                                         epochs,
                                                         is_retrained_by_lcs=True):
        # get results of model retrained on our generated testcases.
        # by reading evaluating the retrained model with checklist testsuite
        _retrained_model_name = retrained_model_name.replace('/', '-')
        if is_retrained_by_lcs:
            _retrained_model_name = f"{task}_{dataset_name}_{selection_method}_epochs{epochs}_lcs_{_retrained_model_name}"
            model_result_dir = Macros.retrain_model_dir / task / _retrained_model_name
            checklist_testsuite_dict = dict()
            cksum_ls = Utils.read_txt(model_result_dir / "cksum_map.txt")
            for cksum_l in cksum_ls:
                cksum_l = cksum_l.strip()
                retrained_lc, cksum_val = cksum_l.split('\t')[0], cksum_l.split('\t')[1]                
                model_testsuite_result_file = model_result_dir / cksum_val / "eval_on_testsuite_results_lcs.txt"
                result_str = cls.read_result_file(model_testsuite_result_file)
                result_lcs = cls.read_result_file_by_lcs(result_str, is_checklist=True)
                res_str = result_lcs[retrained_lc]
                checklist_testsuite_dict[retrained_lc] = {
                    retrained_model_name: list()
                }
                result_reqs = cls.get_checklist_results_per_requirement_from_string(
                    res_str, retrained_model_name, is_retrained_model=True
                )
                temp_dict = dict()
                for r in result_reqs:
                    lc = r['lc']
                    if lc in cls.CHECKLIST_LC_LIST:
                        if lc.startswith('Q & A: yes'):
                            temp_dict[lc] = r
                        else:
                            checklist_testsuite_dict[retrained_lc][retrained_model_name].append(r)
                        # end if
                    # end if
                # end for
                # lc = '::'.join(temp_dict.keys())
                lc = str(list(temp_dict.keys()))
                checklist_testsuite_dict[retrained_lc][retrained_model_name].append({
                    'lc': lc,
                    'pass': [s for key in temp_dict.keys() for s in temp_dict[key]['pass']],
                    'fail': [s for key in temp_dict.keys() for s in temp_dict[key]['fail']]
                })
            # end for
        else:
            _retrained_model_name = f"{task}_{dataset_name}_{selection_method}_epochs{epochs}_all_{_retrained_model_name}"
            model_result_dir = Macros.retrain_model_dir / task / _retrained_model_name
            model_testsuite_result_file = model_result_dir / "eval_on_testsuite_results_lcs.txt"
            result_str = cls.read_result_file(model_testsuite_result_file)
            # model_result_file = model_result_dir / "eval_results.json"
            checklist_testsuite_dict = {
                retrained_model_name: list()
            }
            result_reqs = cls.get_checklist_results_per_requirement_from_string(
                result_str, retrained_model_name, is_retrained_model=True
            )
            temp_dict = dict()
            for r in result_reqs:
                lc = r['lc']
                if lc in cls.CHECKLIST_LC_LIST:
                    if lc.startswith('Q & A: yes'):
                        temp_dict[lc] = r
                    else:
                        checklist_testsuite_dict[lc][retrained_model_name].append(r)
                    # end if
                # end if
            # end for
            # lc = '::'.join(temp_dict.keys())
            lc = str(list(temp_dict.keys()))
            checklist_testsuite_dict[lc][retrained_model_name].append({
                'lc': lc,
                'pass': [s for key in temp_dict.keys() for s in temp_dict[key]['pass']],
                'fail': [s for key in temp_dict.keys() for s in temp_dict[key]['fail']]
            })
        # end if
        return checklist_testsuite_dict

    @classmethod
    def read_ours_result_of_model_retrained_on_checklist(cls,
                                                         task,
                                                         epochs,
                                                         retrained_model_name,
                                                         is_retrained_by_lcs=True):
        # get results of model retrained on cehcklist testcases.
        # by reading evaluating the retrained model with our generated testcases
        _retrained_model_name = retrained_model_name.replace('/', '-')
        if is_retrained_by_lcs:
            _retrained_model_name = f"{task}_checklist_epochs{epochs}_lcs_{_retrained_model_name}"
            model_result_dir = Macros.retrain_model_dir / task / _retrained_model_name

            checklist_testsuite_dict = dict()
            cksum_ls = Utils.read_txt(model_result_dir / "cksum_map.txt")
            for cksum_l in cksum_ls:
                cksum_l = cksum_l.strip()
                retrained_lc, cksum_val = cksum_l.split('\t')[0], cksum_l.split('\t')[1]                
                model_testsuite_result_file = model_result_dir / cksum_val / "eval_on_testsuite_results_lcs.txt"
                result_str = cls.read_result_file(model_testsuite_result_file)
                result_lcs = cls.read_result_file_by_lcs(result_str, is_checklist=False)
                res_str = result_lcs[retrained_lc]
                checklist_testsuite_dict[retrained_lc] = {
                    retrained_model_name: list()
                }
                result_reqs = cls.get_ours_results_per_requirement_from_string(res_str,
                                                                               task,
                                                                               retrained_model_name,
                                                                               is_retrained_model=True)
                for r in result_reqs:
                    checklist_testsuite_dict[retrained_lc][retrained_model_name].append(r)
                # end for
            # end for
        else:
            _retrained_model_name = f"{task}_checklist_epochs{epochs}_all_{_retrained_model_name}"
            model_result_dir = Macros.retrain_model_dir / task / _retrained_model_name
            # model_result_file = model_result_dir / "eval_results.json"
            model_testsuite_result_file = model_result_dir / "eval_on_testsuite_results_lcs.txt"
            result_str = cls.read_result_file(model_testsuite_result_file)
            # model_result_str = cls.parse_model_results(result_str,
            #                                            retrained_model_name,
            #                                            task,
            #                                            is_retrained_model=True)
            result_reqs = cls.get_ours_results_per_requirement_from_string(result_str,
                                                                           task,
                                                                           retrained_model_name,
                                                                           is_retrained_model=True)
            checklist_testsuite_dict = {retrained_model_name: list()}
            for r in result_reqs:
                checklist_testsuite_dict[retrained_model_name].append(r)
            # end for
        # end if
        return checklist_testsuite_dict

    @classmethod
    def find_fail_to_pass(cls,
                          orig_testsuite_result,
                          retrained_testsuite_result,
                          model_name,
                          is_retrained_by_lcs=True):
        result = {model_name: dict()}
        if is_retrained_by_lcs:
            # get the fail_to_pass cases for the model
            # before and after retraining on checklist testcase for each lc.
            keys = sorted(list(set([orig_r['lc'] for orig_r in orig_testsuite_result[model_name]])))
            for lc_key in retrained_testsuite_result.keys():
                result[model_name][str(lc_key)] = dict()
                # ref_lc = cls.CH_TO_OUR_MAP[str(lc_key)]
                
                result_retrain_lc = dict()
                for k in keys:
                    ref_lc = k
                    result_before_retrain_lc = dict()
                    for orig_r in orig_testsuite_result[model_name]:
                        if orig_r['lc']==ref_lc:
                            if orig_r['sent_type']=='SEED':
                                result_before_retrain_lc['seed'] = orig_r
                            elif orig_r['sent_type']=='EXP':
                                result_before_retrain_lc['exp'] = orig_r
                            # end if
                        # end if
                    # end for
                    result_retrain_lc['before'] = result_before_retrain_lc
                    
                    result_after_retrain_lc = dict()
                    for retrain_r in retrained_testsuite_result[str(lc_key)][model_name]:
                        if retrain_r['lc']==ref_lc:
                            if retrain_r['sent_type']=='SEED':
                                result_after_retrain_lc['seed'] = retrain_r
                            elif retrain_r['sent_type']=='EXP':
                                result_after_retrain_lc['exp'] = retrain_r
                            # end if
                        # end if
                    # end for
                    result_retrain_lc['after'] = result_after_retrain_lc

                    lc_seed = f"{ref_lc}::SEED"
                    result[model_name][str(lc_key)][lc_seed] = {
                        'train_data': 'checklist',
                        'test_data': 'ours',
                        'fail2pass': list(),
                    }
                    for f in result_retrain_lc['before']['seed']['fail']:
                        found = False
                        for p in result_retrain_lc['after']['seed']['pass']:
                            if f['sent']==p['sent'] and f['label']==p['label'] and not found:
                                # result[model_name][str(lc_key)] = list()
                                result[model_name][str(lc_key)][lc_seed]['fail2pass'].append({
                                    'sent': f['sent'],
                                    'label': f['label'],
                                    'pred': (f['pred'], p['pred']),
                                    'ent': (f['ent'], p['ent']),
                                    'conf': (f['conf'], p['conf'])
                                })
                                found = True
                            # end if
                        # end for
                    # end for
                    result[model_name][str(lc_key)][lc_seed]['num_fail2pass'] = len(result[model_name][str(lc_key)][lc_seed]['fail2pass'])
                    result[model_name][str(lc_key)][lc_seed]['num_fail_orig'] = len(result_retrain_lc['before']['seed']['fail'])
                    result[model_name][str(lc_key)][lc_seed]['num_pass_orig'] = len(result_retrain_lc['before']['seed']['pass'])
                    result[model_name][str(lc_key)][lc_seed]['num_fail_retrained'] = len(result_retrain_lc['after']['seed']['fail'])
                    result[model_name][str(lc_key)][lc_seed]['num_pass_retrained'] = len(result_retrain_lc['after']['seed']['pass'])

                    if ('exp' in result_retrain_lc['before'].keys()) and \
                       ('exp' in result_retrain_lc['after'].keys()):
                        lc_exp = f"{ref_lc}::EXP"
                        result[model_name][str(lc_key)][lc_exp] = {
                            'train_data': 'checklist',
                            'test_data': 'ours',
                            'fail2pass': list(),
                        }
                        
                        for f in result_retrain_lc['before']['exp']['fail']:
                            found = False
                            for p in result_retrain_lc['after']['exp']['pass']:
                                if f['sent']==p['sent'] and f['label']==p['label'] and not found:
                                    result[model_name][str(lc_key)][lc_exp]['fail2pass'].append({
                                        'sent': f['sent'],
                                        'label': f['label'],
                                        'pred': (f['pred'], p['pred']),
                                        'ent': (f['ent'], p['ent']),
                                        'conf': (f['conf'], p['conf'])
                                    })
                                    found = True
                                # end if
                            # end for
                        # end for
                        result[model_name][str(lc_key)][lc_exp]['num_fail2pass'] = len(result[model_name][str(lc_key)][lc_exp]['fail2pass'])
                        result[model_name][str(lc_key)][lc_exp]['num_fail_orig'] = len(result_retrain_lc['before']['exp']['fail'])
                        result[model_name][str(lc_key)][lc_exp]['num_pass_orig'] = len(result_retrain_lc['before']['exp']['pass'])
                        result[model_name][str(lc_key)][lc_exp]['num_fail_retrained'] = len(result_retrain_lc['after']['exp']['fail'])
                        result[model_name][str(lc_key)][lc_exp]['num_pass_retrained'] = len(result_retrain_lc['after']['exp']['pass'])
                    # end if
                # end for
            # end for
        else:
            # get the fail_to_pass cases for the model
            # before and after retraining on ours testcase.
            for orig_r in orig_testsuite_result[model_name]:
                req = orig_r['lc']
                result[model_name][req] = {
                    'fail2pass': list(),
                    'num_fail2pass': -1,
                    'num_fail_orig': -1,
                    'num_pass_orig': -1,
                    'num_fail_retrained': -1,
                    'num_pass_retrained': -1
                }
                ret_r = [r for r in retrained_testsuite_result[model_name] if r['lc']==req][0]
                for f in orig_r['fail']:
                    found = False
                    for p in ret_r['pass']:
                        if f['sent']==p['sent'] and f['label']==p['label'] and not found:
                            result[model_name][req]['fail2pass'].append({
                                'sent': f['sent'],
                                'label': f['label'],
                                'pred': (f['pred'], p['pred']),
                                'ent': (f['ent'], p['ent']),
                                'conf': (f['conf'], p['conf'])
                            })
                            found = True
                        # end if
                    # end for
                # end for
                result[model_name][req]['num_fail2pass'] = len(result[model_name][req]['fail2pass'])
                result[model_name][req]['num_fail_orig'] = len(orig_r['fail'])
                result[model_name][req]['num_pass_orig'] = len(orig_r['pass'])
                result[model_name][req]['num_fail_retrained'] = len(ret_r['fail'])
                result[model_name][req]['num_pass_retrained'] = len(ret_r['pass'])
            # end for
        # end if
        return result

    @classmethod
    def find_fail_to_pass_in_ours(cls,
                                  orig_testsuite_result,
                                  retrained_testsuite_result,
                                  model_name,
                                  is_retrained_by_lcs=True):
        result = {model_name: dict()}
        if is_retrained_by_lcs:
            # get the fail_to_pass cases for the model
            # before and after retraining on ours for each lc.
            keys = sorted(list(set([orig_r['lc'] for orig_r in orig_testsuite_result[model_name]])))
            for lc_key in retrained_testsuite_result.keys():
                result[model_name][str(lc_key)] = dict()
                # ref_lc = str(cls.OUR_TO_CH_MAP[lc_key])
                
                result_retrain_lc = dict()
                for k in keys:
                    ref_lc = k
                    for orig_r in orig_testsuite_result[model_name]:
                        if orig_r['lc']==ref_lc:
                            result_retrain_lc['before'] = orig_r
                            break
                        # end if
                    # end for
                    
                    for retrain_r in retrained_testsuite_result[str(lc_key)][model_name]:
                        if retrain_r['lc']==ref_lc:
                            result_retrain_lc['after'] = retrain_r
                            break
                        # end if
                    # end for

                    result[model_name][str(lc_key)][ref_lc] = {
                        'train_data': 'ours',
                        'test_data': 'checklist',
                        'fail2pass': list(),
                    }
                    
                    for f in result_retrain_lc['before']['fail']:
                        found = False
                        for p in result_retrain_lc['after']['pass']:
                            if f['sent']==p['sent'] and f['label']==p['label'] and not found:
                                result[model_name][str(lc_key)][ref_lc]['fail2pass'].append({
                                    'sent': f['sent'],
                                    'label': f['label'],
                                    'pred': (f['pred'], p['pred']),
                                    'ent': (f['ent'], p['ent']),
                                    'conf': (f['conf'], p['conf'])
                                })
                                found = True
                            # end if
                        # end for
                    # end for
                    result[model_name][str(lc_key)][ref_lc]['num_fail2pass'] = len(result[model_name][str(lc_key)][ref_lc]['fail2pass'])
                    result[model_name][str(lc_key)][ref_lc]['num_fail_orig'] = len(result_retrain_lc['before']['fail'])
                    result[model_name][str(lc_key)][ref_lc]['num_pass_orig'] = len(result_retrain_lc['before']['pass'])
                    result[model_name][str(lc_key)][ref_lc]['num_fail_retrained'] = len(result_retrain_lc['after']['fail'])
                    result[model_name][str(lc_key)][ref_lc]['num_pass_retrained'] = len(result_retrain_lc['after']['pass'])
                # end for
            # end for
        else:
            # get the fail_to_pass cases for the model
            # before and after retraining on checklist testsuite.
            for orig_r in orig_testsuite_result[model_name]:
                req = orig_r['lc']
                sent_type = orig_r['sent_type']
                result[model_name][f"{req}::{sent_type}"] = {
                    'fail2pass': list(),
                    'num_fail2pass': -1
                }
                ret_r = [r for r in retrained_testsuite_result[model_name] if r['lc']==req and r['sent_type']==sent_type][0]
                
                for f in orig_r['fail']:
                    found = False
                    for p in ret_r['pass']:
                        if f['sent']==p['sent'] and f['label']==p['label'] and not found:
                            result[model_name][f"{req}::{sent_type}"]['fail2pass'].append({
                                'sent': f['sent'],
                                'label': f['label'],
                                'pred': (f['pred'], p['pred']),
                                'ent': (f['ent'], p['ent']),
                                'conf': (f['conf'], p['conf'])
                            })
                            found = True
                        # end if
                    # end for
                # end for
                result[model_name][f"{req}::{sent_type}"]['num_fail2pass'] = len(result[model_name][f"{req}::{sent_type}"]['fail2pass'])
                result[model_name][f"{req}::{sent_type}"]['num_fail_orig'] = len(orig_r['fail'])
                result[model_name][f"{req}::{sent_type}"]['num_pass_orig'] = len(orig_r['pass'])
                result[model_name][f"{req}::{sent_type}"]['num_fail_retrained'] = len(ret_r['fail'])
                result[model_name][f"{req}::{sent_type}"]['num_pass_retrained'] = len(ret_r['pass'])
            # end for
        # end if
        return result
    
    @classmethod
    def _analyze_checklist_n_model_retrained_on_ours(cls,
                                                     task,
                                                     dataset_name,
                                                     selection_method,
                                                     retrained_model_name,
                                                     epochs,
                                                     is_retrained_by_lcs=True):
        checklist_testsuite_dict = None
        # read original checklist results: "test_results_checklist.txt"
        orig_checklist_testsuite_result = \
            cls.read_orig_checklist_result(task,
                                           dataset_name,
                                           selection_method,
                                           retrained_model_name,
                                           is_retrained_by_lcs=is_retrained_by_lcs)
        
        # read checklist results running on retrained model
        retrained_checklist_testsuite_result = \
            cls.read_checklist_result_of_model_retrained_on_ours(task, dataset_name,
                                                                 selection_method,
                                                                 retrained_model_name,
                                                                 epochs,
                                                                 is_retrained_by_lcs=is_retrained_by_lcs)

        if is_retrained_by_lcs:
            fail_to_pass_cases = cls.find_fail_to_pass_in_ours(orig_checklist_testsuite_result,
                                                               retrained_checklist_testsuite_result,
                                                               retrained_model_name,
                                                               is_retrained_by_lcs=is_retrained_by_lcs)
                                                               
        else:
            fail_to_pass_cases = cls.find_fail_to_pass(orig_checklist_testsuite_result,
                                                       retrained_checklist_testsuite_result,
                                                       retrained_model_name,
                                                       is_retrained_by_lcs=is_retrained_by_lcs)
        # end if
        return fail_to_pass_cases

    @classmethod
    def _analyze_ours_n_model_retrained_on_checklist(cls,
                                                     task,
                                                     dataset_name,
                                                     selection_method,
                                                     retrained_model_name,
                                                     epochs,
                                                     is_retrained_by_lcs=True):
        our_testcase_dict = None
        # read original ours testcase results
        orig_our_testcase_result = \
            cls.read_orig_ours_result(task,
                                      dataset_name,
                                      selection_method,
                                      retrained_model_name)

        # read ours testcase results running on model retrained on checklist testsuite
        retrained_our_testcase_result = \
            cls.read_ours_result_of_model_retrained_on_checklist(task,
                                                                 epochs,
                                                                 retrained_model_name,
                                                                 is_retrained_by_lcs=is_retrained_by_lcs)
        if is_retrained_by_lcs:
            # it is becasse each lc key in the fail_to_pass_cases represents
            # checklist lc retrained on and performance on each ours lc
            fail_to_pass_cases = cls.find_fail_to_pass(orig_our_testcase_result,
                                                       retrained_our_testcase_result,
                                                       retrained_model_name,
                                                       is_retrained_by_lcs=is_retrained_by_lcs)
        else:
            # it is becasse each lc key in the fail_to_pass_cases represents
            # performance on each ours lc
            fail_to_pass_cases = cls.find_fail_to_pass_in_ours(orig_our_testcase_result,
                                                               retrained_our_testcase_result,
                                                               retrained_model_name,
                                                               is_retrained_by_lcs=is_retrained_by_lcs)
        # end if
        return fail_to_pass_cases

    @classmethod
    def fail2pass_summary(cls,
                          fail2pass_retrained_on_ours,
                          fail2pass_retrained_on_checklist,
                          savedir,
                          is_retrained_by_lcs=True):
        for model_name in fail2pass_retrained_on_ours.keys():
            if is_retrained_by_lcs:
                result = 'approach,retrained_lc,eval_lc,num_fail2pass,num_fail_orig,num_pass_orig,num_fail_retrained,num_pass_retrained\n'
                # result[model_name][str(lc_key)][ref_lc]['num_fail2pass']
                res_ours = fail2pass_retrained_on_ours[model_name]
                summary_ours = list()
                for retrained_lc_key in res_ours.keys():
                    for eval_lc_key in res_ours[retrained_lc_key].keys():
                        num_fail2pass = res_ours[retrained_lc_key][eval_lc_key]['num_fail2pass']
                        num_fail_orig = res_ours[retrained_lc_key][eval_lc_key]['num_fail_orig']
                        num_pass_orig = res_ours[retrained_lc_key][eval_lc_key]['num_pass_orig']
                        num_fail_retrained = res_ours[retrained_lc_key][eval_lc_key]['num_fail_retrained']
                        num_pass_retrained = res_ours[retrained_lc_key][eval_lc_key]['num_pass_retrained']
                        _eval_lc_key = cls.CH_TO_OUR_MAP[eval_lc_key]
                        summary_ours.append([
                            'Retrain:Ours::Test:Checklist',
                            retrained_lc_key.replace(',', ' '),
                            _eval_lc_key.replace(',', ' '),
                            num_fail2pass,
                            num_fail_orig,
                            num_pass_orig,
                            num_fail_retrained,
                            num_pass_retrained
                        ])
                        # print(summary_ours[-1])
                    # end for
                # end for
                for summary in summary_ours:
                    print(summary)
                    result += ','.join([str(s) for s in summary])
                    result += '\n'
                # end for
                print('~~~~~~~~~~~')
                res_checklist = fail2pass_retrained_on_checklist[model_name]
                summary_checklist = list()
                for retrained_lc_key in res_checklist.keys():
                    # retrained_lc_desc = retrained_lc_key.split('::')[0]
                    for eval_lc_key in res_checklist[retrained_lc_key].keys():
                        eval_lc_desc = eval_lc_key.split('::')[0]
                        if eval_lc_key.endswith('::SEED'):
                            num_fail2pass = res_checklist[retrained_lc_key][eval_lc_key]['num_fail2pass']
                            num_fail_orig = res_checklist[retrained_lc_key][eval_lc_key]['num_fail_orig']
                            num_pass_orig = res_checklist[retrained_lc_key][eval_lc_key]['num_pass_orig']
                            num_fail_retrained = res_checklist[retrained_lc_key][eval_lc_key]['num_fail_retrained']
                            num_pass_retrained = res_checklist[retrained_lc_key][eval_lc_key]['num_pass_retrained']
                            
                            eval_exp_lc_key = f"{eval_lc_desc}::EXP"
                            if eval_exp_lc_key in res_checklist[retrained_lc_key].keys():
                                num_fail2pass += res_checklist[retrained_lc_key][eval_exp_lc_key]['num_fail2pass']
                                num_fail_orig += res_checklist[retrained_lc_key][eval_exp_lc_key]['num_fail_orig']
                                num_pass_orig += res_checklist[retrained_lc_key][eval_exp_lc_key]['num_pass_orig']
                                num_fail_retrained += res_checklist[retrained_lc_key][eval_exp_lc_key]['num_fail_retrained']
                                num_pass_retrained += res_checklist[retrained_lc_key][eval_exp_lc_key]['num_pass_retrained']
                            # end if
                            _retrained_lc_key = cls.CH_TO_OUR_MAP[retrained_lc_key]
                            summary_checklist.append([
                                'Retrain:Checklist::Test:Ours',
                                _retrained_lc_key.replace(',', ' '),
                                eval_lc_desc.replace(',', ' '),
                                num_fail2pass,
                                num_fail_orig,
                                num_pass_orig,
                                num_fail_retrained,
                                num_pass_retrained
                            ])
                        # end if
                    # end for
                # end for
                result += ',,,,,,\n'
                for summary in summary_checklist:
                    print(summary)
                    result += ','.join([str(s) for s in summary])
                    result += '\n'
                # end for
                _model_name = model_name.replace('/', '_')
                saveto = savedir / f"debug_comparison_{_model_name}.csv"
                Utils.write_txt(result, saveto)
            # end if
        # end for
        return
    
    @classmethod
    def analyze(cls,
                task,
                dataset_name,
                selection_method,
                retrained_model_name,
                epochs,
                is_retrained_by_lcs=True):
        model_name = retrained_model_name.replace('/', '-')
        _retrained_model_name = f"{task}_{dataset_name}_{selection_method}_{model_name}"
        _retrained_checklist_name = f"{task}_checklist_{model_name}"

        # compare the results of "test_results_checklist.txt" and
        # the retrained model results (retrained on ours).
        # "test_results_checklist.txt": the evaluation results
        # of the pretrained model(without retraining) on checklist testsuite.
        # it is genereated from eval_models in run_sa.sh.
        # these models are evaluated on checklist testsuite.
        fail2pass_ours = cls._analyze_checklist_n_model_retrained_on_ours(
            task,
            dataset_name,
            selection_method,
            retrained_model_name,
            epochs,
            is_retrained_by_lcs=is_retrained_by_lcs
        )
        
        # compare the results of "test_results.txt" and
        # the retrained model results (retrained on checklist).
        # test_results.txt: the evaluation results
        # of the pretrained model(without retraining) on our generated testcases.
        # it is genereated from eval_models in run_sa.sh.
        # these models are evaluated on our generated testcases.
        fail2pass_checklist = cls._analyze_ours_n_model_retrained_on_checklist(
            task,
            dataset_name,
            selection_method,
            retrained_model_name,
            epochs,
            is_retrained_by_lcs=is_retrained_by_lcs
        )
        cls.fail2pass_summary(fail2pass_ours,
                              fail2pass_checklist,
                              Macros.retrain_output_dir,
                              is_retrained_by_lcs=is_retrained_by_lcs)

        Utils.write_json(fail2pass_ours,
                         Macros.retrain_model_dir / task / _retrained_model_name / "debug_result.json",
                         pretty_format=True
        )
        Utils.write_json(fail2pass_checklist,
                         Macros.retrain_model_dir / task / _retrained_checklist_name / "debug_result.json",
                         pretty_format=True
        )
        return
