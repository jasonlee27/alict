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
    
    CHECKLIST_LC_LIST = ChecklistTestcases.LC_LIST
    # [
    #     'Sentiment-laden words in context',
    #     'neutral words in context',
    #     'used to, but now',
    #     'simple negations: not neutral is still neutral',
    #     'Hard: Negation of positive with neutral stuff in the middle (should be negative)',
    #     'my opinion is what matters',
    #     'Q & A: yes',
    #     'Q & A: yes (neutral)'
    # ]

    OUR_LC_LIST = [
        'Short sentences with neutral adjectives and nouns',
        'Short sentences with sentiment-laden adjectives',
        'Sentiment change over time, present should prevail',
        'Negated neutral should still be neutral',
        'Author sentiment is more important than of others',
        'parsing sentiment in (question, yes) form',
        'Negated positive with neutral content in the middle',
    ]

    OUR_TO_CH_MAP = Retrain.LC_MAP
    
    CH_TO_OUR_MAP = {
        CHECKLIST_LC_LIST[0]: [OUR_LC_LIST[0]],
        CHECKLIST_LC_LIST[1]: [OUR_LC_LIST[1]],
        CHECKLIST_LC_LIST[2]: [OUR_LC_LIST[2]],
        CHECKLIST_LC_LIST[3]: [OUR_LC_LIST[3]],
        CHECKLIST_LC_LIST[4]: [OUR_LC_LIST[4]],
        CHECKLIST_LC_LIST[5]: [OUR_LC_LIST[5]],
        f"{CHECKLIST_LC_LIST[6]}::{CHECKLIST_LC_LIST[7]}": OUR_LC_LIST[6]
    }
    
    @classmethod
    def get_model_results_from_string(cls, result_str, model_name, is_retrained_model=False):
        pattern = f">>>>> MODEL: {model_name}\n(.*?)?\n<<<<< MODEL: {model_name}"
        if is_retrained_model:
            pattern = f">>>>> RETRAINED MODEL: {model_name}\n(.*?)?\n<<<<< RETRAINED MODEL: {model_name}"
        # end if
        p = re.compile(pattern, re.DOTALL)
        print(pattern)
        model_results = [m.strip() for m in p.findall(result_str)]
        print(len(model_results))
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
            model_results_per_reqs.append({
                'lc': lc,
                'sent_type': sent_type,
                'pass': cls.get_pass_sents_from_model_string(r),
                'fail': cls.get_fail_sents_from_model_string(r)
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
            results.append({
                'sent_type': sent_type,
                'lc': lc,
                'pass': cls.get_pass_sents_from_model_string(r),
                'fail': cls.get_fail_sents_from_model_string(r)
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
    def read_result_file_by_lcs(cls, result_str, is_checklist=False, is_retrained_model=False):
        target_lcs = cls.OUR_LC_LIST
        if not is_checklist:
            _lcs = [lc for lc in cls.CHECKLIST_LC_LIST if not lc.startswith('q & a: yes')]
            _lcs.append('::'.join(cls.CHECKLIST_LC_LIST[7:9]))
            target_lcs = _lcs
            del _lcs
        # end if
        model_results = dict()
        for lc in target_lcs:
            _lc = cls.escape_string(lc)
            # pattern = f">>>>> MODEL: {model_name}(.*?)?\n<<<<< MODEL: {model_name}"
            pattern = f">>>>> Retrain\: LC<{_lc}>\+SST2\n(.*)\n<<<<< Retrain\: LC<{_lc}>\+SST2"
            p = re.compile(pattern, re.DOTALL)
            model_results[lc] = [m.strip() for m in p.findall(result_str)][0]
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
        if is_retrained_by_lcs:
            temp_dict = dict()
        # end if
        # result_lcs = cls.read_result_file_by_lcs(result_str, is_checklist=True)
        for r in result_reqs:
            lc = r['lc'].lower()
            if lc in cls.CHECKLIST_LC_LIST and not is_retrained_by_lcs:
                checklist_testsuite_dict[retrained_model_name].append(r)
            elif lc in cls.CHECKLIST_LC_LIST and is_retrained_by_lcs:
                if lc.startswith('q & a: yes'):
                    temp_dict[lc] = r
                else:
                    checklist_testsuite_dict[retrained_model_name].append(r)
                # end if
            # end if
        # end for
        if is_retrained_by_lcs:
            lc = '::'.join(temp_dict.keys())

            checklist_testsuite_dict[retrained_model_name].append({
                'lc': lc,
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
            ours_testsuite_dict[retrained_model_name].append(r)
        # end for
        return ours_testsuite_dict

    @classmethod
    def read_checklist_result_of_model_retrained_on_ours(cls,
                                                         task,
                                                         dataset_name,
                                                         selection_method,
                                                         retrained_model_name,
                                                         is_retrained_by_lcs=True):
        # get results of model retrained on our generated testcases.
        # by reading evaluating the retrained model with checklist testsuite
        _retrained_model_name = retrained_model_name.replace('/', '-')
        _retrained_model_name = f"{task}_{dataset_name}_{selection_method}_{_retrained_model_name}"
        model_result_dir = Macros.retrain_model_dir / task / _retrained_model_name
        # model_result_file = model_result_dir / "eval_results.json"
        model_testsuite_result_file = model_result_dir / "eval_on_testsuite_results_lcs.txt"
        result_str = cls.read_result_file(model_testsuite_result_file)
        if is_retrained_by_lcs:
            result_lcs = cls.read_result_file_by_lcs(result_str, is_checklist=True)
            checklist_testsuite_dict = dict()
            for lc_key in result_lcs.keys():
                res_str = result_lcs[lc_key]
                checklist_testsuite_dict[lc_key] = {
                    retrained_model_name: list()
                }
                result_reqs = cls.get_checklist_results_per_requirement_from_string(
                    res_str, retrained_model_name, is_retrained_model=True
                )
                for r in result_reqs:
                    if r['lc'].lower() in cls.CHECKLIST_LC_LIST:
                        checklist_testsuite_dict[lc_key][retrained_model_name].append(r)
                    # end if
                # end for
            # end for
        else:
            checklist_testsuite_dict = {
                retrained_model_name: list()
            }
            result_reqs = cls.get_checklist_results_per_requirement_from_string(
                result_str, retrained_model_name, is_retrained_model=True
            )
            for r in result_reqs:
                if r['lc'] in cls.CHECKLIST_LC_LIST:
                    checklist_testsuite_dict[retrained_model_name].append(r)
                # end if
            # end for
        # end if
        return checklist_testsuite_dict

    @classmethod
    def read_ours_result_of_model_retrained_on_checklist(cls,
                                                         task,
                                                         retrained_model_name,
                                                         is_retrained_by_lcs=True):
        # get results of model retrained on cehcklist testcases.
        # by reading evaluating the retrained model with our generated testcases
        _retrained_model_name = retrained_model_name.replace('/', '-')
        _retrained_model_name = f"{task}_checklist_{_retrained_model_name}"
        model_result_dir = Macros.retrain_model_dir / task / _retrained_model_name
        # model_result_file = model_result_dir / "eval_results.json"
        model_testsuite_result_file = model_result_dir / "eval_on_testsuite_results_lcs.txt"
        result_str = cls.read_result_file(model_testsuite_result_file)
        if is_retrained_by_lcs:
            result_lcs = cls.read_result_file_by_lcs(result_str, is_checklist=False)
            checklist_testsuite_dict = dict()
            for lc_key in result_lcs.keys():
                res_str = result_lcs[lc_key]
                checklist_testsuite_dict[lc_key] = {
                    retrained_model_name: list()
                }
                result_reqs = cls.get_ours_results_per_requirement_from_string(result_str,
                                                                               task,
                                                                               retrained_model_name,
                                                                               is_retrained_model=True)
                for r in result_reqs:
                    checklist_testsuite_dict[lc_key][retrained_model_name].append(r)
                # end for
            # end for
        else:
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
        # get the fail_to_pass cases for the model
        # before and after retraining on ours testcase.
        result = {model_name: dict()}
        if is_retrained_by_lcs:
            for lc_key in retrained_testsuite_result.keys():
                for r in orig_testsuite_result[model_name]:
                    print(r['lc'], lc_key, cls.OUR_TO_CH_MAP[lc_key])
                orig_r = [
                    r
                    for r in orig_testsuite_result[model_name]
                    if r['lc'] in cls.OUR_TO_CH_MAP[lc_key]
                ][0]

                ret_r = [
                    r
                    for r in retrained_testsuite_result[lc_key][model_name]
                    if r['lc'] in cls.OUR_TO_CH_MAP[lc_key]
                ][0]

                result[model_name][lc_key] = {
                    'fail2pass': list(),
                    'num_fail2pass': -1,
                    'num_fail_orig': -1,
                    'num_pass_orig': -1,
                    'num_fail_retrained': -1,
                    'num_pass_retrained': -1
                }
                
                for f in orig_r['fail']:
                    found = False
                    for p in ret_r['pass']:
                        if f['sent']==p['sent'] and f['label']==p['label'] and not found:
                            result[model_name][lc_key]['fail2pass'].append({
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
                result[model_name][lc_key]['num_fail2pass'] = len(result[model_name][lc_key]['fail2pass'])
                result[model_name][lc_key]['num_fail_orig'] = len(orig_r['fail'])
                result[model_name][lc_key]['num_pass_orig'] = len(orig_r['pass'])
                result[model_name][lc_key]['num_fail_retrained'] = len(ret_r['fail'])
                result[model_name][lc_key]['num_pass_retrained'] = len(ret_r['pass'])
            # end for
        else:
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
        # get the fail_to_pass cases for the model
        # before and after retraining on checklist testsuite.
        result = {model_name: dict()}
        if retrained_by_lcs:
            for lc_key in retrained_testsuite_result.keys():
                orig_r = [
                    r
                    for r in orig_testsuite_result[model_name]
                    if r['lc'].lower()==cls.CH_TO_OUR_MAP[lc_key]
                ][0]
                sent_type = orig_r['sent_type']

                ret_r = [
                    r
                    for r in retrained_testsuite_result[lc_key][model_name]
                    if r['lc']==cls.CH_TO_OUR_MAP[lc_key] and r['sent_type']==sent_type
                ][0]
                
                result[model_name][f"{lc_key}::{sent_type}"] = {
                    'fail2pass': list(),
                    'num_fail2pass': -1
                }

                for f in orig_r['fail']:
                    found = False
                    for p in ret_r['pass']:
                        if f['sent']==p['sent'] and f['label']==p['label'] and not found:
                            result[model_name][f"{lc_key}::{sent_type}"]['fail2pass'].append({
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
                result[model_name][f"{lc_key}::{sent_type}"]['num_fail2pass'] = len(result[model_name][f"{lc_key}::{sent_type}"]['fail2pass'])
                result[model_name][f"{lc_key}::{sent_type}"]['num_fail_orig'] = len(orig_r['fail'])
                result[model_name][f"{lc_key}::{sent_type}"]['num_pass_orig'] = len(orig_r['pass'])
                result[model_name][f"{lc_key}::{sent_type}"]['num_fail_retrained'] = len(ret_r['fail'])
                result[model_name][f"{lc_key}::{sent_type}"]['num_pass_retrained'] = len(ret_r['pass'])
            # end for
        else:
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
                                                                 is_retrained_by_lcs=is_retrained_by_lcs)
        
        fail_to_pass_cases = cls.find_fail_to_pass(orig_checklist_testsuite_result,
                                                   retrained_checklist_testsuite_result,
                                                   retrained_model_name,
                                                   is_retrained_by_lcs=is_retrained_by_lcs)
        # Utils.write_json(fail_to_pass_cases, saveto, pretty_format=True)
        return fail_to_pass_cases

    @classmethod
    def _analyze_ours_n_model_retrained_on_checklist(cls,
                                                     task,
                                                     dataset_name,
                                                     selection_method,
                                                     retrained_model_name,
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
                                                                 retrained_model_name,
                                                                 is_retrained_by_lcs=is_retrained_by_lcs)
        
        fail_to_pass_cases = cls.find_fail_to_pass_in_ours(orig_checklist_testsuite_result,
                                                           retrained_checklist_testsuite_result,
                                                           retrained_model_name)
        # Utils.write_json(fail_to_pass_cases, saveto, pretty_format=True)
        return fail_to_pass_cases

    @classmethod
    def fail2pass_summary(cls,
                          fail2pass_retrained_on_ours,
                          fail2pass_retrained_on_checklist,
                          savedir,
                          is_retrained_by_lcs=True):
        # checklist_lc_for_retrain = [
        #     'Sentiment-laden words in context',
        #     'neutral words in context',
        #     'used to, but now',
        #     'simple negations: not neutral is still neutral',
        #     'Hard: Negation of positive with neutral stuff in the middle (should be negative)',
        #     'my opinion is what matters',
        #     'Q & A: yes',
        #     'Q & A: yes (neutral)',
        # ]

        # our_lc_for_retrain = [
        #     'Short sentences with neutral adjectives and nouns',
        #     'Short sentences with sentiment-laden adjectives',
        #     'Sentiment change over time, present should prevail',
        #     'Negated neutral should still be neutral',
        #     'Author sentiment is more important than of others',
        #     'parsing sentiment in (question, yes) form',
        #     'Negated positive with neutral content in the middle',
        # ]
        
        for model_name in fail2pass_retrained_on_ours.keys():
            result = 'retrained_lc,num_fail2pass,num_fail_orig,num_pass_orig,num_fail_retrained,num_pass_retrained\n'
            res_ours = fail2pass_retrained_on_ours[model_name]
            summary_ours = list()
            for lc_i, lc_key in enumerate(res_ours.keys()):
                if not lc_key.startswith('[') and \
                   not lc_key.endswith(']') and \
                   lc_key in cls.CHECKLIST_LC_LIST:
                    num_fail2pass = res_ours[lc_key]['num_fail2pass']
                    num_fail_orig = res_ours[lc_key]['num_fail_orig']
                    num_pass_orig = res_ours[lc_key]['num_pass_orig']
                    num_fail_retrained = res_ours[lc_key]['num_fail_retrained']
                    num_pass_retrained = res_ours[lc_key]['num_pass_retrained']
                    if lc_i+1==len(res_ours.keys()):
                        summary_ours.append([
                            lc_key.replace(',', ''),
                            num_fail2pass,
                            num_fail_orig,
                            num_pass_orig,
                            num_fail_retrained,
                            num_pass_retrained
                        ])
                    else:
                        summary_ours[-1][1] += num_fail2pass
                        summary_ours[-1][2] += num_fail_orig
                        summary_ours[-1][3] += num_pass_orig
                        summary_ours[-1][4] += num_fail_retrained
                        summary_ours[-1][5] += num_pass_retrained
                    # end if
                elif lc_key.startswith('[') and lc_key.endswith(']'):
                    # convert str to list and check if all elements in list in lcs
                    _lc_key = [lc for lc in eval(lc_key) if lc in cls.CHECKLIST_LC_LIST]
                    if len(eval(lc_key))==len(_lc_key):
                        num_fail2pass = res_ours[lc_key]['num_fail2pass']
                        num_fail_orig = res_ours[lc_key]['num_fail_orig']
                        num_pass_orig = res_ours[lc_key]['num_pass_orig']
                        num_fail_retrained = res_ours[lc_key]['num_fail_retrained']
                        num_pass_retrained = res_ours[lc_key]['num_pass_retrained']
                        if lc_i+1==len(res_ours.keys()):
                            summary_ours.append([
                                lc_key.replace(',', ''),
                                num_fail2pass,
                                num_fail_orig,
                                num_pass_orig,
                                num_fail_retrained,
                                num_pass_retrained
                            ])
                        else:
                            summary_ours[-1][1] += num_fail2pass
                            summary_ours[-1][2] += num_fail_orig
                            summary_ours[-1][3] += num_pass_orig
                            summary_ours[-1][4] += num_fail_retrained
                            summary_ours[-1][5] += num_pass_retrained
                        # end if
                    # end if
                # end if
            # end for
            for summary in summary_ours:
                result += ','.join([str(s) for s in summary])
                result += '\n'
            # end for
            
            res_checklist = fail2pass_retrained_on_checklist[model_name]
            summary_checklist = list()
            seed_lc_descs = list()
            for lc_key in res_checklist.keys():
                lc_desc = lc_key.split('::')[0]
                if lc_desc in our_lc_for_retrain:
                    num_fail2pass = res_checklist[lc_key]['num_fail2pass']
                    num_fail_orig = res_checklist[lc_key]['num_fail_orig']
                    num_pass_orig = res_checklist[lc_key]['num_pass_orig']
                    num_fail_retrained = res_checklist[lc_key]['num_fail_retrained']
                    num_pass_retrained = res_checklist[lc_key]['num_pass_retrained']
                    if lc_key.endswith('::SEED'):
                        seed_lc_descs.append(lc_desc)
                        summary_checklist.append([
                            lc_desc.replace(',', ''),
                            num_fail2pass,
                            num_fail_orig,
                            num_pass_orig,
                            num_fail_retrained,
                            num_pass_retrained
                        ])
                    elif lc_key.endswith('::EXP') and seed_lc_descs[-1]==lc_desc:
                        summary_checklist[-1][1] += num_fail2pass
                        summary_checklist[-1][2] += num_fail_orig
                        summary_checklist[-1][3] += num_pass_orig
                        summary_checklist[-1][4] += num_fail_retrained
                        summary_checklist[-1][5] += num_pass_retrained
                    # end if
                # end if
            # end for
            result += ',,,,,\n'
            for summary in summary_checklist:
                result += ','.join([str(s) for s in summary])
                result += '\n'
            # end for
            _model_name = model_name.replace('/', '_')
            saveto = savedir / f"debug_comparison_{_model_name}.csv"
            Utils.write_txt(result, saveto)
        # end for
        return
    
    @classmethod
    def analyze(cls, task, dataset_name, selection_method, retrained_model_name, is_retrained_by_lcs=True):
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
            is_retrained_by_lcs=is_retrained_by_lcs
        )
        
        # compare the results of "test_results.txt" and
        # the retrained model results (retrained on checklist).
        # test_results.txt: the evaluation results
        # of the pretrained model(without retraining) on our generated testcases.
        # it is genereated from eval_models in run_sa.sh.
        # these models are evaluated on our generated testcases.
        fail2pass_checklist = cls._analyze_checklist_n_model_retrained_on_ours(
            task,
            dataset_name,
            selection_method,
            retrained_model_name,
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
