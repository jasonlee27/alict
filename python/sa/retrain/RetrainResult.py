# This script is to analyze the 
# model test results

from typing import *
from pathlib import Path
from scipy.stats import entropy

# from checklist.test_suite import TestSuite as suite

from ..utils.Macros import Macros
from ..utils.Utils import Utils

import os, re


class RetrainResult:

    CHECKLIST_LC_LIST = [
        'Sentiment-laden words in context',
        'neutral words in context',
        '"used to" should reduce',
        'used to, but now',
        'simple negations: not neutral is still neutral',
        'Hard: Negation of positive with neutral stuff in the middle (should be negative)',
        'my opinion is what matters',
        'Q & A: yes',
        'Q & A: yes (neutral)',
        'Q & A: no',
    ]

    @classmethod
    def get_model_results_from_string(cls, result_str, model_name, is_retrained_model=False):
        pattern = f">>>>> MODEL: {model_name}(.*?)?\n<<<<< MODEL: {model_name}"
        if is_retrained_model:
            pattern = f">>>>> RETRAINED MODEL: {model_name}(.*?)?\n<<<<< RETRAINED MODEL: {model_name}"
        # end if
        p = re.compile(pattern, re.DOTALL)
        model_results = [m.strip() for m in p.findall(result_str)]
        return model_results

    @classmethod
    def get_requirement_from_string(cls, model_result_str, task):
        req_search = re.search(f"Running {task}::([A-Z]+)::(.*)", model_result_str)
        if req_search:
            sent_type = req_search.group(1).strip()
            req = req_search.group(2).strip()
            return sent_type, req
        # end if
        return None, None

    @classmethod
    def get_ours_results_per_requirement_from_string(cls, result_str, task, model_name, is_retrained_model=False):
        pattern = f">>>>> MODEL: {model_name}(.*?)?\n<<<<< MODEL: {model_name}"
        if is_retrained_model:
            pattern = f">>>>> RETRAINED MODEL: {model_name}(.*?)?\n<<<<< RETRAINED MODEL: {model_name}"
        # end if
        p = re.compile(pattern, re.DOTALL)
        model_results = p.findall(result_str)
        model_results_per_reqs = list()
        for r_i, r in enumerate(model_results):
            sent_type, req = cls.get_requirement_from_string(r, task)
            model_results_per_reqs.append({
                'req': req,
                'sent_type': sent_type,
                'pass': cls.get_pass_sents_from_model_string(r),
                'fail': cls.get_fail_sents_from_model_string(r)
            })
        # end for
        return model_results_per_reqs
    
    @classmethod
    def get_checklist_results_per_requirement_from_string(cls, result_str, retrained_model_name, is_retrained_model=False, is_by_lcs=True):
        model_result_str = cls.get_model_results_from_string(result_str, retrained_model_name, is_retrained_model=is_retrained_model)
        model_results_per_reqs = list()
        for m in model_result_str[0].split('\n\n\n'):
            pattern = '(.*?)?\nTest cases\:'
            p = re.compile(pattern, re.DOTALL)
            req_search = p.search(m)
            req = req_search.group(1).splitlines()[-1]
            model_results_per_reqs.append({
                'req': req,
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
            sent_type, req = cls.get_requirement_from_string(r, task)
            results.append({
                'sent_type': sent_type,
                'req': req,
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
    def read_result_file_by_lcs(cls, result_str, is_checklist=False):
        target_lcs = cls.OUR_LC_LIST
        if not is_checklist:
            target_lcs = cls.CHECKLIST_LC_LIST
        # end if
        model_results = dict()
        for lc in target_lcs:
            pattern = f">>>>> Retrain: LC<{lc}>+SST2(.*?)?\n<<<<< Retrain: LC<{lc}>+SST2"
            p = re.compile(pattern, re.DOTALL)
            model_results[lc] = [m.strip() for m in p.findall(result_str)]
        # end for
        return model_results

    @classmethod
    def read_orig_checklist_result(cls, task, dataset_name, selection_method, retrained_model_name, is_by_lcs=True):
        # testing original models with checklist generated testcases
        test_result_dir = Macros.result_dir / f"test_results_{task}_{dataset_name}_{selection_method}"
        checklist_orig_result_file = test_result_dir / "test_results_checklist.txt"
        result_str = cls.read_result_file(checklist_orig_result_file)
        result_reqs = cls.get_checklist_results_per_requirement_from_string(result_str, retrained_model_name, is_by_lcs=is_by_lcs)
        checklist_testsuite_dict = {retrained_model_name: list()}
        for r in result_reqs:
            if r['req'] in cls.CHECKLIST_LC_LIST:
                checklist_testsuite_dict[retrained_model_name].append(r)
            # end if
        # end for
        return checklist_testsuite_dict

    @classmethod
    def read_orig_ours_result(cls, task, dataset_name, selection_method, retrained_model_name, is_retrained_by_lcs=True):
        # testing original models with our generated testcases
        test_result_dir = Macros.result_dir / f"test_results_{task}_{dataset_name}_{selection_method}"
        ours_orig_result_file = test_result_dir / "test_results.txt"
        result_str = cls.read_result_file(ours_orig_result_file)
        if is_retrained_by_lcs:
            result_str_lcs = cls.read_result_file_by_lcs(result_str, is_checklist=True)
        # end if
        # model_result_str = cls.parse_model_results(result_str, retrained_model_name, task)
        result_reqs = cls.get_ours_results_per_requirement_from_string(result_str, task, retrained_model_name)
        ours_testsuite_dict = {retrained_model_name: list()}
        for r in result_reqs:
            ours_testsuite_dict[retrained_model_name].append(r)
        # end for
        return ours_testsuite_dict

    @classmethod
    def read_retrained_checklist_result(cls, task, dataset_name, selection_method, retrained_model_name, is_retrained_by_lcs=True):
        # testing retrained models with checklist generated testcases
        _retrained_model_name = retrained_model_name.replace('/', '-')
        _retrained_model_name = f"{task}_{dataset_name}_{selection_method}_{_retrained_model_name}"
        model_result_dir = Macros.retrain_model_dir / task / _retrained_model_name
        # model_result_file = model_result_dir / "eval_results.json"
        model_testsuite_result_file = model_result_dir / "eval_on_testsuite_results_lcs.txt"
        result_str = cls.read_result_file(model_testsuite_result_file)
        if is_retrained_by_lcs:
            result_str_lcs = cls.read_result_file_by_lcs(result_str, is_checklist=True)
        # end if
        result_reqs = cls.get_checklist_results_per_requirement_from_string(result_str, retrained_model_name, is_retrained_model=True)
        checklist_testsuite_dict = {retrained_model_name: list()}
        for r in result_reqs:
            if r['req'] in cls.CHECKLIST_LC_LIST:
                checklist_testsuite_dict[retrained_model_name].append(r)
            # end if
        # end for
        return checklist_testsuite_dict

    @classmethod
    def read_retrained_ours_result(cls, task, retrained_model_name):
        # testing retrained models with our generated testcases
        _retrained_model_name = retrained_model_name.replace('/', '-')
        _retrained_model_name = f"{task}_checklist_{_retrained_model_name}"
        model_result_dir = Macros.retrain_model_dir / task / _retrained_model_name
        # model_result_file = model_result_dir / "eval_results.json"
        model_testsuite_result_file = model_result_dir / "eval_on_testsuite_results_lcs.txt"
        result_str = cls.read_result_file(model_testsuite_result_file)
        # model_result_str = cls.parse_model_results(result_str, retrained_model_name, task, is_retrained_model=True)
        result_reqs = cls.get_ours_results_per_requirement_from_string(result_str, task, retrained_model_name, is_retrained_model=True)
        checklist_testsuite_dict = {retrained_model_name: list()}
        for r in result_reqs:
            checklist_testsuite_dict[retrained_model_name].append(r)
        # end for
        return checklist_testsuite_dict

    @classmethod
    def find_fail_to_pass(cls, orig_testsuite_result, retrained_testsuite_result, model_name):
        result = {model_name: dict()}
        for orig_r in orig_testsuite_result[model_name]:
            req = orig_r['req']
            result[model_name][req] = {
                'fail2pass': list(),
                'num_fail2pass': -1,
                'num_fail_orig': -1,
                'num_pass_orig': -1,
                'num_fail_retrained': -1,
                'num_pass_retrained': -1
            }
            ret_r = [r for r in retrained_testsuite_result[model_name] if r['req']==req][0]
            
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
        return result

    @classmethod
    def find_fail_to_pass_in_ours(cls, orig_testsuite_result, retrained_testsuite_result, model_name):
        result = {model_name: dict()}
        for orig_r in orig_testsuite_result[model_name]:
            req = orig_r['req']
            sent_type = orig_r['sent_type']
            result[model_name][f"{req}::{sent_type}"] = {
                'fail2pass': list(),
                'num_fail2pass': -1
            }
            ret_r = [r for r in retrained_testsuite_result[model_name] if r['req']==req and r['sent_type']==sent_type][0]

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
        return result
    
    @classmethod
    def _analyze_on_checklist(cls, task, dataset_name, selection_method, retrained_model_name):
        checklist_testsuite_dict = None
        # read original checklist results
        orig_checklist_testsuite_result = cls.read_orig_checklist_result(
            task, dataset_name, selection_method, retrained_model_name
        )

        # read checklist results running on retrained model
        retrained_checklist_testsuite_result = cls.read_retrained_checklist_result(
            task, dataset_name, selection_method, retrained_model_name, is_retrained_by_lcs=True
        )
        
        fail_to_pass_cases = cls.find_fail_to_pass(
            orig_checklist_testsuite_result,
            retrained_checklist_testsuite_result,
            retrained_model_name
        )        
        # Utils.write_json(fail_to_pass_cases, saveto, pretty_format=True)
        return fail_to_pass_cases

    @classmethod
    def _analyze_on_ours(cls, task, dataset_name, selection_method, retrained_model_name):
        checklist_testsuite_dict = None
        # read original checklist results
        orig_checklist_testsuite_result = cls.read_orig_ours_result(
            task, dataset_name, selection_method, retrained_model_name
        )

        # read checklist results running on retrained model
        retrained_checklist_testsuite_result = cls.read_retrained_ours_result(
            task, retrained_model_name
        )
        
        fail_to_pass_cases = cls.find_fail_to_pass_in_ours(
            orig_checklist_testsuite_result,
            retrained_checklist_testsuite_result,
            retrained_model_name
        )        
        # Utils.write_json(fail_to_pass_cases, saveto, pretty_format=True)
        return fail_to_pass_cases

    @classmethod
    def fail2pass_summary(cls, fail2pass_ours, fail2pass_checklist, savedir):
        checklist_req_for_retrain = [
            'Sentiment-laden words in context',
            'neutral words in context',
            'used to, but now',
            'simple negations: not neutral is still neutral',
            'Hard: Negation of positive with neutral stuff in the middle (should be negative)',
            'my opinion is what matters',
            'Q & A: yes',
            'Q & A: yes (neutral)',
        ]

        our_req_for_retrain = [
            'Short sentences with neutral adjectives and nouns',
            'Short sentences with sentiment-laden adjectives',
            'Sentiment change over time, present should prevail',
            'Negated neutral should still be neutral',
            'Author sentiment is more important than of others',
            'parsing sentiment in (question, yes) form',
            'Negated positive with neutral content in the middle',
        ]
        
        for model_name in fail2pass_ours.keys():
            result = 'req,num_fail2pass,num_fail_orig,num_pass_orig,num_fail_retrained,num_pass_retrained\n'
            res_ours = fail2pass_ours[model_name]
            summary_ours = list()
            for req in res_ours.keys():
                if req in checklist_req_for_retrain:
                    num_fail2pass = res_ours[req]['num_fail2pass']
                    num_fail_orig = res_ours[req]['num_fail_orig']
                    num_pass_orig = res_ours[req]['num_pass_orig']
                    num_fail_retrained = res_ours[req]['num_fail_retrained']
                    num_pass_retrained = res_ours[req]['num_pass_retrained']
                    if req != checklist_req_for_retrain[-1]:
                        summary_ours.append([
                            req.replace(',', ''),
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
            # end for
            for summary in summary_ours:
                result += ','.join([str(s) for s in summary])
                result += '\n'
            # end for
            
            res_checklist = fail2pass_checklist[model_name]
            summary_checklist = list()
            seed_req_descs = list()
            for req in res_checklist.keys():
                req_desc = req.split('::')[0]
                if req_desc in our_req_for_retrain:
                    num_fail2pass = res_checklist[req]['num_fail2pass']
                    num_fail_orig = res_checklist[req]['num_fail_orig']
                    num_pass_orig = res_checklist[req]['num_pass_orig']
                    num_fail_retrained = res_checklist[req]['num_fail_retrained']
                    num_pass_retrained = res_checklist[req]['num_pass_retrained']
                    if req.endswith('::SEED'):
                        seed_req_descs.append(req_desc)
                        summary_checklist.append([
                            req_desc.replace(',', ''),
                            num_fail2pass,
                            num_fail_orig,
                            num_pass_orig,
                            num_fail_retrained,
                            num_pass_retrained
                        ])
                    elif req.endswith('::EXP') and seed_req_descs[-1]==req_desc:
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
    def analyze(cls, task, dataset_name, selection_method, retrained_model_name):
        model_name = retrained_model_name.replace('/', '-')
        _retrained_model_name = f"{task}_{dataset_name}_{selection_method}_{model_name}"
        _retrained_checklist_name = f"{task}_checklist_{model_name}"

        # parse test_restults_checklist.txt
        fail2pass_ours = cls._analyze_on_checklist(task, dataset_name, selection_method, retrained_model_name)
        fail2pass_checklist = cls._analyze_on_ours(task, dataset_name, selection_method, retrained_model_name)
        cls.fail2pass_summary(fail2pass_ours, fail2pass_checklist, Macros.retrain_output_dir)
        Utils.write_json(fail2pass_ours,
                         Macros.retrain_model_dir / task / _retrained_model_name / "debug_result.json",
                         pretty_format=True
        )
        Utils.write_json(fail2pass_checklist,
                         Macros.retrain_model_dir / task / _retrained_checklist_name / "debug_result.json",
                         pretty_format=True
        )
