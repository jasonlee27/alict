# This script is to analyze the 
# model test results

from typing import *
from pathlib import Path
from scipy.stats import entropy

# from checklist.test_suite import TestSuite as suite

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..testsuite.Search import Hatecheck

import os, re


class Result:

    @classmethod
    def find_lc_desc(cls, short_desc):
        for key, val in Hatecheck.FUNCTIONALITY_MAP.items():
            if short_desc==val:
                return key
            # end if
        # end for
        return

    @classmethod
    def get_model_results_from_string(cls, result_str, model_name):
        p = re.compile(
            f">>>>> MODEL: {model_name}(.*?)?\n<<<<< MODEL: {model_name}",
            re.DOTALL
        )
        model_results = [m.strip() for m in p.findall(result_str)]
        return model_results

    # @classmethod
    # def get_model_checklist_results_from_string(cls, result_str, model_name):
    #     p = re.compile(
    #         f">>>>> MODEL: {model_name}(.*?)?\n<<<<< MODEL: {model_name}",
    #         re.DOTALL
    #     )
    #     model_results_str = [m.strip() for m in p.findall(result_str)]
    #     model_results = list()
    #     for m in model_results_str[0].split('\n\n\n'):
    #         pattern = '(.*?)?\nTest cases\:'
    #         p = re.compile(pattern, re.DOTALL)
    #         req_search = p.search(m)
    #         lc = req_search.group(1).splitlines()[-1]
    #         if lc in Macros.CHECKLIST_LC_LIST:
    #             model_results.append({
    #                 'lc': lc,
    #                 'pass': cls.get_pass_sents_from_model_string(m),
    #                 'fail': cls.get_fail_sents_from_model_string(m)
    #             })
    #         # end if
    #     # end for
    #     return model_results

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
    def get_pass_sents_from_model_string(cls, model_result_str):
        result = list()
        for l in model_result_str.splitlines():
            sent_search = re.search(r"DATA::PASS::(\d*\.?\d*)::(\d)::(\d?|None?)::(.*)", l)
            if sent_search:
                sent = sent_search.group(4)
                tokens = Utils.tokenize(sent)
                sent = Utils.detokenize(tokens)
                pred_prob = float(sent_search.group(1))
                result.append({
                    'conf': sent_search.group(1),
                    'pred': sent_search.group(2),
                    'label': sent_search.group(3),
                    'ent': str(round(entropy([pred_prob, 1-pred_prob], base=2), 5)),
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
            sent_search = re.search(r"DATA::FAIL::(\d*\.?\d*)::(\d)::(\d?|None?)::(.*)", l)
            if sent_search:
                sent = sent_search.group(4)
                tokens = Utils.tokenize(sent)
                sent = Utils.detokenize(tokens)
                pred_prob = float(sent_search.group(1))
                result.append({
                    'conf': sent_search.group(1),
                    'pred': sent_search.group(2),
                    'label': sent_search.group(3),
                    'ent': str(round(entropy([pred_prob, 1-pred_prob], base=2), 5)),
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
    def parse_model_results(cls, result_str, model_name, task):
        results = list()
        model_results = cls.get_model_results_from_string(result_str, model_name)
        for r in model_results:
            sent_type, lc = cls.get_requirement_from_string(r, task)
            lc_desc = cls.find_lc_desc(lc)
            results.append({
                'sent_type': sent_type,
                'req': lc,
                'lc_desc': lc_desc,
                'pass': cls.get_pass_sents_from_model_string(r),
                'fail': cls.get_fail_sents_from_model_string(r)
            })
        # end for
        return results
    
    @classmethod
    def parse_model_on_hatecheck_results(cls, result_str, model_name, task):
        results = list()
        model_results = cls.get_model_results_from_string(result_str, model_name)
        for r in model_results:
            sent_type, lc = cls.get_requirement_from_string(r, task)
            lc_desc = cls.find_lc_desc(lc)
            results.append({
                'req': lc,
                'lc_desc': lc_desc,
                'pass': cls.get_pass_sents_from_model_string(r),
                'fail': cls.get_fail_sents_from_model_string(r)
            })
        # end for
        return results

    @classmethod
    def parse_results(cls, result_file, model_name_file):
        with open(result_file, "r") as f:
            line = f.read()
            task = cls.get_task_from_result_str(line)
            return {
                model.strip(): cls.parse_model_results(
                    line, model.strip(), task
                )
                for model in Utils.read_txt(model_name_file)
            }
        # end with

    @classmethod
    def parse_hatecheck_results(cls, result_file, model_name_file):
        with open(result_file, "r") as f:
            line = f.read()
            task = cls.get_task_from_result_str(line)
            return {
                model.strip(): cls.parse_model_on_hatecheck_results(
                    line, model.strip(), task
                )
                for model in Utils.read_txt(model_name_file)
            }
        # end with

    @classmethod
    def get_seed_to_exp_map(cls, template_file):
        templates = Utils.read_json(template_file)
        seed_exp_map = dict()
        for t in templates:
            seed_exp_map[t['requirement']['description']] = dict()
            for seed in t['inputs'].keys():
                exp_list = list()
                for exp in t['inputs'][seed]['exp_inputs']:
                    if exp[5] is not None:
                        tokens = Utils.tokenize(exp[5])
                        sent = Utils.detokenize(tokens)
                        exp_list.append(sent.replace(' ', ''))
                    # end if
                # end for
                tokens = Utils.tokenize(seed)
                _seed = Utils.detokenize(tokens)
                _seed = _seed.replace(' ', '')
                seed_exp_map[t['requirement']['description']][_seed] = exp_list
            # end for
        # end for
        return seed_exp_map

    @classmethod
    def analyze_model_hatecheck(cls, model_results):
        reqs = sorted(set([r['req'] for r in model_results]))
        results = list()
        for r in model_results:
            result = {
                'req': r['req'],
                'lc_desc': r['lc_desc'],
                'is_exps_exist': False
            }
            tcs_pass = r['pass']
            tcs_fail = r['fail']
            result['num_tcs'] = len(tcs_pass)+len(tcs_fail)
            result['num_tc_fail'] = len(tcs_fail)
            results.append(result)
        # end for
        return results

    @classmethod
    def analyze_model(cls, model_results, seed_exp_map):
        reqs = sorted(set([r['req'] for r in model_results]))
        results = list()
        for r in reqs:
            seeds = [mr for mr in model_results if mr['req']==r and mr['sent_type']=='SEED']
            exps = [mr for mr in model_results if mr['req']==r and mr['sent_type']=='EXP']
            result = {'req': r, 'is_exps_exist': False}
            num_pass2fail, num_fail2pass = 0, 0
            num_pass2pass, num_fail2fail = 0, 0
            num_pass2pass_ent_inc, num_pass2pass_ent_dec, num_pass2pass_ent_same = 0,0,0
            num_fail2fail_ent_inc, num_fail2fail_ent_dec, num_fail2fail_ent_same = 0,0,0
            num_pass2fail_ent_inc, num_pass2fail_ent_dec, num_pass2fail_ent_same = 0,0,0
            num_fail2pass_ent_inc, num_fail2pass_ent_dec, num_fail2pass_ent_same = 0,0,0
            
            seeds_pass = seeds[0]['pass']
            seeds_fail = seeds[0]['fail']
            result['num_seeds'] = len(seeds_pass)+len(seeds_fail)
            result['num_seed_fail'] = len(seeds_fail)
            if any(exps):
                result['is_exps_exist'] = True
                result['pass->fail'] = list()
                result['fail->pass'] = list()
                result['pass->pass'] = list()
                result['fail->fail'] = list()
                
                exps_pass = exps[0]['pass']
                exps_fail = exps[0]['fail']
                for p in seeds_pass:
                    pass2fail_dict = {'from': list(), 'to': list()}
                    pass2pass_dict = {'from': list(), 'to': list()}
                    for exp in seed_exp_map[r][p['key']]:
                        for ef in exps_fail:
                            if exp==ef['key']:
                                pass2fail_dict['to'].append({
                                    'sent': ef['sent'],
                                    'pred': ef['pred'],
                                    'label': ef['label'],
                                    'conf': ef['conf'],
                                    'ent': ef['ent']
                                })
                                num_pass2fail += 1
                                if p['ent']<ef['ent']:
                                    num_pass2fail_ent_inc += 1
                                elif p['ent']>ef['ent']:
                                    num_pass2fail_ent_dec += 1
                                else:
                                    num_pass2fail_ent_same += 1
                                # end if
                            # end if
                        # end for

                        for ep in exps_pass:
                            if exp==ep['key'] and p['ent']!=ep['ent']:
                                pass2pass_dict['to'].append({
                                    'sent': ep['sent'],
                                    'pred': ep['pred'],
                                    'label': ep['label'],
                                    'conf': ep['conf'],
                                    'ent': ep['ent']
                                })
                                num_pass2pass += 1
                                if p['ent']<ep['ent']:
                                    num_pass2pass_ent_inc += 1
                                elif p['ent']>ep['ent']:
                                    num_pass2pass_ent_dec += 1
                                # end if
                            elif exp==ep['key'] and p['ent']==ep['ent']:
                                num_pass2pass_ent_same += 1
                            # end if
                        # end for
                    # end for
                    if any(pass2fail_dict['to']):
                        pass2fail_dict['from'] = {
                            'sent': p['sent'],
                            'pred': p['pred'],
                            'label': p['label'],
                            'conf': p['conf'],
                            'ent': p['ent']
                        }
                        result['pass->fail'].append(pass2fail_dict)
                    # end if

                    if any(pass2pass_dict['to']):
                        pass2pass_dict['from'] = {
                            'sent': p['sent'],
                            'pred': p['pred'],
                            'label': p['label'],
                            'conf': p['conf'],
                            'ent': p['ent']
                        }
                        result['pass->pass'].append(pass2pass_dict)
                    # end if
                # end for
                    
                for f in seeds_fail:
                    fail2pass_dict = {'from': list(), 'to': list()}
                    fail2fail_dict = {'from': list(), 'to': list()}
                    for exp in seed_exp_map[r][f['key']]:
                        for ep in exps_pass:
                            if exp==ep['key']:
                                fail2pass_dict['to'].append({
                                    'sent': ep['sent'],
                                    'pred': ep['pred'],
                                    'label': ep['label'],
                                    'conf': ep['conf'],
                                    'ent': ep['ent']
                                })
                                num_fail2pass += 1
                                if f['ent']<ep['ent']:
                                    num_fail2pass_ent_inc += 1
                                elif f['ent']>ep['ent']:
                                    num_fail2pass_ent_dec += 1
                                else:
                                    num_fail2pass_ent_same += 1
                                # end if
                            # end if
                        # end for

                        for ef in exps_fail:
                            if exp==ef['key'] and f['ent']!=ef['ent']:
                                fail2fail_dict['to'].append({
                                    'sent': ef['sent'],
                                    'pred': ef['pred'],
                                    'label': ef['label'],
                                    'conf': ef['conf'],
                                    'ent': ef['ent']
                                })
                                num_fail2fail += 1
                                if f['ent']<ef['ent']:
                                    num_fail2fail_ent_inc += 1
                                elif f['ent']>ef['ent']:
                                    num_fail2fail_ent_dec += 1
                                # end if
                            elif exp==ef['key'] and f['ent']==ef['ent']:
                                num_fail2fail_ent_same += 1
                            # end if
                        # end for
                    # end for
                    if any(fail2pass_dict['to']):
                        fail2pass_dict['from'] = {
                            'sent': f['sent'],
                            'pred': f['pred'],
                            'label': f['label'],
                            'conf': f['conf'],
                            'ent': f['ent']
                        }
                        result['fail->pass'].append(fail2pass_dict)
                    # end if

                    if any(fail2fail_dict['to']):
                        fail2fail_dict['from'] = {
                            'sent': f['sent'],
                            'pred': f['pred'],
                            'label': f['label'],
                            'conf': f['conf'],
                            'ent': f['ent']
                        }
                        result['fail->fail'].append(fail2fail_dict)
                    # end if
                # end for
                result['num_pass2pass_ent_inc'] = num_pass2pass_ent_inc
                result['num_pass2pass_ent_dec'] = num_pass2pass_ent_dec
                result['num_pass2pass_ent_same'] = num_pass2pass_ent_same
                result['num_fail2fail_ent_inc'] = num_fail2fail_ent_inc
                result['num_fail2fail_ent_dec'] = num_fail2fail_ent_dec
                result['num_fail2fail_ent_same'] = num_fail2fail_ent_same
                if result['is_exps_exist']:
                    result['num_exps'] = len(exps_pass)+len(exps_fail)
                    result['num_exp_fail'] = len(exps_fail)
                    result['num_pass2fail'] = num_pass2fail
                    result['num_fail2pass'] = num_fail2pass
                    result['num_pass2fail_ent_inc'] = num_pass2fail_ent_inc
                    result['num_pass2fail_ent_dec'] = num_pass2fail_ent_dec
                    result['num_pass2fail_ent_same'] = num_pass2fail_ent_same
                    result['num_fail2pass_ent_inc'] = num_fail2pass_ent_inc
                    result['num_fail2pass_ent_dec'] = num_fail2pass_ent_dec
                    result['num_fail2pass_ent_same'] = num_fail2pass_ent_same
                # end if
            # end if
            results.append(result)
        # end for
        return results
        
    @classmethod
    def analyze(cls, result_file, model_name_file, template_file, saveto):
        result_dict = cls.parse_results(result_file, model_name_file)
        seed_exp_map = cls.get_seed_to_exp_map(template_file)
        results = dict()
        for model in result_dict.keys():
            model_result = result_dict[model]
            results[model] = cls.analyze_model(model_result, seed_exp_map)
        # end for
        Utils.write_json(results, saveto, pretty_format=True)
        return results
    
    @classmethod
    def analyze_hatecheck(cls, result_file, model_name_file, saveto):
        # result_file: Macros.result_dir / f"test_results_{nlp_task}_{search_dataset_name}_{selection_method}" / 'test_results_checklist.txt'
        result_dict = cls.parse_hatecheck_results(result_file, model_name_file)
        results = dict()
        for model in result_dict.keys():
            model_result = result_dict[model]
            results[model] = cls.analyze_model_hatecheck(model_result)
        # end for
        Utils.write_json(results, saveto, pretty_format=True)
        return results