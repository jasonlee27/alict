# This script is to analyze the 
# model test results

from typing import *
from pathlib import Path

# from checklist.test_suite import TestSuite as suite

from ..utils.Macros import Macros
from ..utils.Utils import Utils

import os, re


class Result:

    @classmethod
    def get_model_results_from_string(cls, result_str, model_name):
        p = re.compile(
            f">>>>> MODEL: {model_name}(.*?)?<<<<< MODEL\: {model_name}",
            re.DOTALL
        )
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
    def get_pass_sents_from_model_string(cls, model_result_str):
        result = list()
        for l in model_result_str.splitlines():
            sent_search = re.search(r"DATA::PASS::(\d)::(\d)::(.*)", l)
            if sent_search:
                sent = sent_search.group(3)
                tokens = Utils.tokenize(sent)
                sent = Utils.detokenize(tokens)
                result.append({
                    'pred': sent_search.group(1),
                    'label': sent_search.group(2),
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
            sent_search = re.search(r"DATA::FAIL::(\d)::(\d)::(.*)", l)
            if sent_search:
                sent = sent_search.group(3)
                tokens = Utils.tokenize(sent)
                result.append({
                    'pred': sent_search.group(1),
                    'label': sent_search.group(2),
                    'sent': Utils.detokenize(tokens),
                    'key': sent.replace(' ', '')
                })
            # end if
        # end for
        return result

    @classmethod
    def get_task_from_result_str(cls, result_str):
        task_search = re.search(
            f"\*\*\*\*\* TASK\: (.*) \*\*\*\*\*",
            result_str.splitlines()[0]
        )
        return task_search.group(1).strip()
            
    @classmethod
    def parse_model_results(cls, result_str, model_name, task):
        results = list()
        model_results = cls.get_model_results_from_string(result_str, model_name)
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
    def get_seed_to_exp_map(cls, template_file):
        templates = Utils.read_json(template_file)
        seed_exp_map = dict()
        for t in templates:
            seed_exp_map[t['requirement']['description']] = dict()
            for seed in t['inputs'].keys():
                exp_list = list()
                for exp in t['inputs'][seed]['exp_inputs']:
                    tokens = Utils.tokenize(exp[5])
                    sent = Utils.detokenize(tokens)
                    exp_list.append(sent.replace(' ',''))
                # end for
                tokens = Utils.tokenize(seed)
                _seed = Utils.detokenize(tokens)
                _seed = _seed.replace(' ','')
                seed_exp_map[t['requirement']['description']][_seed] = exp_list
            # end for
        # end for
        return seed_exp_map

    @classmethod
    def analyze_model(cls, model_results, seed_exp_map):
        reqs = set([r['req'] for r in model_results])
        results = list()
        for r in reqs:
            result = {
                'req': r,
                'is_exps_exist': False
            }
            seeds = [mr for mr in model_results if mr['req']==r and mr['sent_type']=='SEED']
            exps = [mr for mr in model_results if mr['req']==r and mr['sent_type']=='EXP']
            num_pass2fail = 0
            num_fail2pass = 0
            if any(exps):
                result['is_exps_exist'] = True
                result['pass->fail'] = list()
                result['fail->pass'] = list()
                seeds_pass = seeds[0]['pass']
                seeds_fail = seeds[0]['fail']
                exps_pass = exps[0]['pass']
                exps_fail = exps[0]['fail']
                for p in seeds_pass:
                    pass2fail_dict = {'from': list(), 'to': list()}
                    for exp in seed_exp_map[r][p['key']]:
                        for ef in exps_fail:
                            if exp==ef['key']:
                                pass2fail_dict['to'].append((ef['sent'], ef['pred'], ef['label']))
                                num_pass2fail += 1
                            # end if
                        # end for
                    # end for
                    if any(pass2fail_dict['to']):
                        pass2fail_dict['from'] = (p['sent'], p['pred'], p['label'])
                        result['pass->fail'].append(pass2fail_dict)
                    # end if
                # end for
                
                for f in seeds_fail:
                    fail2pass_dict = {'from': list(), 'to': list()}
                    for exp in seed_exp_map[r][f['key']]:
                        for ep in exps_pass:
                            if exp==ep['key']:
                                fail2pass_dict['to'].append((ep['sent'], ep['pred'], ep['label']))
                                num_fail2pass += 1
                            # end if
                        # end for
                    # end for
                    if any(fail2pass_dict['to']):
                        fail2pass_dict['from'] = (f['sent'], f['pred'], f['label'])
                        result['fail->pass'].append(fail2pass_dict)
                    # end if
                # end for
                if result['is_exps_exist']:
                    result['num_pass2fail'] = num_pass2fail
                    result['num_fail2pass'] = num_fail2pass
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
