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
        model_results = re.findall(
            f"\>\>\>\>\> MODEL\: {model_name}\n\_.*\<\<\<\<\<  MODEL\: {model_name}",
            result_str
        )
        return model_results

    @classmethod
    def get_requirement_from_string(cls, model_result_str, task):
        req_search = re.search(f"Running {task}\:\:([A-Z]+)\:\:(.*)$", model_result_str)
        if req_search:
            sent_type = req_search.group(1)
            req =req_search.group(2)
            return sent_type, req
        # end if
        return None, None
        
    @classmethod
    def get_pass_sents_from_model_string(cls, model_result_str_list):
        result = list()
        for l in model_result_str_list:
            sent_search = re.searvch(r"DATA\:\:PASS\:\:(\d)\:\:(\d)\:\:(.*)$", l)
            if sent_search:
                result.append({
                    'pred': sent_search.group(1),
                    'label': sent_search.group(2),
                    'sent': sent_search.group(3)
                })
            # end if
        # end for
        return result

    @classmethod
    def get_fail_sents_from_model_string(cls, model_result_str_list):
        result = list()
        for l in model_result_str_list:
            sent_search = re.searvch(r"DATA\:\:FAIL\:\:(\d)\:\:(\d)\:\:(.*)$", l)
            if sent_search:
                result.append({
                    'pred': sent_search.group(1),
                    'label': sent_search.group(2),
                    'sent': sent_search.group(3)
                })
            # end if
        # end for
        return result

    @classmethod
    def get_task_from_result_str(cls, result_str):
        task_results = re.search(
            f"\*\*\*\*\* TASK\: (.*) \*\*\*\*\*",
            result_str.splitlines()[0]
        )
        return task_results.group(1)

            
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
            task = get_task_from_result_str(line)
            results = dict()
            for model in Utils.read_txt(model_name_file):
                results[model] = cls.parse_model_results(line, model, task)
            # end for
            return results
        # end with

    @classmethod
    def get_seed_to_exp_map(cls, template_file):
        templates = Utils.read_json(json_file)
        seed_exp_map = dict()
        for t in templates:
            seed_exp_map[t['description']] = {
                seed: [
                    exp[5] for exp in t['inputs'][seed]['exp_inputs']
                ]
                for seed in t['inputs'].keys()
            }
        # end for
        return seed_exp_map

    @classmethod
    def analyze_model(cls, model_results, seed_exp_map):
        reqs = set([r['req'] for r in model_results])
        results = list()
        for r in reqs:
            result = {
                'req': r,
                'pass->fail': dict(),
                'fail->pass': dict()
            }
            seeds = [mr for mr in model_results if mr['req']==r and mr['sent_type']=='SEED']
            exps = [mr for mr in model_results if mr['req']==r and mr['sent_type']=='EXP']
            if any(exps):
                seeds_pass = seeds[0]['pass']
                seeds_fail = seeds[0]['fail']
                exps_pass = exps[0]['pass']
                exps_fail = exps[0]['fail']
                for p in seeds_pass:
                    result['pass->fail'][p] = list()
                    for exp in seed_exp_map[r][p]:
                        if exp in exps_fail:
                            result['pass->fail'][p].append(exp)
                        # end if
                    # end for
                # end for
                for f in seeds_fail:
                    result['fail->pass'][f] = list()
                    for exp in seed_exp_map[r][f]:
                        if exp in exps_pass:
                            result['fail->pass'][f].append(exp)
                        # end if
                    # end for
                # end for
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
        Utils.write_json(results, saveto)
        return results
