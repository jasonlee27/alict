# This script is to analyze the 
# model test results

import os, re

from typing import *
from pathlib import Path
from scipy.stats import entropy

# from checklist.test_suite import TestSuite as suite

from ..requirement.Requirements import Requirements
from ..seed.FairnessSearch import FairnessSearch, FAIRNESS_REQ
from ..utils.Macros import Macros
from ..utils.Utils import Utils


class Result:

    @classmethod
    def get_model_results_from_string(cls, result_str, model_name):
        p = re.compile(
            f">>>>> MODEL: {model_name}(.*?)?\n<<<<< MODEL: {model_name}",
            re.DOTALL
        )
        model_results = [m.strip() for m in p.findall(result_str)]
        return model_results

    @classmethod
    def get_model_checklist_results_from_string(cls, result_str, model_name):
        p = re.compile(
            f">>>>> MODEL: {model_name}(.*?)?\n<<<<< MODEL: {model_name}",
            re.DOTALL
        )
        model_results_str = [m.strip() for m in p.findall(result_str)]
        model_results = list()
        if any(model_results_str):
            model_res_split = model_results_str[0].split('Test cases:')
            for m_i, m in enumerate(model_res_split):
                if m_i>0:
                    lc = model_res_split[m_i-1].splitlines()[-1].strip()
                    model_results.append({
                        'lc': lc,
                        'pass': cls.get_pass_sents_from_model_string(m),
                        'fail': cls.get_fail_sents_from_model_string(m)
                    })
                # end if
            # end for
        # end for
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
    def get_requirement_from_string_for_fairness(cls, model_result_str, task):
        req_search = re.search(f"Running {task}::([A-Z]+)\_\d+::(.*)", model_result_str)
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
    def get_pass_sents_from_model_string_for_fairness(cls, model_result_str):
        result = list()
        for l in model_result_str.splitlines():
            sent_search = re.search(r"DATA::PASS::(\d*\.?\d* \d*\.?\d* \d*\.?\d*)::\[([0-9 ]+)\]::None::(.*)", l)
            if sent_search:
                sent = sent_search.group(3)
                tokens = Utils.tokenize(sent)
                sent = Utils.detokenize(tokens)
                conf = [
                    eval(v) for v in sent_search.group(1).split()
                ]
                preds = [
                    eval(v) for v in sent_search.group(2).split()
                ]
                if len(preds)>1:
                    result.append({
                        'conf': conf,
                        'pred': preds,
                        'label': None,
                        'sent': sent,
                        'key': sent.replace(' ', '')
                    })
                # end if
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
    def get_fail_sents_from_model_string_for_fairness(cls, model_result_str):
        result = list()
        for l in model_result_str.splitlines():
            sent_search = re.search(r"DATA::FAIL::(\d*\.?\d* \d*\.?\d* \d*\.?\d*)::\[([0-9 ]+)\]::None::(.*)", l)
            if sent_search:
                sent = sent_search.group(3)
                tokens = Utils.tokenize(sent)
                sent = Utils.detokenize(tokens)
                conf = [
                    eval(v) for v in sent_search.group(1).split()
                ]
                preds = [
                    eval(v) for v in sent_search.group(2).split()
                ]
                if len(preds)>1:
                    result.append({
                        'conf': conf,
                        'pred': preds,
                        'label': None,
                        'sent': sent,
                        'key': sent.replace(' ', '')
                    })
                # end if
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
    def parse_model_on_checklist_results(cls, result_str, model_name, task):
        results = list()
        model_results = cls.get_model_checklist_results_from_string(result_str, model_name)
        temp_dict = dict()
        for r in model_results:
            # if r['lc'] in Macros.CHECKLIST_LC_LIST[8:10]:
            #     lc = Macros.LC_MAP[str(Macros.CHECKLIST_LC_LIST[8:10])]
            #     if lc not in temp_dict.keys():
            #         temp_dict[lc] = {
            #             'req': lc,
            #             'pass': r['pass'],
            #             'fail': r['fail']
            #         }
            #     else:
            #         temp_dict[lc]['pass'].extend(r['pass'])
            #         temp_dict[lc]['fail'].extend(r['fail'])
            #         results.append({
            #             'req': lc,
            #             'pass': temp_dict[lc]['pass'],
            #             'fail': temp_dict[lc]['fail']
            #         })
            #     # end if
            # else:
            lc = Macros.LC_MAP[r['lc']]
            results.append({
                'req': lc,
                'pass': r['pass'],
                'fail': r['fail']
            })
            # end if
        # end for
        return results

    @classmethod
    def parse_model_results_for_fairness(cls, result_str, model_name, task):
        results = list()
        model_results = cls.get_model_results_from_string(result_str, model_name)
        for r in model_results:
            sent_type, req = cls.get_requirement_from_string_for_fairness(r, task)
            results.append({
                'sent_type': sent_type,
                'req': req,
                'pass': cls.get_pass_sents_from_model_string_for_fairness(r),
                'fail': cls.get_fail_sents_from_model_string_for_fairness(r)
            })
        # end for
        return results

    @classmethod
    def parse_results(cls, result_file, model_name_file):
        if type(model_name_file)!=list:
            model_names = Utils.read_txt(model_name_file)
        else:
            model_names = model_name_file
        # end if
        with open(result_file, "r") as f:
            line = f.read()
            task = cls.get_task_from_result_str(line)
            return {
                model.strip(): cls.parse_model_results(
                    line, model.strip(), task
                )
                for model in model_names
            }
        # end with
    
    @classmethod
    def parse_results_for_fairness(cls, result_file, model_name_file):
        if type(model_name_file)!=list:
            model_names = Utils.read_txt(model_name_file)
        else:
            model_names = model_name_file
        # end if
        with open(result_file, "r") as f:
            line = f.read()
            task = cls.get_task_from_result_str(line)
            return {
                model.strip(): cls.parse_model_results_for_fairness(
                    line, model.strip(), task
                )
                for model in model_names
            }
        # end with

    @classmethod
    def parse_checklist_results(cls, result_file, model_name_file):
        if type(model_name_file)!=list:
            model_names = Utils.read_txt(model_name_file)
        else:
            model_names = model_name_file
        # end if
        with open(result_file, "r") as f:
            line = f.read()
            task = cls.get_task_from_result_str(line)
            return {
                model.strip(): cls.parse_model_on_checklist_results(
                    line, model.strip(), task
                )
                for model in model_names
            }
        # end with

    @classmethod
    def get_seed_to_exp_map(cls, template_file):
        t = Utils.read_json(template_file)
        seed_exp_map = dict()
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
            seed_exp_map[_seed] = exp_list
        # end for
        return seed_exp_map

    @classmethod
    def analyze_model_checklist(cls, model_results):
        reqs = sorted(set([r['req'] for r in model_results]))
        results = list()
        for r in reqs:
            testcases = [mr for mr in model_results if mr['req']==r]
            result = {'req': r, 'is_exps_exist': False}
            tcs_pass = testcases[0]['pass']
            tcs_fail = testcases[0]['fail']
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
                                    'ent': ef.get('ent', None)
                                })
                                num_pass2fail += 1
                                if p.get('ent', None) is not None and\
                                    ef.get('ent', None) is not None:
                                    if p['ent']<ef['ent']:
                                        num_pass2fail_ent_inc += 1
                                    elif p['ent']>ef['ent']:
                                        num_pass2fail_ent_dec += 1
                                    else:
                                        num_pass2fail_ent_same += 1
                                    # end if
                                # end if
                            # end if
                        # end for

                        for ep in exps_pass:
                            if exp==ep['key']:
                                pass2pass_dict['to'].append({
                                    'sent': ep['sent'],
                                    'pred': ep['pred'],
                                    'label': ep['label'],
                                    'conf': ep['conf'],
                                    'ent': ep.get('ent', None)
                                })
                                num_pass2pass += 1
                                if p.get('ent', None) is not None and\
                                    ep.get('ent', None) is not None:
                                    if p['ent']<ep['ent']:
                                        num_pass2pass_ent_inc += 1
                                    elif p['ent']>ep['ent']:
                                        num_pass2pass_ent_dec += 1
                                    else:
                                        num_pass2pass_ent_same += 1
                                    # end if
                                # end if
                            # end if
                        # end for
                    # end for
                    if any(pass2fail_dict['to']):
                        pass2fail_dict['from'] = {
                            'sent': p['sent'],
                            'pred': p['pred'],
                            'label': p['label'],
                            'conf': p['conf'],
                            'ent': p.get('ent', None)
                        }
                        result['pass->fail'].append(pass2fail_dict)
                    # end if

                    if any(pass2pass_dict['to']):
                        pass2pass_dict['from'] = {
                            'sent': p['sent'],
                            'pred': p['pred'],
                            'label': p['label'],
                            'conf': p['conf'],
                            'ent': p.get('ent', None)
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
                                    'ent': ep.get('ent', None)
                                })
                                num_fail2pass += 1
                                if f.get('ent', None) is not None and\
                                    ep.get('ent', None) is not None:
                                    if f['ent']<ep['ent']:
                                        num_fail2pass_ent_inc += 1
                                    elif f['ent']>ep['ent']:
                                        num_fail2pass_ent_dec += 1
                                    else:
                                        num_fail2pass_ent_same += 1
                                    # end if
                                # end if
                            # end if
                        # end for

                        for ef in exps_fail:
                            if exp==ef['key']:
                                fail2fail_dict['to'].append({
                                    'sent': ef['sent'],
                                    'pred': ef['pred'],
                                    'label': ef['label'],
                                    'conf': ef['conf'],
                                    'ent': ef.get('ent', None)
                                })
                                num_fail2fail += 1
                                if f.get('ent', None) is not None and\
                                    ef.get('ent', None) is not None:
                                    if f['ent']<ef['ent']:
                                        num_fail2fail_ent_inc += 1
                                    elif f['ent']>ef['ent']:
                                        num_fail2fail_ent_dec += 1
                                    else:
                                        num_fail2fail_ent_same += 1
                                    # end if
                                # end if
                            # end if
                        # end for
                    # end for
                    if any(fail2pass_dict['to']):
                        fail2pass_dict['from'] = {
                            'sent': f['sent'],
                            'pred': f['pred'],
                            'label': f['label'],
                            'conf': f['conf'],
                            'ent': f.get('ent', None)
                        }
                        result['fail->pass'].append(fail2pass_dict)
                    # end if

                    if any(fail2fail_dict['to']):
                        fail2fail_dict['from'] = {
                            'sent': f['sent'],
                            'pred': f['pred'],
                            'label': f['label'],
                            'conf': f['conf'],
                            'ent': f.get('ent', None)
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
    def analyze(cls,
                nlp_task,
                template_result_dir,
                result_dir,
                model_name_file,
                saveto):
        result_file = result_dir / 'test_results.txt'
        result_dict = cls.parse_results(result_file, model_name_file)
        reqs = Requirements.get_requirements(nlp_task)
        seed_exp_map = dict()
        for req in reqs:
            lc_desc = req['description']
            cksum_val = Utils.get_cksum(lc_desc)
            template_file = template_result_dir / f"cfg_expanded_inputs_{cksum_val}.json"
            if os.path.exists(result_dir / f"{nlp_task}_testsuite_seeds_{cksum_val}.pkl") or \
               os.path.exists(result_dir / f"{nlp_task}_testsuite_exps_{cksum_val}.pkl"):
                seed_exp_map[lc_desc] = cls.get_seed_to_exp_map(template_file)
            # end if
        # end for
        
        results = dict()
        for model in result_dict.keys():
            model_result = result_dict[model]
            results[model] = cls.analyze_model(model_result, seed_exp_map)
            print(model)
        # end for
        Utils.write_json(results, saveto, pretty_format=True)
        return results
    
    @classmethod
    def analyze_checklist(cls, result_file, model_name_file, saveto):
        # result_file: Macros.result_dir / f"test_results_{nlp_task}_{search_dataset_name}_{selection_method}" / 'test_results_checklist.txt'
        result_dict = cls.parse_checklist_results(result_file, model_name_file)
        results = dict()
        for model in result_dict.keys():
            model_result = result_dict[model]
            results[model] = cls.analyze_model_checklist(model_result)
        # end for
        Utils.write_json(results, saveto, pretty_format=True)
        return results

    @classmethod
    def analyze_seed_performance(cls,
                                 seed_result_file,
                                 bl_result_file,
                                 model_name_file=Macros.sa_models_file,
                                 saveto=None):
        # get the our generated seed performance
        result_dict = cls.parse_results(seed_result_file, model_name_file)

        # get the checklist performance
        bl_result_dict = cls.parse_checklist_results(bl_result_file, model_name_file)
        
        results = dict()
        for model in result_dict.keys():
            model_result = result_dict[model]
            model_bl_result = bl_result_dict[model]
            results[model] = list()
            for res in model_result:
                lc = res['req']
                bl_res = [r for r in model_bl_result if lc in r['req']][0]
                    
                num_pass_ours = len(res['pass'])
                num_fail_ours = len(res['fail'])
                num_tot_ours = num_pass_ours+num_fail_ours
                fr_ours = round(100.0*(num_fail_ours*1./num_tot_ours), 3)

                num_pass_bl = len(bl_res['pass'])
                num_fail_bl = len(bl_res['fail'])
                num_tot_bl = num_pass_bl+num_fail_bl
                fr_bl = round(100.0*(num_fail_bl*1./num_tot_bl), 3)
                results[model].append({
                    'lc': lc,
                    'num_total_ours': num_tot_ours,
                    'num_pass_ours': num_pass_ours,
                    'num_fail_ours': num_fail_ours,
                    'failrate_ours': fr_ours,
                    'num_total_bl': num_tot_bl,
                    'num_pass_bl': num_pass_bl,
                    'num_fail_bl': num_fail_bl,
                    'failrate_bl': fr_bl
                })
            # end for
        # end for
        Utils.write_json(results, saveto, pretty_format=True)
        return results

    @classmethod
    def analyze_tosem(
        cls,
        nlp_task,
        template_result_dir,
        result_dir,
        tosem_model_names,
        saveto
    ):
        result_file = result_dir / 'test_results_tosem.txt'
        result_dict = cls.parse_results(result_file, tosem_model_names)
        reqs = Requirements.get_requirements(nlp_task)
        seed_exp_map = dict()
        for req in reqs:
            lc_desc = req['description']
            cksum_val = Utils.get_cksum(lc_desc)
            template_file = template_result_dir / f"cfg_expanded_inputs_{cksum_val}.json"
            if os.path.exists(result_dir / f"{nlp_task}_testsuite_seeds_{cksum_val}.pkl") or \
               os.path.exists(result_dir / f"{nlp_task}_testsuite_exps_{cksum_val}.pkl"):
                seed_exp_map[lc_desc] = cls.get_seed_to_exp_map(template_file)
            # end if
        # end for
        
        results = dict()
        for model in result_dict.keys():
            model_result = result_dict[model]
            results[model] = cls.analyze_model(model_result, seed_exp_map)
            print(model)
        # end for
        Utils.write_json(
            results, 
            saveto, 
            pretty_format=True
        )
        return results

    @classmethod
    def analyze_checklist_tosem(
        cls, 
        result_file, 
        model_name_file, 
        saveto
    ):
        # result_file: Macros.result_dir / f"test_results_{nlp_task}_{search_dataset_name}_{selection_method}" / 'test_results_checklist.txt'
        result_dict = cls.parse_checklist_results(result_file, model_name_file)
        results = dict()
        for model in result_dict.keys():
            model_result = result_dict[model]
            results[model] = cls.analyze_model_checklist(model_result)
        # end for
        Utils.write_json(
            results, 
            saveto, 
            pretty_format=True
        )
        return results

    @classmethod
    def analyze_fairness(
        cls,
        nlp_task,
        search_dataset_name,
        template_result_dir,
        result_dir,
        tosem_model_names,
        saveto
    ):
        result_file = result_dir / 'test_results_fairness.txt'
        result_dict = cls.parse_results_for_fairness(result_file, tosem_model_names)
        seed_exp_map = dict()
        for req in [FAIRNESS_REQ]:
            lc_desc = req['description']
            cksum_val = Utils.get_cksum(lc_desc)
            template_file = template_result_dir / f"cfg_expanded_inputs_{cksum_val}.json"
            if os.path.exists(result_dir / f"{nlp_task}_testsuite_fairness_seeds_{cksum_val}.pkl") or \
               os.path.exists(result_dir / f"{nlp_task}_testsuite_fairness_exps_{cksum_val}.pkl"):
                seed_exp_map[lc_desc] = cls.get_seed_to_exp_map(template_file)
            # end if
        # end for
        
        results = dict()
        for model in result_dict.keys():
            model_result = result_dict[model]
            results[model] = cls.analyze_model(model_result, seed_exp_map)
            print(model)
        # end for
        Utils.write_json(
            results, 
            saveto, 
            pretty_format=True
        )
        return results

    @classmethod
    def analyze_checklist_fairness(
        cls,
        nlp_task,
        search_dataset_name,
        result_dir,
        tosem_model_names,
        saveto
    ):
        result_file = result_dir / 'test_results_fairness_checklist.txt'
        result_dict = cls.parse_results_for_fairness(result_file, tosem_model_names)
        results = dict()
        for model in result_dict.keys():
            model_result = result_dict[model]
            results[model] = cls.analyze_model_checklist(model_result)
            print(model)
        # end for
        Utils.write_json(
            results, 
            saveto, 
            pretty_format=True
        )
        return results
