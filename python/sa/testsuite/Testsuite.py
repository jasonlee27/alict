# This script is to add test templates to
# the checklist Testing model framework
# and test model.

from typing import *

import re, os
import copy
import spacy
import random
import numpy as np

from pathlib import Path

from nltk.tokenize import word_tokenize as tokenize

from checklist.editor import Editor
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger
from .Search import Search
# from .Transform import TransformOperator
from .Template import Template

random.seed(Macros.SEED)

class Testsuite:

    @classmethod
    def map_labels(cls, task: str, label):
        if task==Macros.sa_task:
            if type(label)==list:
                label_not= [v for k, v in Macros.sa_label_map.items() if k not in label]
                is_not_label = lambda x, pred, *args: pred != label_not[0]
                return is_not_label
            # end if
            return Macros.sa_label_map[label]
        # end if
        return

    @classmethod
    def add_lexicon(cls, editor, lexicon_dict):
        placeholder_keys = list(lexicon_dict.keys())
        if len(placeholder_keys)>Macros.num_synonym_placeholders:
            # when there are too many synonyms, it is memory-costly and it kills code running.
            # So, when too many synonyms detected, we shrink the size of synonyms by reducing them.
            # first check which placeholder has too many number of synonyms
            random.shuffle(placeholder_keys)
            for k_i, key in enumerate(placeholder_keys):
                if k_i>=Macros.num_synonym_placeholders:
                    lexicon_dict[key] = lexicon_dict[key][0]
                else:
                    lexicon_dict[key] = lexicon_dict[key][:Macros.max_num_synonyms//3]
                # end if
            # end for
        else:
            for key in placeholder_keys:
                lexicon_dict[key] = lexicon_dict[key][:Macros.max_num_synonyms//3]
            # end for
        # end if
        for key, val in lexicon_dict.items():
            if key not in editor.lexicons.keys():
                editor.add_lexicon(key, val)
            # end if
        # end for
        return editor

    @classmethod
    def add_template(cls, t, editor, template):
        if t is None:
            if callable(template["label"]):
                t = editor.template(template["sent"],
                                    save=True)
            else:
                t = editor.template(template["sent"],
                                    labels=template["label"],
                                    save=True)
            # end if
        elif t is not None and len(t.data)<Macros.max_num_sents:
            if callable(template["label"]):
                t += editor.template(template["sent"],
                                    save=True)
            else:
                t += editor.template(template["sent"],
                                     labels=template["label"],
                                     save=True)
            # end if
        # end if
        return t
    
    @classmethod
    def get_template(cls, template, task):
        template_list = list()
        template_values = dict()
        input_sent = template["input"]
        for tok_i in range(len(template["place_holder"])):
            if type(template["place_holder"][tok_i])==str:
                template_list.append(template["place_holder"][tok_i])
            elif type(template["place_holder"][tok_i])==dict:
                key = list(template["place_holder"][tok_i].keys())[0]
                _key = key[1:-1]
                template_list.append(key)
                template_values[_key] = template["place_holder"][tok_i][key]
            # end if
        # end for
        return {
            "sent": Utils.detokenize(template_list), #" ".join(template_list),
            "values": template_values,
            "label": cls.map_labels(task, template["label"])
        }
    
    @classmethod
    def get_templates(cls, nlp_task, dataset, selection_method, num_seeds, logger):
        task = nlp_task
        if num_seeds<0:
            template_out_dir = f"templates2_{nlp_task}_{dataset}_{selection_method}"
            cfg_res_file_name = f"cfg_expanded_inputs2_{task}_{dataset}_{selection_method}.json"
        else:
            template_out_dir = f"templates2_{nlp_task}_{dataset}_{selection_method}_{num_seeds}seeds"
            cfg_res_file_name = f"cfg_expanded_inputs2_{task}_{dataset}_{selection_method}_{num_seeds}seeds.json"
        # end if
        # selection_method = 'RANDOM' if is_random_select else 'PROB'
        # new_input_dicts = Template.get_new_inputs(
        #     Macros.result_dir / cfg_res_file_name,
        #     task,
        #     dataset,
        #     num_seeds=num_seeds,
        #     selection_method=selection_method,
        #     logger=logger
        # )
        seeds_per_task = list()
        exps_per_task = list()
        seed_templates_per_task = list()
        exp_templates_per_task = list()
        transform_reqs = list()
        new_input_dicts = Utils.read_json(Macros.result_dir / cfg_res_file_name)
        for t_i in range(len(new_input_dicts)):
            req_cksum = Utils.get_cksum(new_input_dicts[t_i]["requirement"]["description"])
            res_dir = Macros.result_dir/ template_out_dir
            # if (not os.path.exists(str(res_dir / f"seeds_{req_cksum}.json"))):
            #     Template.get_templates(
            #         num_seeds=num_seeds,
            #         nlp_task=nlp_task,
            #         dataset=dataset,
            #         selection_method=selection_method
            #     )
            # # end if

            transform_reqs.append(new_input_dicts[t_i]["requirement"]["transform"])

            seed_res = list()
            seeds = Utils.read_json(res_dir / f"seeds_{req_cksum}.json")
            for sd in seeds:
                sd_res = cls.get_template(sd, task)
                seed_res.append(sd_res)
            # end for
            seeds_per_task.append({
                "capability": new_input_dicts[t_i]["requirement"]["capability"],
                "description": new_input_dicts[t_i]["requirement"]["description"],
                "templates": seed_res
            })
            
            exps = Utils.read_json(res_dir / f"exps_{req_cksum}.json")
            if exps is not None:
                exp_res = list()
                for e in exps:
                    e_res = cls.get_template(e, task)
                    exp_res.append(e_res)
                # end for
                exps_per_task.append({
                    "capability": new_input_dicts[t_i]["requirement"]["capability"],
                    "description": new_input_dicts[t_i]["requirement"]["description"],
                    "templates": exp_res
                })
            # end if
            
            # seed_templates = Utils.read_json(res_dir / f"templates_seed_{req_cksum}.json")
            # if seed_templates is not None:
            #     seed_template_res = list()
            #     for tp in seed_templates:
            #         tp_res = cls.get_template(tp, task)
            #         seed_template_res.append(tp_res)
            #     # end for
            #     seed_templates_per_task.append({
            #         "capability": new_input_dicts[t_i]["requirement"]["capability"],
            #         "description": new_input_dicts[t_i]["requirement"]["description"],
            #         "templates": seed_template_res
            #     })
            # # end if

            # exp_templates = Utils.read_json(res_dir / f"templates_exp_{req_cksum}.json")
            # if exp_templates is not None:
            #     exp_template_res = list()
            #     for tp in exp_templates:
            #         tp_res = cls.get_template(tp, task)
            #         exp_template_res.append(tp_res)
            #     # end for
            #     exp_templates_per_task.append({
            #         "capability": new_input_dicts[t_i]["requirement"]["capability"],
            #         "description": new_input_dicts[t_i]["requirement"]["description"],
            #         "templates": exp_template_res
            #     })
            # # end if
        # end for
        yield task, seeds_per_task, exps_per_task, seed_templates_per_task, exp_templates_per_task, transform_reqs
        return

    @classmethod
    def write_seed_testsuite(cls,
                             task,
                             dataset,
                             seed_dicts,
                             res_dir,
                             logger):
        for t_i, templates_per_req in enumerate(seed_dicts):
            test_cksum = Utils.get_cksum(templates_per_req["description"])
            if not os.path.exists(str(res_dir / f'{task}_testsuite_seeds_{test_cksum}.pkl')):
                logger.print(f"{task}::SEED::<"+templates_per_req["description"]+f">::{test_cksum}::", end='')
                t = None
                suite = TestSuite()
                editor = Editor()
                for template in templates_per_req["templates"]:
                    t = cls.add_template(t, editor, template)
                # end for
                
                if callable(templates_per_req["templates"][0]['label']):
                    test = MFT(t.data, Expect.single(templates_per_req["templates"][0]['label']), templates=t.templates)
                else:
                    test = MFT(**t)
                # end if
                suite.add(test,
                          name=f"{task}::SEED::"+templates_per_req["description"],
                          capability=templates_per_req["capability"]+"::SEED",
                          description=templates_per_req["description"])
                num_data = sum([len(suite.tests[k].data) for k in suite.tests.keys()])
                if num_data>0:
                    # test_cksum = Utils.get_cksum(
                    #     task+templates_per_req["capability"]+templates_per_req["description"]
                    # )
                    suite.save(res_dir / f'{task}_testsuite_seeds_{test_cksum}.pkl')
                    logger.print('SAVED')
                else:
                    logger.print('NO_DATA')
                # end if
            # end if
        # end for
        return

    @classmethod
    def write_exp_testsuite(cls,
                            task,
                            dataset,
                            exp_dicts,
                            res_dir,
                            logger):
        for t_i, templates_per_req in enumerate(exp_dicts):
            test_cksum = Utils.get_cksum(templates_per_req["description"])
            if not os.path.exists(str(res_dir / f'{task}_testsuite_exps_{test_cksum}.pkl')):
                logger.print(f"{task}::EXP::<"+templates_per_req["description"]+f">::{test_cksum}::", end='')
                t = None
                suite = TestSuite()
                editor = Editor()
                for template in templates_per_req["templates"]:
                    t = cls.add_template(t, editor, template)
                # end for
                
                if callable(templates_per_req["templates"][0]['label']):
                    test = MFT(t.data, Expect.single(templates_per_req["templates"][0]['label']), templates=t.templates)
                else:
                    test = MFT(**t)
                # end if
                suite.add(test,
                          name=f"{task}::EXP::"+templates_per_req["description"],
                          capability=templates_per_req["capability"]+"::EXP",
                          description=templates_per_req["description"])
                num_data = sum([len(suite.tests[k].data) for k in suite.tests.keys()])
                if num_data>0:
                    # test_cksum = Utils.get_cksum(
                    #     task+templates_per_req["capability"]+templates_per_req["description"]
                    # )
                    suite.save(res_dir / f'{task}_testsuite_exps_{test_cksum}.pkl')
                    logger.print('SAVED')
                else:
                    logger.print('NO_DATA')
                # end if
            # end if
        # end for
        return

    @classmethod
    def write_seed_template_testsuite(cls,
                                      task,
                                      dataset,
                                      seed_template_dicts,
                                      res_dir,
                                      logger):
        for t_i, templates_per_req in enumerate(seed_template_dicts):
            test_cksum = Utils.get_cksum(templates_per_req["description"])
            if not os.path.exists(str(res_dir / f'{task}_testsuite_seed_templates_{test_cksum}.pkl')):
                logger.print(f"{task}::SEED_TEMPS::<"+templates_per_req["description"]+f">::{test_cksum}::", end='')
                t = None
                suite = TestSuite()
                editor = Editor()
                for template in templates_per_req["templates"]:
                    editor = cls.add_lexicon(editor, template["values"])
                    t = cls.add_template(t, editor, template)
                # end for
                if callable(templates_per_req["templates"][0]['label']):
                    test = MFT(t.data, Expect.single(templates_per_req["templates"][0]['label']), templates=t.templates)
                else:
                    test = MFT(**t)
                # end if
                suite.add(test,
                          name=f"{task}::SEED_TEMPS::"+templates_per_req["description"],
                          capability=templates_per_req["capability"]+"::SEED_TEMPS",
                          description=templates_per_req["description"])
                num_data = sum([len(suite.tests[k].data) for k in suite.tests.keys()])
                if num_data>0:
                    suite.save(res_dir / f'{task}_testsuite_seed_templates_{test_cksum}.pkl')
                    logger.print('SAVED')
                else:
                    logger.print('NO_DATA')
                # end if
                del t, test, suite
            # end if
        # end for
        return

    @classmethod
    def write_exp_template_testsuite(cls,
                                     task,
                                     dataset,
                                     exp_template_dicts,
                                     res_dir,
                                     logger):
        for t_i, templates_per_req in enumerate(exp_template_dicts):
            test_cksum = Utils.get_cksum(templates_per_req["description"])
            if not os.path.exists(str(res_dir / f'{task}_testsuite_exp_templates_{test_cksum}.pkl')) and \
               any(templates_per_req["templates"]):
                logger.print(f"{task}::EXP_TEMPS::<"+templates_per_req["description"]+f">::{test_cksum}::", end='')
                t = None
                suite = TestSuite()
                editor = Editor()
                for template in templates_per_req["templates"]:
                    editor = cls.add_lexicon(editor, template["values"])
                    t = cls.add_template(t, editor, template)
                # end for
                if callable(templates_per_req["templates"][0]['label']):
                    test = MFT(t.data, Expect.single(templates_per_req["templates"][0]['label']), templates=t.templates)
                else:
                    test = MFT(**t)
                # end if
                suite.add(test, 
                          name=f"{task}::EXP_TEMPS::"+templates_per_req["description"],
                          capability=templates_per_req["capability"]+"::EXP_TEMPS",
                          description=templates_per_req["description"])
                num_data = sum([len(suite.tests[k].data) for k in suite.tests.keys()])
                if num_data>0:
                    suite.save(res_dir / f'{task}_testsuite_exp_templates_{test_cksum}.pkl')
                    logger.print('SAVED')
                else:
                    logger.print('NO_DATA')
                # end if
            # end for
        # end for
        return
        
    @classmethod
    def write_editor_templates(cls,
                               task,
                               dataset,
                               selection_method,
                               seed_dicts,
                               exp_dicts,
                               seed_template_dicts,
                               exp_template_dicts,
                               transform_reqs,
                               num_seeds,
                               logger):
        # selection_method = 'RANDOM' if is_random_select else 'PROB'
        if num_seeds<0:
            res_dir = Macros.result_dir / f"test_results2_{task}_{dataset}_{selection_method}"
        else:
            res_dir = Macros.result_dir / f"test_results2_{task}_{dataset}_{selection_method}_{num_seeds}seeds"
        # end if
        res_dir.mkdir(parents=True, exist_ok=True)
        cls.write_seed_testsuite(task,
                                 dataset,
                                 seed_dicts,
                                 res_dir,
                                 logger)
        
        cls.write_exp_testsuite(task,
                                dataset,
                                exp_dicts,
                                res_dir,
                                logger)

        if any(seed_template_dicts):
            cls.write_seed_template_testsuite(task,
                                              dataset,
                                              seed_template_dicts,
                                              res_dir,
                                              logger)
        # end if
        if any(exp_template_dicts):
            cls.write_exp_template_testsuite(task,
                                             dataset,
                                             exp_template_dicts,
                                             res_dir,
                                             logger)
        # end if
        return

    @classmethod
    def write_testsuites(cls, nlp_task, dataset, selection_method, num_seeds, log_file):
        logger = Logger(logger_file=log_file,
                        logger_name='testsuite')
        logger.print('Generate Testsuites from Templates ...')
        for task, seed, exp, seed_temp, exp_temp, transform_reqs \
            in cls.get_templates(nlp_task, dataset, selection_method, num_seeds, logger):
            Testsuite.write_editor_templates(task,
                                             dataset,
                                             selection_method,
                                             seed,
                                             exp,
                                             seed_temp,
                                             exp_temp,
                                             transform_reqs,
                                             num_seeds,
                                             logger)
        # end for
        return


# if __name__=="__main__":
#     Testsuite.write_testsuites()
