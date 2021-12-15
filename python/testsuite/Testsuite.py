# This script is to add test templates to
# the checklist Testing model framework
# and test model.

from typing import *

import re, os
import copy
import spacy
import numpy as np

from pathlib import Path

from nltk.tokenize import word_tokenize as tokenize

from checklist.editor import Editor
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
# from checklist.perturb import Perturb

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from .Template import Template

class Testsuite:

    @classmethod
    def map_labels(cls, task: str, label):
        if task==Macros.sa_task:
            return Macros.sa_label_map[label]
        # end if
        return

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
            "sent": " ".join(template_list),
            "values": template_values,
            "label": cls.map_labels(task, template["label"])
        }

    @classmethod
    def get_templates(cls):
        for task in Macros.datasets.keys():
            print(f"TASK: {task}")
            new_input_dicts = Template.get_new_inputs(Macros.result_dir/f"cfg_expanded_inputs_{task}.json")
            seeds_per_task = list()
            seed_templates_per_task = list()
            exp_templates_per_task = list()
            for t_i in range(len(new_input_dicts)):
                req_cksum = Utils.get_cksum(new_input_dicts[t_i]["requirement"]["description"])
                print("CAP: ", new_input_dicts[t_i]["requirement"]["capability"])
                print("REQ: ", new_input_dicts[t_i]["requirement"]["description"])
                
                res_dir = Macros.result_dir/ f"templates_{task}"
                if (not os.path.exists(str(res_dir / f"seeds_{req_cksum}.json"))) or \
                   (not os.path.exists(str(res_dir / f"templates_seed_{req_cksum}.json"))) or \
                   (not os.path.exists(str(res_dir / f"templates_exp_{req_cksum}.json"))):
                    Template.get_templates()
                # end if
                
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

                seed_template_res = list()
                seed_templates = Utils.read_json(res_dir / f"templates_seed_{req_cksum}.json")
                for tp in seed_templates:
                    tp_res = cls.get_template(tp, task)
                    seed_template_res.append(tp_res)
                # end for
                seed_templates_per_task.append({
                    "capability": new_input_dicts[t_i]["requirement"]["capability"],
                    "description": new_input_dicts[t_i]["requirement"]["description"],
                    "templates": seed_template_res
                })

                exp_template_res = list()
                exp_templates = Utils.read_json(res_dir / f"templates_exp_{req_cksum}.json")
                for tp in exp_templates:
                    tp_res = cls.get_template(tp, task)
                    exp_template_res.append(tp_res)
                # end for
                exp_templates_per_task.append({
                    "capability": new_input_dicts[t_i]["requirement"]["capability"],
                    "description": new_input_dicts[t_i]["requirement"]["description"],
                    "templates": exp_template_res
                })
            # end for
            yield task, seeds_per_task, seed_templates_per_task, exp_templates_per_task
        # end for
        return

    @classmethod
    def write_editor_template(cls, editor, task,
                              seed_dicts, seed_template_dicts, exp_template_dicts):
        res_dir = Macros.result_dir / "test_results"
        res_dir.mkdir(parents=True, exist_ok=True)

        for templates_per_req in seed_dicts:
            t = None
            suite = TestSuite()
            for temp_i, temp in enumerate(templates_per_req["templates"]):
                if temp_i==0:
                    t = editor.template(temp["sent"],
                                        labels=temp["label"],
                                        save=True)
                else:
                    t += editor.template(temp["sent"],
                                         labels=temp["label"],
                                         save=True)
                # end if
            # end for
            test = MFT(**t)
            suite.add(test, 
                      name=task,
                      capability=templates_per_req["capability"]+"::SEED",
                      description=templates_per_req["description"])
            test_cksum = Utils.get_cksum(
                task+templates_per_req["capability"]+templates_per_req["description"]
            )
            suite.save(res_dir / f'{task}_testsuite_seeds_{test_cksum}.pkl')
        # end for
        
        for templates_per_req in seed_template_dicts:
            t = None
            suite = TestSuite()
            for temp_i, temp in enumerate(templates_per_req["templates"]):
                for key, val in temp["values"].items():
                    if key not in editor.lexicons.keys():
                        editor.add_lexicon(key, val)
                    # end if
                # end for
                if temp_i==0:
                    t = editor.template(temp["sent"],
                                        labels=temp["label"],
                                        save=True)
                else:
                    t += editor.template(temp["sent"],
                                         labels=temp["label"],
                                         save=True)
                # end if
            # end for
            test = MFT(**t)
            suite.add(test, 
                      name=task,
                      capability=templates_per_req["capability"]+"::SEED_TEMPS",
                      description=templates_per_req["description"])
            test_cksum = Utils.get_cksum(
                task+templates_per_req["capability"]+templates_per_req["description"]
            )
            suite.save(res_dir / f'{task}_testsuite_seed_templates_{test_cksum}.pkl')
        # end for
        
        for templates_per_req in exp_template_dicts:
            t = None
            suite = TestSuite()
            for temp_i, temp in enumerate(templates_per_req["templates"]):
                for key, val in temp["values"].items():
                    if key not in editor.lexicons.keys():
                        editor.add_lexicon(key, val)
                    # end if
                # end for
                if temp_i==0:
                    t = editor.template(temp["sent"],
                                        labels=temp["label"],
                                        save=True)
                else:
                    t += editor.template(temp["sent"],
                                         labels=temp["label"],
                                         save=True)
                # end if
            # end for
            test = MFT(**t)
            suite.add(test, 
                      name=task,
                      capability=templates_per_req["capability"]+"::EXP_TEMPS",
                      description=templates_per_req["description"])
            test_cksum = Utils.get_cksum(
                task+templates_per_req["capability"]+templates_per_req["description"]
            )
            suite.save(res_dir / f'{task}_testsuite_exp_templates_{test_cksum}.pkl')
        # end for
        return

    @classmethod
    def write_testsuites(cls):
        for task, seed, seed_temp, exp_temp in cls.get_templates():
            editor = Editor()
            Testsuite.write_editor_template(editor, task, seed, seed_temp, exp_temp)
        # end for
        return


# if __name__=="__main__":
#     Testsuite.write_testsuites()
Testsuite.write_testsuites()
