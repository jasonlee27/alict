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

from Macros import Macros
from Utils import Utils
from Template import Template

class Testmodel:


    @classmethod
    def map_labels(cls, task: str, label):
        if task==Macros.sa_task:
            return Macros.sa_label_map[label]
        # end if
        return

    @classmethod
    def get_templates(cls):
        for task in Macros.datasets.keys():
            new_input_dicts = Template.get_new_inputs(Macros.result_dir/f"cfg_expanded_inputs_{task}.json")
            templates_per_task = list()
            for t_i in range(len(new_input_dicts)):
                req_cksum = Utils.get_cksum(new_input_dicts[t_i]["requirement"]["description"])
                res_dir = Macros.result_dir/ f"templates_{task}"
                if os.path.exists(str(res_dir / f"templates_{req_cksum}.json")):
                    templates = Utils.read_json(res_dir / f"templates_{req_cksum}.json")
                else:
                    Template.get_templates()
                    templates = Utils.read_json(res_dir / f"templates_{req_cksum}.json")
                # end if
                template_res = list()
                for tp in templates:
                    template_list = list()
                    template_values = dict()
                    input_sent = tp["input"]
                    for tok_i in range(len(tp["place_holder"])):
                        if type(tp["place_holder"][tok_i])==str:
                            template_list.append(tp["place_holder"][tok_i])
                        elif type(tp["place_holder"][tok_i])==dict:
                            key = list(tp["place_holder"][tok_i].keys())[0]
                            _key = key[1:-1]
                            template_list.append(key)
                            template_values[_key] = tp["place_holder"][tok_i][key]
                        # end if
                    # end for
                    template_res.append({
                        "sent": " ".join(template_list),
                        "values": template_values,
                        "label": cls.map_labels(task, tp["label"])
                    })
                # end for
                templates_per_task.append({
                    "requirement": new_input_dicts[t_i]["requirement"]["description"],
                    "templates": template_res
                })
                print("REQ: ", new_input_dicts[t_i]["requirement"]["description"])
                print(f"#TEMPLATES: {len(template_res)}")
            # end for
            yield templates_per_task
        # end for
        return

    @classmethod
    def get_editor_template(cls, editor, template_dicts):
        suite = TestSuite()
        t = None
        for templates_per_req in template_dicts:
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
            suite.add(test, 'neutral words in context', 'Vocabulary', templates_per_req["requirement"])
        # end for
        return
    
if __name__=="__main__":
    for temp in Testmodel.get_templates():
        editor = Editor()
        Testmodel.get_editor_template(editor, temp)
        print(f"\n")
        
