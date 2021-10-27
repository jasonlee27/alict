# This script is to add test templates to
# the checklist Testing model framework
# and test model.

from typing import *

import re, os
import nltk
import copy
import random
import numpy

from pathlib import Path
from nltk.tokenize import word_tokenize as tokenize

from Macros import Macros
from Utils import Utils
from Template import Template

class Template:

    @classmethod
    def get_templates(cls):
        for task in Macros.datasets.keys():
            new_input_dicts = cls.get_new_inputs(Macros.result_dir/f"cfg_expanded_inputs_{task}.json")
            templates_per_task = list()
            for t_i in range(len(new_input_dicts)):
                req_cksum = Utils.get_cksum(new_input_dicts[t_i]["requirement"]["description"])
                res_dir = Macros.result_dir/ f"templates_{task}"
                templates = Utils.read_json(res_dir / f"templates_{req_cksum}.json")
                template_res = list()
                template_list = list()
                template_values = dict()
                for tp in templates:
                    input_sent = tp["input"]
                    for tok_i in range(len(tp["place_holder"])):
                        if type(tp["place_holder"][tok_i])==str:
                            template_list.append(tp["place_holder"][tok_i])
                        elif type(tp["place_holder"][tok_i])==dict:
                            key = tp["place_holder"][tok_i].keys()[0]
                            template_list.append(key)
                            template_values[key] = tp["place_holder"][tok_i][key]
                        # end if
                    # end for
                    template_res.append({
                        "sent": " ".join(template_list),
                        "values": template_values
                    })
                    print(template_res)
                # end for
                templates_per_task.append({
                    "requirement": new_input_dicts[t_i]["requirement"]["description"],
                    "templates": templates_res
                })
            # end for
            yield templates_per_task
        # end for
        return
