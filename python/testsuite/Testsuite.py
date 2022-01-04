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
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from .Search import Search
from .Transform import TransformOperator
from .Template import Template

random.seed(Macros.SEED)

class Testsuite:

    @classmethod
    def map_labels(cls, task: str, label):
        if task==Macros.sa_task:
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
            too_many_synonym_keys = [key for key in placeholder_keys if len(lexicon_dict[key])>Macros.max_num_synonyms]
            # random.shuffle(too_many_synonym_keys)
            for key in too_many_synonym_keys:
                lexicon_dict[key] = lexicon_dict[key][:Macros.max_num_synonyms]
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
            t = editor.template(template["sent"],
                                labels=template["label"],
                                save=True)
        elif t is not None and len(t.data)<Macros.max_num_sents:
            t += editor.template(template["sent"],
                                 labels=template["label"],
                                 save=True)
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
            "sent": " ".join(template_list),
            "values": template_values,
            "label": cls.map_labels(task, template["label"])
        }

    @classmethod
    def get_templates(cls, nlp_task, dataset, num_seeds):
        task = nlp_task
        print(f"TASK: {task}")
        new_input_dicts = Template.get_new_inputs(
            Macros.result_dir/f"cfg_expanded_inputs_{task}.json",
            task,
            dataset,
            n=num_seeds
        )
        seeds_per_task = list()
        seed_templates_per_task = list()
        exp_templates_per_task = list()
        transform_reqs = list()
        for t_i in range(len(new_input_dicts)):
            req_cksum = Utils.get_cksum(new_input_dicts[t_i]["requirement"]["description"])
            # print(new_input_dicts[t_i]["requirement"]["capability"]+"::"+new_input_dicts[t_i]["requirement"]["description"])
            res_dir = Macros.result_dir/ f"templates_{task}"
            if (not os.path.exists(str(res_dir / f"seeds_{req_cksum}.json"))) or \
               (not os.path.exists(str(res_dir / f"templates_seed_{req_cksum}.json"))) or \
               (not os.path.exists(str(res_dir / f"templates_exp_{req_cksum}.json"))):
                Template.get_templates(
                    nlp_task=nlp_task,
                    dataset=dataset,
                    num_seeds=num_seeds
                )
            # end if

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
        yield task, seeds_per_task, seed_templates_per_task, exp_templates_per_task, transform_reqs
        return

    @classmethod
    def suite_add_transform(cls,
                            editor,
                            sentences,
                            transform_req,
                            templates_per_req,
                            task, dataset, suite, seed_type):
        transformer = TransformOperator(editor,
                                        templates_per_req['capability'],
                                        templates_per_req['description'],
                                        transform_req,
                                        nlp_task=task,
                                        search_dataset=dataset)
        test_type, func, _property, woi = transformer.transformation_funcs.split('_')
        if test_type=="INV":
            if func=="replace" and woi=="phrase":
                if task==Macros.sa_task:
                    if len(sentences)>Macros.max_num_sents_for_perturb: sentences = sentences[:Macros.max_num_sents_for_perturb]
                    t = Perturb.perturb(sentences, transformer.replace, nsamples=Macros.nsamples)
                    test = INV(t.data)
                    suite.add(test,
                              name=f"{task}::{seed_type}::"+templates_per_req["description"],
                              capability=templates_per_req["capability"]+f"::{seed_type}",
                              description=templates_per_req["description"])
                # end if
            elif func=="add" and _property=="random" and woi=="URL_handles":
                if task==Macros.sa_task:
                    # we set the maximum number of sentences not to kill process during perturbation.
                    if len(sentences)>Macros.max_num_sents_for_perturb: sentences = sentences[:Macros.max_num_sents_for_perturb]
                    t = Perturb.perturb(sentences, transformer.add_irrelevant, nsamples=Macros.nsamples)
                    test = INV(t.data)
                    suite.add(test,
                              name=f"{task}::{seed_type}::"+templates_per_req["description"],
                              capability=templates_per_req["capability"]+"::{seed_type}",
                              description=templates_per_req["description"])
                # end if
            elif func=="strip" and _property=="None" and woi=="puncuation":
                if task==Macros.sa_task:
                    if len(sentences)>Macros.max_num_sents_for_perturb: sentences = sentences[:Macros.max_num_sents_for_perturb]
                    nlp = spacy.load('en_core_web_sm')
                    parsed_data = list(nlp.pipe(sentences))
                    t = Perturb.perturb(parsed_data, Perturb.punctuation, nsamples=Macros.nsamples)
                    test = INV(t.data)
                    suite.add(test,
                              name=f"{task}::{seed_type}::"+templates_per_req["description"],
                              capability=templates_per_req["capability"]+"::{seed_type}",
                              description=templates_per_req["description"])
                # end if
            elif func=="swap" and _property=="one" and woi=="two_adjacent_characters":
                if task==Macros.sa_task:
                    if len(sentences)>Macros.max_num_sents_for_perturb: sentences = sentences[:Macros.max_num_sents_for_perturb]
                    t = Perturb.perturb(sentences, Perturb.add_typos, nsamples=Macros.nsamples, typos=1)
                    test = INV(t.data)
                    suite.add(test,
                              name=f"{task}::{seed_type}::"+templates_per_req["description"],
                              capability=templates_per_req["capability"]+"::{seed_type}",
                              description=templates_per_req["description"])
                # end if
            elif func=="swap" and _property=="two" and woi=="two_adjacent_characters":
                if task==Macros.sa_task:
                    if len(sentences)>Macros.max_num_sents_for_perturb: sentences = sentences[:Macros.max_num_sents_for_perturb]
                    t = Perturb.perturb(sentences, Perturb.add_typos, nsamples=Macros.nsamples, typos=2)
                    test = INV(t.data)
                    suite.add(test,
                              name=f"{task}::{seed_type}::"+templates_per_req["description"],
                              capability=templates_per_req["capability"]+"::{seed_type}",
                              description=templates_per_req["description"])
                # end if
            elif func=="contract/expand" and _property=="None" and woi=="contraction":
                if task==Macros.sa_task:
                    if len(sentences)>Macros.max_num_sents_for_perturb: sentences = sentences[:Macros.max_num_sents_for_perturb]
                    t = Perturb.perturb(sentences, Perturb.contractions, nsamples=2*Macros.nsamples)
                    test = INV(t.data)
                    suite.add(test,
                              name=f"{task}::{seed_type}::"+templates_per_req["description"],
                              capability=templates_per_req["capability"]+"::{seed_type}",
                              description=templates_per_req["description"])
                # end if
            # end if
        elif test_type=="DIR":
            if func=="add" and woi=="phrase":
                if task==Macros.sa_task:
                    if len(sentences)>Macros.max_num_sents_for_perturb: sentences = sentences[:Macros.max_num_sents_for_perturb]
                    print(transform_req, len(sentences))
                    phrases = [ _s[1]
                        for s in Search.search_sentiment_analysis(
                                transformer.search_reqs,
                                transformer.search_dataset
                        ) for _s in s["selected_inputs"]
                    ][:5]
                    print(transform_req, len(sentences))
                    nlp = spacy.load('en_core_web_sm')
                    parsed_data = list(nlp.pipe(sentences))
                    print(transform_req, len(sentences))
                    t = Perturb.perturb(parsed_data, transformer.add_phrase(phrases), nsamples=Macros.nsamples)
                    print(transform_req, len(sentences))
                    test = DIR(t.data, transformer.dir_expect_func)
                    print(transform_req, len(sentences))
                    suite.add(test,
                              name=f"{task}::{seed_type}::"+templates_per_req["description"],
                              capability=templates_per_req["capability"]+"::{seed_type}",
                              description=templates_per_req["description"])
                    print("@@@@@@@@@@@")
                # end if
            # end if
        # end if
        return suite

    @classmethod
    def write_seed_testsuite(cls,
                             task,
                             dataset,
                             seed_dicts,
                             transform_reqs,
                             res_dir):
        for t_i, templates_per_req in enumerate(seed_dicts):
            print(f"{task}::SEED::"+templates_per_req["description"])
            t = None
            suite = TestSuite()
            editor = Editor()
            for template in templates_per_req["templates"]:
                t = cls.add_template(t, editor, template)
            # end for
            if transform_reqs[t_i] is not None:
                suite = cls.suite_add_transform(editor,
                                                t.data,
                                                transform_reqs[t_i],
                                                templates_per_req,
                                                task, dataset, suite, "SEED")
            else:
                test = MFT(**t)
                suite.add(test,
                          name=f"{task}::SEED::"+templates_per_req["description"],
                          capability=templates_per_req["capability"]+"::SEED",
                          description=templates_per_req["description"])
            # end if
            test_cksum = Utils.get_cksum(
                task+templates_per_req["capability"]+templates_per_req["description"]
            )
            suite.save(res_dir / f'{task}_testsuite_seeds_{test_cksum}.pkl')
        # end for
        return

    @classmethod
    def write_seed_template_testsuite(cls,
                                      task,
                                      dataset,
                                      seed_template_dicts,
                                      transform_reqs,
                                      res_dir):
        for t_i, templates_per_req in enumerate(seed_template_dicts):
            print(f"{task}::SEED_TEMPS::"+templates_per_req["description"])
            t = None
            suite = TestSuite()
            editor = Editor()
            for template in templates_per_req["templates"]:
                editor = cls.add_lexicon(editor, template["values"])
                print(editor)
                t = cls.add_template(t, editor, template)
                print(len(t.data))
            # end for
            if transform_reqs[t_i] is not None:
                suite = cls.suite_add_transform(editor,
                                                t.data,
                                                transform_reqs[t_i],
                                                templates_per_req,
                                                task, dataset, suite, "SEED_TEMPS")
            else:
                test = MFT(**t)
                suite.add(test,
                          name=f"{task}::SEED_TEMPS::"+templates_per_req["description"],
                          capability=templates_per_req["capability"]+"::SEED_TEMPS",
                          description=templates_per_req["description"])
            # end if
            test_cksum = Utils.get_cksum(
                task+templates_per_req["capability"]+templates_per_req["description"]
            )
            suite.save(res_dir / f'{task}_testsuite_seed_templates_{test_cksum}.pkl')
        # end for
        return

    @classmethod
    def write_exp_template_testsuite(cls,
                                     task,
                                     dataset,
                                     exp_template_dicts,
                                     transform_reqs,
                                     res_dir):
        for t_i, templates_per_req in enumerate(exp_template_dicts):
            print(f"{task}::EXP_TEMPS::"+templates_per_req["description"])
            if any(templates_per_req["templates"]):
                t = None
                suite = TestSuite()
                editor = Editor()
                for template in templates_per_req["templates"]:
                    editor = cls.add_lexicon(editor, template["values"])
                    t = cls.add_template(t, editor, template)
                # end for
            
                if transform_reqs[t_i] is not None:
                    suite = cls.suite_add_transform(editor,
                                                    t.data,
                                                    transform_reqs[t_i],
                                                    templates_per_req,
                                                    task, dataset, suite, "EXP_TEMPS")
                else:
                    test = MFT(**t)
                    suite.add(test, 
                              name=f"{task}::EXP_TEMPS::"+templates_per_req["description"],
                              capability=templates_per_req["capability"]+"::EXP_TEMPS",
                              description=templates_per_req["description"])
                # end if
                test_cksum = Utils.get_cksum(
                    task+templates_per_req["capability"]+templates_per_req["description"]
                )
                suite.save(res_dir / f'{task}_testsuite_exp_templates_{test_cksum}.pkl')
            # end for
        # end for
        return
        
    @classmethod
    def write_editor_templates(cls,
                               task,
                               dataset,
                               seed_dicts,
                               seed_template_dicts,
                               exp_template_dicts,
                               transform_reqs):
        res_dir = Macros.result_dir / "test_results"
        res_dir.mkdir(parents=True, exist_ok=True)
        cls.write_seed_testsuite(task,
                                 dataset,
                                 seed_dicts,
                                 transform_reqs,
                                 res_dir)
        cls.write_seed_template_testsuite(task,
                                          dataset,
                                          seed_template_dicts,
                                          transform_reqs,
                                          res_dir)
        cls.write_exp_template_testsuite(task,
                                         dataset,
                                         exp_template_dicts,
                                         transform_reqs,
                                         res_dir)
        return

    @classmethod
    def write_testsuites(cls, nlp_task, dataset, num_seeds):
        for task, seed, seed_temp, exp_temp, transform_reqs in cls.get_templates(nlp_task=nlp_task, dataset=dataset, num_seeds=num_seeds):
            Testsuite.write_editor_templates(task, dataset, seed, seed_temp, exp_temp, transform_reqs)
        # end for
        return


# if __name__=="__main__":
#     Testsuite.write_testsuites()
