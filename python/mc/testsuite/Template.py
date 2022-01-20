# This script is to generate new templates
# for testing given new generated inputs

from typing import *

import re, os
import nltk
import copy
import random
import numpy
import spacy

from pathlib import Path
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..requirement.Requirements import Requirements

from .Generator import Generator
from .Qgenerator import Qgenerator
from .Synonyms import Synonyms
from .Search import Search
from .Suggest import Suggest
from .cfg.CFGExpander import CFGExpander


class Template:

    POS_MAP = {
        "NNP": "NP",
        "NNPS": "NPS",
        "PRP": "PP",
        "PRP$": "PP$"
    }

    # SEARCH_MAP = {
    #     Macros.sa_task: Search.search_sentiment_analysis
    #     Macros.mc_task:
    #     Macros.qqp_task: 
    # }
    SEARCH_FUNC = {
        Macros.qqp_task: Search.search_qqp
    }
    
    @classmethod
    def generate_inputs(cls, task, dataset, n=None, save_to=None):
        cfg_ref_file = Macros.result_dir / 'treebank_cfg.json'
        print("Analyzing CFG ...")
        reqs = Requirements.get_requirements(task)
        results = list()
        if os.path.exists(save_to):
            # skip requirements that have been completed.
            results = Utils.read_json(save_to)
            _reqs = list()
            for req in reqs:
                if not any([True for r in results if r["requirement"]["description"]==req["description"]]):
                    _reqs.append(req)
                # end if
            # end for
            reqs = _reqs
        # end if
        for selected in cls.SEARCH_FUNC[task](reqs, dataset):
            exp_inputs = dict()
            print(f">>>>> REQUIREMENT:", selected["requirement"]["description"])
            num_selected_inputs = len(selected["selected_inputs"])
            print(f"\t{num_selected_inputs} inputs are selected.")
            index = 0
            num_seed_for_exp = 0
            for selected_sent in selected["selected_inputs"][:Macros.max_num_seeds]:
                index += 1
                _id, seed_q, seed_c, seed_a = selected_sent['id'], selected_sent['question'], selected_sent['context'], selected_sent['answers']
                print(f"\tSELECTED_Q {index}: {_id}, {seed_q}, {seed_a}")
                expander = CFGExpander(seed_input=seed_q, cfg_ref_file=cfg_ref_file)
                generator = Generator(expander=expander)
                gen_inputs = generator.masked_input_generator()
                new_input_results = list()
                if any(gen_inputs) and num_seed_for_exp<=n:
                    # get the word suggesteion at the expended grammar elements
                    gen_inputs = Suggest.get_new_inputs(
                        generator.editor,
                        gen_inputs,
                        num_target=Macros.num_suggestions_on_exp_grammer_elem
                    )
                    for g_i in range(len(gen_inputs)):
                        eval_results = Suggest.eval_word_suggest(gen_inputs[g_i], selected["requirement"])
                        if any(eval_results):
                            del gen_inputs[g_i]["words_suggest"]
                            new_input_results.extend(eval_results)
                            num_seed_for_exp += 1
                            print(".", end="")
                        # end if
                    # end for
                    print() 
                # end if
                    
                
                if any(questions[seed]):
                    exp_inputs[seed_q] = {
                        'cfg_seed': expander.cfg_seed,
                        'exp_inputs': new_input_results,
                        'context': seed_c,
                        'label': seed_a
                    }
                # end if
            # end for
            results.append({
                "requirement": selected["requirement"],
                "inputs": exp_inputs
            })
            # write raw new inputs for each requirement
            Utils.write_json(results, save_to, pretty_format=True)
            print(f"<<<<< REQUIREMENT:", selected["requirement"]["description"])
        # end for
        
        # # write raw new inputs
        # Utils.write_json(results, save_to, pretty_format=True)
        print(f"**********")        
        return results
    
    @classmethod
    def get_new_inputs(cls, input_file, nlp_task, dataset_name, n=None):
        # if os.path.exists(input_file):
        #     return Utils.read_json(input_file)
        # # end if
        return cls.generate_inputs(
            task=nlp_task,
            dataset=dataset_name,
            n=n,
            save_to=input_file
        )

    @classmethod
    def find_pos_from_cfg_seed(cls, token, cfg_seed):
        # when tokenized token can be found in the leaves of cfg
        for key, vals in cfg_seed.items():
            for val in vals:
                if val["pos"]==val["word"] and token==val["word"]:
                    return key
                # end if
            # end for
        # end for
        # when tokenized token cannot be found in the leaves of cfg
        for key, vals in cfg_seed.items():
            for val in vals:
                if type(val["word"])==list and token in val["word"]:
                    tok_idx = val["word"].index(token)
                    return val["pos"][tok_idx]
                # end if
            # end for
        # end for
        return
    
    @classmethod
    def get_token_n_pos(cls, nlp, input_pair):
        q1, q2 = input_pair[0], input_pair[1]
        doc1, doc2 = nlp(q1), nlp(q2)
        tokens_indices1, tokens_indices2 = list(), list()
        skip_synonym_indices1, skip_synonym_indices2 = list(), list()
        skip_synonyms = False
        for t_i, t in enumerate(doc1):            
            if str(t)=='<' and not skip_synonyms:
                skip_synonyms = True
            elif str(t)=='>' and skip_synonyms:
                skip_synonyms = False
            elif str(t)!='<' and str(t)!='>' and skip_synonyms:
                skip_synonym_indices1.append(t_i)
                tokens_indices1.append(t_i)
            elif str(t)!='<' and str(t)!='>' and not skip_synonyms:
                tokens_indices1.append(t_i)
            # end if
        # end for
        skip_synonyms = False
        for t_i, t in enumerate(doc2):
            if str(t)=='<' and not skip_synonyms:
                skip_synonyms = True
            elif str(t)=='>' and skip_synonyms:
                skip_synonyms = False
            elif str(t)!='<' and str(t)!='>' and skip_synonyms:
                skip_synonym_indices2.append(t_i)
                tokens_indices2.append(t_i)
            elif str(t)!='<' and str(t)!='>' and not skip_synonyms:
                tokens_indices2.append(t_i)
            # end if
        # end for

        tokens1, tokens2 = list(), list()
        tokens_pos1, tokens_pos2 = list(), list()
        for t_i, t in enumerate(doc1):
            if t_i in tokens_indices1 and t_i not in skip_synonyms_indices1:
                tokens1.append(str(t))
                tokens_pos1.append(t.tag_)
            elif t_i in tokens_indices1 and t_i in skip_synonyms_indices1:
                tokens1.append(f"{str(t)}_SKIP")
                tokens_pos1.append(t.tag_)
            # end if
        # end for
        for t_i, t in enumerate(doc2):
            if t_i in tokens_indices2 and t_i not in skip_synonyms_indices2:
                tokens2.append(str(t))
                tokens_pos2.append(t.tag_)
            elif t_i in tokens_indices2 and t_i in skip_synonyms_indices2:
                tokens2.append(f"{str(t)}_SKIP")
                tokens_pos2.append(t.tag_)
            # end if
        # end for
        return tokens1, tokens_pos1, tokens2, tokens_pos2
        
    @classmethod
    def get_templates_by_synonyms(cls, nlp, input_pair, prev_synonyms, is_seed_pair=False):
        template1, template2 = list(), list()
        tokens1, tokens_pos1, tokens2, tokens_pos2 = cls.get_token_n_pos(nlp, input_pair)
        
        for t, p in zip(tokens1, tokens_pos1):
            key = "{"+f"{t}_{p}"+"}"
            if t.endswith('_SKIP'):
                template1.append(t.split('_SKIP')[0])
            elif key in prev_synonyms.keys():
                if prev_synonyms[key] is None or len(prev_synonyms[key])==0:
                    template1.append(t)
                else:
                    template1.append({
                        key: prev_synonyms[key]
                    })
                # end if
            else:
                syns = Synonyms.get_synonyms(nlp, t, p)
                if len(syns)>1:
                    _syns = list()
                    for s in syns:
                        if len(s.split("_"))>1:
                            _syns.append(" ".join(s.split("_")))
                        else:
                            _syns.append(s)
                        # end if
                    # end for
                    syns_dict = {key: _syns}
                    template1.append(syns_dict)
                    if key not in prev_synonyms.keys():
                        prev_synonyms[key] = syns_dict[key]
                    # end if
                else:
                    template1.append(t)
                    if key not in prev_synonyms.keys():
                        prev_synonyms[key] = None
                    # end if
                # end if
            # end if
        # end for

        for t, p in zip(tokens2, tokens_pos2):
            key = "{"+f"{t}_{p}"+"}"
            if t.endswith('_SKIP'):
                template2.append(t.split('_SKIP')[0])
            elif key in prev_synonyms.keys():
                if prev_synonyms[key] is None or len(prev_synonyms[key])==0:
                    template2.append(t)
                else:
                    template2.append({
                        key: prev_synonyms[key]
                    })
                # end if
            else:
                syns = Synonyms.get_synonyms(nlp, t, p)
                if len(syns)>1:
                    _syns = list()
                    for s in syns:
                        if len(s.split("_"))>1:
                            _syns.append(" ".join(s.split("_")))
                        else:
                            _syns.append(s)
                        # end if
                    # end for
                    syns_dict = {key: _syns}
                    template2.append(syns_dict)
                    if key not in prev_synonyms.keys():
                        prev_synonyms[key] = syns_dict[key]
                    # end if
                else:
                    template2.append(t)
                    if key not in prev_synonyms.keys():
                        prev_synonyms[key] = None
                    # end if
                # end if
            # end if
        # end for
        return {
            "input": input_pair,
            "place_holder": (template1,template2)
        }, prev_synonyms

    @classmethod
    def get_templates(cls, num_seeds, nlp_task, dataset_name):
        assert nlp_task==Macros.qqp_task
        assert dataset_name in Macros.datasets
        print(f"***** TASK: {nlp_task}, SEARCH_DATASET: {dataset_name} *****")
        # Search inputs from searching dataset and expand the inputs using ref_cfg
        nlp = spacy.load('en_core_web_md')
        nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
        task = nlp_task
        
        new_input_dicts = cls.get_new_inputs(
            Macros.result_dir/f"cfg_expanded_inputs_{task}.json",
            task,
            dataset_name,
            n=num_seeds
        )

        # Make templates by synonyms
        print("Generate Templates ...")
        prev_synonyms = dict()
        for t_i in range(len(new_input_dicts)):
            # for each testing linguistic capabilities,
            inputs_per_req = new_input_dicts[t_i]
            req_cksum = Utils.get_cksum(inputs_per_req["requirement"]["description"])
            inputs = inputs_per_req["inputs"]
            print(f">>>>> REQUIREMENT:", inputs_per_req["requirement"]["description"])
            seed_inputs, seed_templates, exp_templates = list(), list(), list()
            for s_i, seed_input in enumerate(inputs.keys()):
                print(f"\tSEED {s_i}: {seed_input}")

                pair_search = re.search(r"([^\:]+)\:\:([^\:]+)", seed_input)
                if pair_search:
                    _input = (pair_search.group(1), pair_search.group(2))
                    cfg_seed1 = inputs[seed_input]['cfg_seed1']
                    cfg_seed2 = inputs[seed_input]['cfg_seed2']
                    label_seed = inputs[seed_input]['label']
                    questions = inputs[seed_input]['questions'] # dict
                    seed_inputs.append({
                        "input": _input,
                        "label": label_seed
                    })
                    
                    # make template for seed input
                    _templates, prev_synonyms = cls.get_templates_by_synonyms(nlp, _input, prev_synonyms, is_seed_pair=True)
                    _templates["label"] = label_seed
                    seed_templates.append(_templates)

                    # make template for generated questions from seed q pair, but not from exp questions
                    for qs_str in questions[seed_input]:
                        pair_search = re.search(r"([^\:]+)\:\:([^\:]+)", qs_str)
                        _input = (pair_search.group(1), pair_search.group(2))
                        _templates, prev_synonyms = cls.get_templates_by_synonyms(nlp, _input, prev_synonyms)
                        _templates["label"] = questions['label']
                        seed_templates.append(_templates)
                    # end for
                else:
                    _input = seed_input
                    cfg_seed = inputs[seed_input]['cfg_seed']
                    questions = inputs[seed_input]['questions']
                    # exp_inputs = inputs[seed_input]['exp_inputs']

                    # make template for generated questions from seed sentence
                    for key in questions.keys():
                        if key!='exp_inputs':
                            _input = (key, questions[key])
                            _templates, prev_synonyms = cls.get_templates_by_synonyms(nlp, _input, prev_synonyms)
                            _templates["label"] = questions['label']
                            seed_templates.append(_templates)
                        # end if
                    # end for

                    # make template for generated questions from exp sentences
                    if any(questions['exp_inputs']):
                        for key in questions['exp_inputs'].keys():
                            _input = (key, questions['exp_inputs'][key])
                            _templates, prev_synonyms = cls.get_templates_by_synonyms(nlp, _input, prev_synonyms)
                            _templates["label"] = questions['label']
                            exp_templates.append(_templates)
                            print(".", end="")
                        # end for
                        print()
                    # end if
                # end if
            # end for

            # Write the template results
            res_dir = Macros.result_dir/ f"templates_{task}"
            res_dir.mkdir(parents=True, exist_ok=True)
            
            Utils.write_json(seed_inputs,
                             res_dir / f"seeds_{req_cksum}.json",
                             pretty_format=True)
            Utils.write_json(seed_templates,
                             res_dir / f"templates_seed_{req_cksum}.json",
                             pretty_format=True)
            Utils.write_json(exp_templates,
                             res_dir / f"templates_exp_{req_cksum}.json",
                             pretty_format=True)
            print(f"<<<<< REQUIREMENT:", inputs_per_req["requirement"]["description"])
        # end for
        return

# Write templates
# Template.get_templates(num_seeds=10)
