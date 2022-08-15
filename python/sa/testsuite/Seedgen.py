# This script is to generate new templates
# for testing given new generated inputs

from typing import *

import re, os
import nltk
import copy
# import random
import numpy
import spacy

from pathlib import Path
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from checklist.editor import Editor
from checklist.test_suite import TestSuite

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger
from ..requirement.Requirements import Requirements

from .cfg.RefPCFG import RefPCFG
from .Generator import Generator
from .Synonyms import Synonyms
from .Search import Search
from .Suggest import Suggest


class Seedgen:

    SEARCH_FUNC = {
        Macros.sa_task: Search.search_sentiment_analysis
    }


    @classmethod
    def add_template(cls, t, editor, seed_dict, num_seeds):
        for s in seed_dict['seeds']:
            _id, seed, seed_label = s
            if t is None:
                if callable(seed_label):
                    t = editor.template(seed,
                                        save=True)
                else:
                    t = editor.template(seed,
                                        labels=seed_label,
                                        save=True)
                # end if
            else:
                if (num_seeds>0 and len(t.data)<num_seeds) or \
                   num_seeds<0:
                    if callable(seed_dict["label"]):
                        t += editor.template(seed,
                                             save=True)
                    else:
                        t += editor.template(seed,
                                             labels=seed_label,
                                             save=True)
                    # end if
                # end if
            # end if
        # end for
        return t
    
    @classmethod
    def write_seed_testsuite(cls,
                             task,
                             dataset,
                             seed_dict,
                             num_seeds,
                             res_dir,
                             logger):
        test_cksum = Utils.get_cksum(seed_dict['requirement']['description'])
        if not os.path.exists(str(res_dir / f"{task}_testsuite_seeds_{test_cksum}.pkl")):
            logger.print(f"{task}::SEED::<"+seed_dict['requirement']['description']+f">::{test_cksum}::", end='')
            t = None
            suite = TestSuite()
            editor = Editor()
            t = cls.add_template(t, editor, seed_dict, num_seeds)
                
            if callable(templates_per_req["templates"][0]['label']):
                test = MFT(t.data, Expect.single(templates_per_req["templates"][0]['label']), templates=t.templates)
            else:
                test = MFT(**t)
            # end if
            suite.add(test,
                      name=f"{task}::SEED::"+seed_dict['requirement']['description'],
                      capability=seed_dict['requirement']["capability"]+"::SEED",
                      description=seed_dict['requirement']["description"])
            num_data = sum([len(suite.tests[k].data) for k in suite.tests.keys()])
            if num_data>0:
                suite.save(res_dir / f"{task}_testsuite_seeds_{test_cksum}.pkl")
                logger.print('SAVED')
            else:
                logger.print('NO_DATA')
            # end if
        # end if
        return
    
    @classmethod
    def generate_inputs(cls, task, dataset, n=Macros.max_num_seeds, save_to=None, selection_method=None, logger=None):
        reqs = Requirements.get_requirements(task)
        results = list()
        if os.path.exists(save_to):
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
            print_str = '>>>>> REQUIREMENT:'+selected["requirement"]["description"]
            num_selected_inputs = len(selected["selected_inputs"])
            logger.print(f"{print_str}\n\t{num_selected_inputs} inputs are selected.")
            index = 1            
            seeds = selected["selected_inputs"][:n] if n>0 else selected["selected_inputs"]
            seed_inputs = list()
            for _id, seed, seed_label, seed_score in seeds:
                logger.print(f"\tSELECTED_SEED {index}: {_id}, {seed}, {seed_label}, {seed_score} :: ", end='')
                index += 1
                seed_inputs.append([_id, seed, seed_label])
            # end for
            seed_dict = {
                'requirement': selected['requirement'],
                'num_seeds': len(seeds)
                'seeds': seed_inputs
            }
            results.append(seed_dict)
            
            # write raw new inputs for each requirement
            Utils.write_json(results, save_to, pretty_format=True)
            
            # write seed inputs into checklist testsuite format
            res_dir = Macros.result_dir/ f"seeds_{task}_{dataset}"
            res_dir.mkdir(parents=True, exist_ok=True)
            cls.write_seed_testsuite(task, dataset, seed_dict, n, rs_dir, logger)
            
            print_str = '<<<<< REQUIREMENT:'+selected["requirement"]["description"]
            logger.print(print_str)
        # end for
        
        # write seed inputs
        Utils.write_json(results, save_to, pretty_format=True)
        logger.print('**********')
        return results
    
    @classmethod
    def get_new_inputs(cls, input_file, nlp_task, dataset_name, n=None, selection_method=None, logger=None):
        # if os.path.exists(input_file):
        #     return Utils.read_json(input_file)
        # # end if
        return cls.generate_inputs(
            task=nlp_task,
            dataset=dataset_name,
            n=n,
            save_to=input_file,
            selection_method=selection_method,
            logger=logger
        )

    @classmethod
    def get_seeds(cls,
                  num_seeds,
                  nlp_task,
                  dataset_name,
                  log_file):
        assert nlp_task in Macros.nlp_tasks
        assert dataset_name in Macros.datasets[nlp_task]
        # Write the template results
        logger = Logger(logger_file=log_file,
                        logger_name='template')
        logger.print(f"***** TASK: {nlp_task}, SEARCH_DATASET: {dataset_name} *****")
        # Search inputs from searching dataset and expand the inputs using ref_cfg
        nlp = spacy.load('en_core_web_md')
        nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
        task = nlp_task
        new_input_dicts = cls.get_new_inputs(
            Macros.result_dir/f"seed_inputs_{task}_{dataset_name}.json",
            task,
            dataset_name,
            n=num_seeds,
            selection_method=selection_method,
            logger=logger
        )
        return


# Write templates
# Template.get_templates(num_seeds=10)
