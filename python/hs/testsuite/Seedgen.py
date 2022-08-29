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
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect

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
    def map_labels(cls, label):
        if type(label)==list:
            label_not= [v for k, v in Macros.sa_label_map.items() if k not in label]
            is_not_label = lambda x, pred, *args: pred != label_not[0]
            return is_not_label
        # end if
        return Macros.sa_label_map[label]

    @classmethod
    def add_template(cls, t, editor, seed_obj, num_seeds):
        _id, seed, seed_label = seed_obj
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
                if callable(seed_label):
                    t += editor.template(seed,
                                         save=True)
                else:
                    t += editor.template(seed,
                                         labels=seed_label,
                                         save=True)
                # end if
            # end if
        # end if
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
            for s in seed_dict['seeds']:
                _id, seed, seed_label = s
                _seed_label = cls.map_labels(seed_label)
                t = cls.add_template(t, editor,
                                     [_id, seed, _seed_label],
                                     num_seeds)
            # end for
            if callable(_seed_label):
                test = MFT(t.data, Expect.single(_seed_label), templates=t.templates)
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
        return test_cksum
    
    @classmethod
    def generate_inputs(cls,
                        task,
                        dataset,
                        save_to,
                        num_seeds=-1,
                        logger=None):
        reqs = Requirements.get_requirements(task)
        cksum_map_str = ""
        if os.path.exists(save_to):
            _reqs = list()
            for req in reqs:
                req_cksum = Utils.get_cksum(req['description'])
                cksum_map_str += f"{req['description']}\t{req_cksum}\n"
                if not os.path.exists(str(save_to / f"seed_{req_cksum}.json")):
                    _reqs.append(req)
                # end if
            # end for
            reqs = _reqs
        # end if

        for selected in cls.SEARCH_FUNC[task](reqs, dataset):
            exp_inputs = dict()
            num_selected_inputs = len(selected["selected_inputs"])
            lc_cksum = Utils.get_cksum(selected['requirement']['description'])
            print_str = f">>>>> REQUIREMENT_{lc_cksum}:"+selected['requirement']['description']
            logger.print(f"{print_str}\n\t{num_selected_inputs} inputs are selected.")
            index = 1
            seeds = selected["selected_inputs"][:num_seeds] if num_seeds>0 else selected["selected_inputs"]
            seed_inputs = list()
            for _id, seed, seed_label, seed_score in seeds:
                logger.print(f"\tSELECTED_SEED {index}: {_id}, {seed}, {seed_label}, {seed_score} :: ", end='\n')
                index += 1
                seed_inputs.append([_id, seed, seed_label])
            # end for
            seed_dict = {
                'requirement': selected['requirement'],
                'num_seeds': len(seeds),
                'seeds': seed_inputs
            }
            
            # write seed inputs into checklist testsuite format
            cksum_val = cls.write_seed_testsuite(task, dataset, seed_dict, num_seeds, save_to, logger)
            cksum_map_str += f"{selected['requirement']['description']}\t{cksum_val}\n"

            # write raw new inputs for each requirement
            Utils.write_json(seed_dict, save_to / f"seed_{lc_cksum}.json", pretty_format=True)
            
            print_str = '<<<<< REQUIREMENT:'+selected["requirement"]["description"]
            logger.print(print_str)
        # end for
        
        # write seed cksum
        # Utils.write_json(results, save_to, pretty_format=True)
        Utils.write_txt(cksum_map_str, save_to / 'cksum_map.txt')
        logger.print('**********')
        return
    
    @classmethod
    def get_new_inputs(cls, input_dir, nlp_task, dataset_name, num_seeds=None, logger=None):
        # if os.path.exists(input_file):
        #     return Utils.read_json(input_file)
        # # end if
        return cls.generate_inputs(
            task=nlp_task,
            dataset=dataset_name,
            num_seeds=num_seeds,
            save_to=input_dir,
            logger=logger
        )

    # @classmethod
    # def get_seeds(cls,
    #               nlp_task,
    #               dataset_name,
    #               num_seeds,
    #               num_trials,
    #               log_file=None):
    #     assert nlp_task in Macros.nlp_tasks
    #     assert dataset_name in Macros.datasets[nlp_task]
    #     if log_file is not None:
    #         # Write the template results
    #         logger = Logger(logger_file=log_file,
    #                         logger_name='template')
    #         logger.print(f"***** TASK: {nlp_task}, SEARCH_DATASET: {dataset_name} *****")
    #     # end if
    #     _num_trials = '' if num_trials==1 else str(num_trials)
    #     if num_seeds<0:
    #         cfg_res_file_name = f"cfg_expanded_inputs{_num_trials}_{nlp_task}_{dataset_name}_{selection_method}.json"
    #         seed_res_file_name = f"seed_inputs{_num_trials}_{task}_{dataset_name}.json"
    #     else:
    #         cfg_res_file_name = f"cfg_expanded_inputs{_num_trials}_{nlp_task}_{dataset_name}_{selection_method}_{num_seeds}seeds.json"
    #         seed_res_file_name = f"seed_inputs{_num_trials}_{task}_{dataset_name}_{num_seeds}seeds.json"
    #     # end if
    #     cfg_results = Utils.read_json(Macros.result_dir / cfg_res_file_name)
    #     # seed_inputs.append([_id, seed, seed_label])
    #     seed_results = dict()
    #     for cfg_res in cfg_results:
    #         lc = cfg_res['requirement']['description']
    #         seed_inputs = list()
    #         index = 0
    #         for seed in cfg_res['inputs'].keys():
    #             seed_label = cfg_res['inputs'][seed]['label']
    #             seed_inputs.append([index, seed, seed_label])
    #             index += 1
    #         # end for
    #         seed_results[seed] = seed_inputs
            
    #         # write raw new inputs for each requirement
    #         Utils.write_json(seed_results,
    #                          Macros.result_dir / seed_res_file_name,
    #                          pretty_format=True)
    #     # end for
    #     return 
            
    
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
        if num_seeds<0:
            seed_res_dir_name = f"seeds_{task}_{dataset_name}"
            # seed_res_file_name = f"seed_inputs_{task}_{dataset_name}.json"
        else:
            seed_res_dir_name = f"seeds_{task}_{dataset_name}_{num_seeds}seeds"
            # seed_res_file_name = f"seed_inputs_{task}_{dataset_name}_{num_seeds}seeds.json"
        # end if
        res_dir = Macros.result_dir / seed_res_dir_name
        res_dir.mkdir(parents=True, exist_ok=True)
        new_input_dicts = cls.get_new_inputs(
            res_dir,
            task,
            dataset_name,
            num_seeds=num_seeds,
            logger=logger
        )
        return


# Write templates
# Template.get_templates(num_seeds=10)
