# This script is to sample sentences
# from seed/exp sentences for pilot study

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
from ..utils.Logger import Logger


class Humanstudy:
    
    @classmethod
    def read_sentences(cls, json_file: Path):
        inputs = Utils.read_json(json_file)
        results = dict()
        for inp in inputs:
            req = inp['requirement']
            seeds, exps = list(), list()
            for seed in inp['inputs'].keys():
                seeds.append(seed)
                exps.extend([e[5] for e in inp['inputs'][seed]['exp_inputs']])
            # end for
            results[req] = {
                'seed': seeds,
                'exp': exp
            }
        # end for
        return results

    @classmethod
    def sample_sents(cls, sent_dict: Dict, num_samples=10):
        sample_results = dict()
        for req in sent_dict.keys():
            num_seed = len(sent_dict[req]['seed'])
            num_exp = len(sent_dict[req]['exp'])
            seed_ids = list(range(num_seed))
            exp_ids = list(range(num_exp))
            random.shuffle(seed_ids)
            random.shuffle(exp_ids)
            sample_results[req] = {
                'seed': [sent_dict[req]['seed'][idx] for idx in seed_ids[:num_samples]],
                'exp': [sent_dict[req]['exp'][idx] for idx in exp_ids[:num_samples]]
            }
        # end for
        return sample_results

    @classmethod
    def write_samples(cls, sample_dict: Dict):
        seeds, exps = list(), list()
        for req in sample_dict.keys():
            seeds.extend(sample_dict[req]['seed'])
            exps.extend(sample_dict[req]['exp'])
        # end for
        seed_res = ""
        exp_res = ""
        random.shuffle(seeds)
        random.shuffle(exps)
        for s in seeds:
            seed_res += f"{s}\n"
        # end for
        for e in exps:
            exp_res += f"{e}\n"
        # end for
        res_dir = Macros.result_dir / 'human_study'
        res_dir.mkdir(parents=True, exist_ok=True)
        Utils.write_txt(seed_res, res_dir / "seed_samples.txt")
        Utils.write_txt(exp_res, res_dir / "exp_samples.txt")
        return

    @classmethod
    def main(cls,
             nlp_task,
             search_dataset_name,
             selection_method):
        target_file = Macros.result_dir / f"cfg_expanded_inputs_{nlp_task}_{search_dataset_name}_{selection_method}.json"
        sent_dict = cls.read_sentences(target_file)
        sample_dict = cls.sample_sents(sent_dict, num_samples=5):
        cls.write_samples(sample_dict)
        return
