from typing import *
from pathlib import Path

import re, os
import sys
import json
import string
import hashlib
import statistics
import contractions

from nltk.corpus import treebank
from checklist.test_suite import TestSuite
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from .Macros import Macros

class Utils:

    @classmethod
    def argparse(cls):
        if len(sys.argv)>1:
            arguments = sys.argv[1:]
            arg_dict = dict()
            for arg_i in range(0, len(arguments), 2):
                if arguments[arg_i].startswith('--'):
                    key = arguments[arg_i][2:]
                    val = arguments[arg_i+1]
                    arg_dict[key] = val
                else:
                    raise("Invalid argument format!")
                # end if
            # end for
            return arg_dict
        # end if
        return

    @classmethod
    def fix_contractions(cls, sent):
        _sent = contractions.fix(sent)
        _sent = re.sub(r" is been ", r" has been ", _sent)
        return _sent
        
    @classmethod
    def tokenize(cls, sent: str)->list:
        return word_tokenize(sent)

    @classmethod
    def detokenize(cls, tokens: list)->str:
        tokens = ['"' if (t=='``' or t=='\'\'') else t for t in tokens]
        sent = TreebankWordDetokenizer().detokenize(tokens)
        sent = re.sub(r"(.+)\-\-(.+)", r"\1 -- \2", sent)
        sent = re.sub(r"(.+)\.\.\.(.+)", r"\1 ... \2", sent)
        sent = cls.fix_contractions(sent)
        return sent

    @classmethod
    def read_txt(cls, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        #end with
        return lines

    @classmethod
    def write_txt(cls, input_str, data_file):
        with open(data_file, 'w') as f:
            lines = f.write(input_str)
        #end with
        return lines

    @classmethod
    def read_json(cls, json_file):
        # read cfg json file
        if os.path.exists(str(json_file)):
            with open(json_file, 'r') as f:
                try:
                    res = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    res = None
                # end try
                return res
            # end with
        # end if
        return

    @classmethod
    def read_testsuite(cls, testsuite_file):
        def example_to_dict_fn(data):
            return { 'text': data }
        # read checklist testsuite file (.pkl)
        suite = TestSuite()
        tsuite = suite.from_file(testsuite_file)
        tsuite_dict = tsuite.to_dict(example_to_dict_fn=example_to_dict_fn)
        return tsuite, tsuite_dict

    @classmethod
    def read_sv(cls, sv_file, delimeter=',', is_first_attributes=True):
        # read comma/tab-seperated valued file (.csv and .tsv)
        if os.path.exists(str(sv_file)):
            with open(sv_file, 'r') as f:
                if is_first_attributes:
                    lines = f.readlines()
                    return {
                        'attributes': [att.strip() for att in lines[0].split(delimeter)],
                        'lines': [l.strip().split(delimeter) for l in lines[1:]]
                    }
                # end if
                return [l.split(delimeter) for l in f.readlines()]
            # end with
        # end if
        return
    
    @classmethod
    def write_json(cls, input_dict, json_file, pretty_format=False):
        with open(json_file, 'w') as f:
            if pretty_format:
                json.dump(input_dict, f, indent=4)
            else:
                json.dump(input_dict, f)
            # end if
        # end with
        return

    @classmethod
    def get_cksum(cls, input_str: str, length=7):
        return hashlib.md5(input_str.encode('utf-8')).hexdigest()[:length]

    @classmethod
    def compute_failure_rate(cls, task, preds, target):
        if task==Macros.hs_task:
            pred_res = [True if p!=t else False for p,t in zip(preds, targets)]
            fail_cnt = sum(pred_res)
            fail_rate = round(fail_cnt*1. / len(pred_res), 2)
            return fail_cnt, fail_rate
        # end if

    @classmethod
    def replace_non_english_letter(cls, sent):
        _sent = sent.replace("-LRB-", "(")
        _sent = _sent.replace("-RRB-", ")")
        _sent = _sent.replace("Ã´", "ô")
        _sent = _sent.replace("8Â 1\/2", "8 1\/2")
        _sent = _sent.replace("2Â 1\/2", "2 1\/2")
        _sent = _sent.replace("Ã§", "ç")
        _sent = _sent.replace("Ã¶", "ö")
        _sent = _sent.replace("Ã»", "û")
        _sent = _sent.replace("Ã£", "ã")        
        _sent = _sent.replace("Ã¨", "è")
        _sent = _sent.replace("Ã¯", "ï")
        _sent = _sent.replace("Ã±", "ñ")
        _sent = _sent.replace("Ã¢", "â")
        _sent = _sent.replace("Ã¡", "á")
        _sent = _sent.replace("Ã©", "é")
        _sent = _sent.replace("Ã¦", "æ")
        _sent = _sent.replace("Ã­", "í")
        _sent = _sent.replace("Ã³", "ó")
        _sent = _sent.replace("Ã¼", "ü")
        _sent = _sent.replace("Ã ", "à")
        _sent = _sent.replace("Ã", "à")
        return _sent

    @classmethod
    def replace_abbreviation(cls, sent):
        pass

    @classmethod
    def is_a_in_x(cls, a_list, x_list):
        for i in range(len(x_list) - len(a_list) + 1):
            if a_list == x_list[i:i+len(a_list)]:
                return True
            # end if
        # end for
        return False            

    @classmethod
    def copy_list_of_dict(cls, data_list):
        import copy
        return [
            copy.deepcopy(d)
            for d in data_list
        ]
  
    @classmethod
    def avg(cls, nums: list, decimal=3):
        return str(round(sum(nums) / len(nums), decimal))

    @classmethod
    def median(cls, nums: list, decimal=3):
        return str(round(statistics.median(nums), decimal))

    @classmethod
    def stdev(cls, nums: list, decimal=3):
        if len(nums)==1:
            return str(0.)
        # end if
        return str(round(statistics.stdev(nums), decimal))

    @classmethod
    def lod_to_dol(cls, list_of_dict: List[dict]) -> Dict[Any, List]:
        """
        Converts a list of dict to a dict of list.
        """
        keys = set.union(*[set(d.keys()) for d in list_of_dict])
        return {k: [d.get(k) for d in list_of_dict] for k in keys}

