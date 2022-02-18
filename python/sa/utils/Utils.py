
import re, os
import sys
import json
import string
import hashlib

from pathlib import Path
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
    def tokenize(cls, sent: str)->list:
        return word_tokenize(sent)

    @classmethod
    def detokenize(cls, tokens: list)->str:
        tokens = ['"' if (t=='``' or t=='\'\'') else t for t in tokens]
        sent = TreebankWordDetokenizer().detokenize(tokens)
        sent = re.sub(r"(.+)\-\-(.+)", r"\1 -- \2", sent)
        sent = re.sub(r"(.+)\.\.\.(.+)", r"\1 ... \2", sent)
        return sent

    @classmethod
    def read_txt(cls, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        #end with
        return lines

    @classmethod
    def read_json(cls, json_file):
        # read cfg json file
        if os.path.exists(str(json_file)):
            with open(json_file, 'r') as f:
                return json.load(f)
            # end with
        # end if
        return None

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
        if task==Macros.sa_task:
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
