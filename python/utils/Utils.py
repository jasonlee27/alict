
import re, os
import sys
import json
import hashlib

from pathlib import Path
from nltk.corpus import treebank

from .Macros import Macros

class Utils:

    @classmethod
    def argparse(cls):
        if len(sys.argv)>1:
            arguments = sys.argv[1:]
            arg_dict = dict()
            for arg in arguments:
                if arg.startswith('--'):
                    arg = arg[2:]
                    key, val = arg.split("=")[0], arg.split("=")[1]
                    arg_dict[key] = val
                else:
                    raise("Invalid argument format!")
                # end if
            # end for
            return arg_dict
        # end if
        return

    @classmethod
    def read_txt(cls, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        #end with
        return lines

    @classmethod
    def read_json(cls, json_file):
        # read cfg json file
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                return json.load(f)
            # end with
        # end if
        return None

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
