
import re, os
import sys
import json
from nltk.corpus import treebank

from Macros import Macros

class Utils:

    @classmethod
    def read_txt(cls, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        #end with
        return lines

    @classmethod
    def read_json(cls, json_file):
        # read cfg json file
        with open(json_file, 'r') as f:
            return json.load(f)
        # end with
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