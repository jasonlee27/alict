# This script is to fill mask token using huggingface pipeline
# loading bert-base-uncased model

from typing import *

import re, os
import nltk
import spacy
import copy
import time
import random
import numpy as np

from pathlib import Path
from scipy.special import softmax

# from transformers import pipeline
from transformers import BertTokenizer, BertModel

class Fillmask:

    checklist_mask_token = '{mask}'
    huggingface_mask_token = '[MASK]'
    unmasker_model_name = 'bert-base-uncased'

    @classmethod
    def replace_mask_token(cls, masked_sent):
        return masked_sent.replace(cls.checklist_mask_token,
                                   cls.huggingface_mask_token)

    @classmethod
    def load_unmasker(cls):
        tokenizer = BertTokenizer.from_pretrained(cls.unmasker_model_name)
        model = BertModel.from_pretrained(cls.unmasker_model_name)
        return model, tokenizer

    @classmethod
    def suggest(cls, masked_sents):
        # out format example
        # {'sequence': "[CLS] hello i'm a fashion model. [SEP]",
        #  'score': 0.1073106899857521,
        #  'token': 4827,
        #  'token_str': 'fashion'}
        if type(masked_sents)==str:
            out = unmasker(masked_sents)
        elif type(masked_sents)==list:
                
        
    
