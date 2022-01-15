# This script perturb and transform inputs in datasets that meet requirements

from typing import *

import re, os
import sys
import json
import spacy
import random
import string
import checklist
import numpy as np

# from checklist.editor import Editor
from checklist.expect import Expect
from pathlib import Path

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from .sentiwordnet.Sentiwordnet import Sentiwordnet
from .Synonyms import Synonyms


# get pos/neg/neu words from SentiWordNet
SENT_WORDS = Sentiwordnet.get_sent_words()
SENT_DICT = {
    "positive_adj": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="adj" and SENT_WORDS[w]["label"]=="positive"],
    "negative_adj": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="adj" and SENT_WORDS[w]["label"]=="negative"],
    "neutral_adj": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="adj" and SENT_WORDS[w]["label"]=="pure neutral"],
    "positive_verb": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="verb" and SENT_WORDS[w]["label"]=="positive"],
    "negative_verb": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="verb" and SENT_WORDS[w]["label"]=="negative"],
    "neutral_verb": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="verb" and SENT_WORDS[w]["label"]=="pure neutral"],
    "positive_noun": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="noun" and SENT_WORDS[w]["label"]=="positive"],
    "negative_noun": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="noun" and SENT_WORDS[w]["label"]=="negative"],
    "neutral_noun": [w for w in SENT_WORDS.keys() if SENT_WORDS[w]["POS"]=="noun" and SENT_WORDS[w]["label"]=="pure neutral"]
}

CONTRACTION_MAP = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot",
    "can't've": "cannot have", "could've": "could have", "couldn't":
    "could not", "didn't": "did not", "doesn't": "does not", "don't":
    "do not", "hadn't": "had not", "hasn't": "has not", "haven't":
    "have not", "he'd": "he would", "he'd've": "he would have",
    "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y":
    "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'll": "I will", "I'm": "I am",
    "I've": "I have", "i'd": "i would", "i'll": "i will",
    "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it'd": "it would", "it'll": "it will", "it's": "it is", "ma'am":
    "madam", "might've": "might have", "mightn't": "might not",
    "must've": "must have", "mustn't": "must not", "needn't":
    "need not", "oughtn't": "ought not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "that'd":
    "that would", "that's": "that is", "there'd": "there would",
    "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what're": "what are", "what's": "what is",
    "when's": "when is", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who's": "who is",
    "who've": "who have", "why's": "why is", "won't": "will not",
    "would've": "would have", "wouldn't": "would not",
    "you'd": "you would", "you'd've": "you would have",
    "you'll": "you will", "you're": "you are", "you've": "you have"
}

NEGATION_PHRASE = [
    "I wouldn't say that ", "I don't think that", "I don't agree with that",
    "I would never say that", "I can't say that", "I disagree that",
    "I never thought that", "I cannot accept that", 
]

AGREEMENT_PHRASE = [
    "I would say that ", "I do think that", "I agree with that",
    "I accept that", "I believe that", "I accept that"
]

CUR_NEG_TEMPORAL_PHRASE = {
    "past": [
        "Previously, I used to {like_VB} it saying that",
        "Last time, I {liked_VBD} it saying that",
        "I {liked_VBD} it much as to say that"
    ],
    "change": ['but', 'even though', 'although'],
    "current": ["now I don't {like_VB} it.", "now I {hate_VB} it."]
}

CUR_POS_TEMPORAL_PHRASE = {
    "past": [
        "I used to {hate_VB} it saying that ",
        "Last time, I didn't {like_VB} it saying that",
        "I {hated_VBD} it much as to say that"
    ],
    "change": ['but', 'even though', 'although'],
    "current": ["now I {like_VB} it."]
}


random.seed(27)

class TransformOperator:

    def __init__(self,
                 editor,
                 req_capability,
                 req_description,
                 transform_reqs
                 ):
        
        self.editor = editor # checklist.editor.Editor()
        self.capability = req_capability
        self.description = req_description
        self.search_dataset = search_dataset
        self.transform_reqs = transform_reqs
        self.inv_replace_target_words = None
        self.inv_replace_forbidden_words = None
        self.transformation_funcs = None
        self.dir_adding_phrases = None

        # Find INV transformation operations
        if transform_reqs["INV"] is not None:
            self.set_inv_env(transform_reqs["INV"])
        # end if
        
        if transform_reqs["DIR"] is not None:
            self.set_dir_env(transform_reqs["DIR"])
        # end if
        
    def set_inv_env(self, inv_transform_reqs):
        if len(inv_transform_reqs.split())==2:
            func = inv_transform_reqs.split()[0]
            _property = None
            woi = inv_transform_reqs.split()[1]
        else:
            func = inv_transform_reqs.split()[0]
            _property = inv_transform_reqs.split()[1]
            woi = inv_transform_reqs.split()[2]
        # end if
        if func=="replace":
            self.inv_replace_target_words = list()
            self.inv_replace_forbidden_words = list()
            if _property=="neutral" and woi=="word":
                self.inv_replace_target_words = set(SENT_DICT[f"{_property}_adj"] + \
                                                    SENT_DICT[f"{_property}_verb"] + \
                                                    SENT_DICT[f"{_property}_noun"])
                self.inv_replace_forbidden_words = set(['No', 'no', 'Not', 'not', 'Nothing', 'nothing', 'without', 'but'] + \
                                                       SENT_DICT["positive_adj"] + \
                                                       SENT_DICT[f"negative_adj"] + \
                                                       SENT_DICT[f"positive_verb"] + \
                                                       SENT_DICT[f"negative_verb"] + \
                                                       SENT_DICT[f"positive_noun"] + \
                                                       SENT_DICT[f"negative_noun"])
            # end if
        # end if
        self.transformation_funcs = f"INV:{func}:{_property}:{woi}"
        return

    def set_dir_env(self, dir_transform_reqs):
        if len(dir_transform_reqs.split())==2:
            func = dir_transform_reqs.split()[0]
            _property = None
            woi = dir_transform_reqs.split()[2]
        else:
            func = dir_transform_reqs.split()[0]
            _property = dir_transform_reqs.split()[1]
            woi = dir_transform_reqs.split()[2]
        # end if
        if func=="add":
            if _property=="positive" and woi=="phrase":
                nlp = spacy.load('en_core_web_sm')
                labels = [
                    {'like_VB': ['like']+Synonyms.get_synonyms(nlp, 'like', 'VB', num_synonyms=10)},
                    {'love_VB': ['love']+Synonyms.get_synonyms(nlp, 'love', 'VB', num_synonyms=10)},
                    {'great_ADJ': ['great']+Synonyms.get_synonyms(nlp, 'great', 'ADJ', num_synonyms=10)}
                ]
                self.dir_adding_phrases = self.editor.template('I {like_VB} you.', labels=labels[0]).data
                self.dir_adding_phrases += self.editor.template('I {love_VB} it.', labels=labels[1]).data
                self.dir_adding_phrases += self.editor.template('You are {great_ADJ}.', labels=labels[2]).data
                self.dir_expect_func = Expect.pairwise(self.diff_up)
            elif _property=="negative" and woi=="phrase":
                labels = [
                    {'hate_VB': ['hate']+Synonyms.get_synonyms(nlp, 'hate', 'VB', num_synonyms=10)},
                    {'dislike_VB': ['dislike']+Synonyms.get_synonyms(nlp, 'dislike', 'VB', num_synonyms=10)},
                    {'bad_ADJ': ['bad']+Synonyms.get_synonyms(nlp, 'bad', 'ADJ', num_synonyms=10)}
                ]
                self.dir_adding_phrases = editor.template('I {hate_VB} you.', labels=labels[0]).data
                self.dir_adding_phrases += editor.template('I {dislike_VB} it.', labels=labels[1]).data
                self.dir_adding_phrases += editor.template('You are {bad_ADJ}.', labels=labels[2]).data
                self.dir_expect_func = Expect.pairwise(self.diff_down)
            # end if
            self.transformation_funcs = f"DIR:{func}:{_property}:{woi}"
        # end if
        return
                
    def replace(self, d):
        examples = list()
        subs = list()
        target_words = set(self.inv_replace_target_words)
        forbidden = self.inv_replace_forbidden_words
        
        words_in = [x for x in d.split() if x in target_words]
        if not words_in:
            return None
        # end if
        for w in words_in:
            suggestions = [
                x for x in self.editor.suggest_replace(d, w, beam_size=5, words_and_sentences=True)
                if x[0] not in forbidden
            ]
            examples.extend([x[1] for x in suggestions])
            subs.extend(['%s -> %s' % (w, x[0]) for x in suggestions])
        # end for
        if examples:
            idxs = np.random.choice(len(examples), min(len(examples), 10), replace=False)
            return [examples[i] for i in idxs]#, [subs[i] for i in idxs])
        # end if

    # functions for adding positive/negative phrase
    def add_phrase(self):
        def pert(d):
            while d[-1].pos_ == 'PUNCT':
                d = d[:-1]
            # end while
            d = d.text
            ret = [d + '. ' + x for x in self.dir_adding_phrases]
            idx = np.random.choice(len(ret), min(len(ret),10), replace=False)
            ret = [ret[i] for i in idx]
            return ret
        return pert
    
    def positive_change(self, orig_conf, conf):
        softmax = type(orig_conf) in [np.array, np.ndarray]
        if not softmax or orig_conf.shape[0] != 3:
            raise(Exception('Need prediction function to be softmax with 3 labels (negative, neutral, positive)'))
        # end if
        return orig_conf[0] - conf[0] + conf[2] - orig_conf[2]

    def diff_up(self, orig_pred, pred, orig_conf, conf, labels=None, meta=None):
        tolerance = 0.1
        change = self.positive_change(orig_conf, conf)
        if change + tolerance >= 0:
            return True
        else:
            return change + tolerance
        # end if
        
    def diff_down(self, orig_pred, pred, orig_conf, conf, labels=None, meta=None):
        tolerance = 0.1
        change = self.positive_change(orig_conf, conf)
        if change - tolerance <= 0:
            return True
        else:
            return -(change - tolerance)
        # end if

    def random_string(self, n):
        return ''.join(np.random.choice([x for x in string.ascii_letters + string.digits], n))
        
    def random_url(self, n=6):
        return 'https://t.co/%s' % self.random_string(n)
    
    def random_handle(self, n=6):
        return '@%s' % self.random_string(n)

    def add_irrelevant(self, sentence):
        urls_and_handles = [self.random_url(n=6) for _ in range(5)] + [self.random_handle() for _ in range(5)]
        irrelevant_before = ['@airline '] + urls_and_handles
        irrelevant_after = urls_and_handles 
        rets = ['%s %s' % (x, sentence) for x in irrelevant_before ]
        rets += ['%s %s' % (sentence, x) for x in irrelevant_after]
        return rets
    
    # def change_temporalness_template(self, templates):
    #     # templates= List[Dict]
    #     # convert every generate templates into temporal awareness formated templates
    #     # each template keys: sent, values, label
    #     new_templates = list()
    #     nlp = spacy.load('en_core_web_md')
    #     nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
    #     for temp in templates:
    #         label = temp["label"]
    #         if label = Macros.sa_label_map['positive']:
    #             prefix_phrases= list()
    #             for phrase in CUR_NEG_TEMPORAL_PHRASE['past']:
    #                 search = re.search("(.+)\{([^\_]+)\_([^\_]+)\}(.+)", phrase)
    #                 prefix, word, pos, postfix = search.group(1), search.group(2), search.group(3), search.group(4)
    #                 syns = Synonyms.get_synonyms(nlp, word, pos)
    #                 if len(syns)>1:
    #                     for s in list(set(syns)):
    #                         prefix_phrases.append(prefix+s+postfix)
    #                     # end for
    #                 else:
    #                     prefix_phrases.append(prefix+word+postfix)
    #                 # end if
    #             # end for
    #         elif label = Macros.sa_label_map['negative']:
