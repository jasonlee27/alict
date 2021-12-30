# This script searches inputs in datasets that meet requirements

from typing import *

import re, os
import sys
import json
import random
import checklist
import numpy as np

from nltk.tokenize import word_tokenize as tokenize
from pathlib import Path

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..requirement.Requirements import Requirements
from .sentiwordnet.Sentiwordnet import Sentiwordnet


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
random.seed(27)

class SearchOperator: 
    
    def __init__(self, requirements):
        self.capability = requirements['capability']
        self.description = requirements['description']
        self.search_reqs_list = requirements['search']
        self.search_method = {
            "length": self.search_by_len,
            "include": self.search_by_include,
            "exclude": self.search_by_exclude,
            "label": self.search_by_label
        }

    def search(self, sents):
        selected = list()
        for search_reqs in self.search_reqs_list:
            _sents = sents.copy()
            for op, param in search_reqs.items():
                _sents = self.search_method[op](_sents, search_reqs)
            # end for
            selected.extend(_sents)
        # end for
        return list(set(selected))

    def search_by_len(self, sents, search_reqs):
        param = search_reqs["length"]
        match = re.search(r"([<>]=?|==)(\d+)", param)
        op, _len = match.groups()
        _sents = [(s_i,tokenize(s),l) for s_i, s, l in sents]
        return [(s_i," ".join(s),l) for s_i, s, l in _sents if len(s) < int(_len)]

    def search_by_label(self, sents, search_reqs):
        label = search_reqs["label"]
        if label=="neutral" or label=="positive" or label=="negative":
            _sents = [(s_i,s,l) for s_i, s, l in sents if l==label]
        # elif label.startswith("same as"):
        #     print(label)
        #     match = re.search(r"same as\s?(include|exclude)?\s?(POS|word)?", label)
        #     in_or_ex_clude, search_key = match.groups()
        #     if in_or_ex_clude is None:
        #         in_or_ex_clude = "include"
        #     # end if
        #     poss = search_reqs[in_or_ex_clude][search_key]
        #     _sents = list()
        #     for pos in poss:
        #         pos_sentiment = pos.split()[0]
        #         _sents.extend([(s_i,s,l) for s_i, s, l in sents if l==pos_sentiment])
        #     # end for
        else:
            return sents
        # end if
        return _sents

    def _search_by_word_include(self, sents, word_cond):
        pass

    def _search_by_pos_include(self, sents, cond_key, cond_number):
        # sents: (s_i, tokenizes sentence, label)
        target_words = SENT_DICT[cond_key]
        selected = list()
        for s_i, s, l in sents:
            found_w = list()
            for w in s:
                if w.lower() in target_words:
                    found_w.append(w)
                # end if
            # end for
            if cond_number>0 and len(found_w)==cond_number:
                selected.append((s_i, s, l))
            elif cond_number<0 and len(found_w)>0:
                selected.append((s_i, s, l))
            # end if
        # end for
        return selected
        
    def search_by_include(self, sents, search_reqs):
        _sents = sents.copy()
        _sents = [(s_i, tokenize(s), l) for s_i, s, l in sents]
        params = search_reqs["include"]
        if type(params)==dict:
            params = [params]
        # end if
        selected_indices = list()
        for param in params:
            word_include = param["word"]
            tpos_include = param["POS"]
            if word_include is not None:
                for w in word_include:
                    _sents = [(s_i, s, l) for s_i, s, l in _sents if w in s]
                # end for
            # end if
        
            if tpos_include is not None:
                temp_sents = list()
                for cond in tpos_include:
                    match = re.search(r"(\d+)?\s?(positive|negative|neutral)?\s?(adj|noun|verb)?(s)?", cond)
                    num, sentiment, pos, is_plural = match.groups()
                    if pos is None: raise("Tag of POS is not valid!")
                    num = -1
                    if num is None and not is_plural:
                        num = 1
                    # end if
                
                    if sentiment is None:
                        for _sentiment in ["neutral","positive","negative"]:
                            temp_sents.extend(self._search_by_pos_include(_sents, f"{_sentiment}_{pos}", num))
                        # end for
                    else:
                        temp_sents = self._search_by_pos_include(_sents, f"{sentiment}_{pos}", num)
                    # end if
                    _sents = temp_sents
                # end for
            # end if
            for s_i, _, _ in _sents:
                if s_i not in selected_indices:
                    selected_indices.append(s_i)
                # end if
            # end for
        # end for
        return [(s_i," ".join(s),l) for s_i, s, l in _sents if s_i in selected_indices]

    def _search_by_pos_exclude(self, sents, cond_key):
        # sents: (s_i, tokenizes sentence, label)
        target_words = SENT_DICT[cond_key]
        selected = list()
        for s_i, s, l in sents:
            found_w = list()
            for w in s:
                if w.lower() in target_words:
                    found_w.append(w)
                # end if
            # end for
            if len(found_w)==0:
                selected.append((s_i, s, l))
            # end if
        # end for
        return selected

    def search_by_exclude(self, sents, search_reqs):
        params = search_reqs["exclude"]
        _sents = sents.copy()
        _sents = [(s_i, tokenize(s), l) for s_i, s, l in sents]
        selected_indices = list()
        if type(params)==dict:
            params = [params]
        # end if
        for param in params:
            word_exclude = param["word"]
            tpos_exclude = param["POS"]
            if word_exclude is not None:
                for w in word_exclude:
                    _sents = [(s_i, s, l) for s_i, s, l in _sents if w not in s]
                # end for
            # end if
        
            if tpos_exclude is not None:
                temp_sents = list()
                for cond in tpos_exclude:
                    match = re.search(r"(positive|negative|neutral)?\s?(adj|noun|verb)?(s)?", cond)
                    sentiment, pos, is_plural = match.groups()
                    if pos is None: raise("Tag of POS is not valid!")
                
                    if sentiment is None:
                        for _sentiment in ["neutral","positive","negative"]:
                            temp_sents.extend(self._search_by_pos_exclude(_sents, f"{_sentiment}_{pos}"))
                        # end for
                    else:
                        temp_sents = self._search_by_pos_exclude(_sents, f"{sentiment}_{pos}")
                    # end if
                    _sents = temp_sents
                # end for
            # end if

            for s_i, _, _ in _sents:
                if s_i not in selected_indices:
                    selected_indices.append(s_i)
                # end if
            # end for
        # end for
        return [(s_i," ".join(s),l) for s_i, s, l in _sents if s_i in selected_indices]


class TransformOperator:

    def __init__(self, editor, req_capability, req_description, transform_reqs):
        self.editor = editor # checklist.editor.Editor()
        self.capability = req_capability
        self.description = req_description
        self.transform_reqs = transform_reqs
        self.inv_replace_target_words = None
        self.inv_replace_forbidden_words = None
        self.transformation_funcs = None

        # Find INV transformation operations
        func = transform_reqs["INV"].split()[0]
        sentiment = transform_reqs["INV"].split()[1]
        woi = transform_reqs["INV"].split()[2]
        if func=="replace":
            self.inv_replace_target_words = list()
            self.inv_replace_forbidden_words = list()
            if woi=="word":
                self.inv_replace_target_words = set(SENT_DICT[f"{sentiment}_adj"] + \
                                                    SENT_DICT[f"{sentiment}_verb"] + \
                                                    SENT_DICT[f"{sentiment}_noun"])
                self.inv_replace_forbidden_words = set(['No', 'no', 'Not', 'not', 'Nothing', 'nothing', 'without', 'but'] + \
                                                       SENT_DICT["positive_adj"] + \
                                                       SENT_DICT[f"negative_adj"] + \
                                                       SENT_DICT[f"positive_verb"] + \
                                                       SENT_DICT[f"negative_verb"])
            else:
                self.inv_replace_target_words = set(SENT_DICT[f"{sentiment}_{woi}"])
                forbidden_sentiment = "negative"
                if sentiment=="negative":
                    forbidden_sentiment = "positive"
                # end if
                self.inv_replace_forbidden_words = set(['No', 'no', 'Not', 'not', 'Nothing', 'nothing', 'without', 'but'] + \
                                                       SENT_DICT[f"{forbidden_sentiment}_adj"] + \
                                                       SENT_DICT[f"{forbidden_sentiment}_verb"] + \
                                                       SENT_DICT[f"{forbidden_sentiment}_noun"])
            # end if
            self.transformation_funcs = f"INV_{func}_{sentiment}_{woi}"
            self.inv_replace_target_words = set(self.inv_replace_target_words)
            self.inv_replace_forbidden_words = set(self.inv_replace_forbidden_words)
        # end if
                
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
        

class Sst:

    @classmethod
    def get_sents(cls, sent_file, label_file):
        # sents: List of [sent_index, sent]
        sents = [tuple(l.split("\t")) for l in Utils.read_txt(sent_file)[1:]]
        label_scores = [tuple(l.split("|")) for l in Utils.read_txt(label_file)[1:]]
        labels = dict()
        for s_i, s in label_scores:
            s = float(s)
            labels[s_i] = "neutral"
            if s<=0.4:
                labels[s_i] = "negative"
            elif s>0.6:
                labels[s_i] = "positive"
            # end if
        #end for
        return [(s_i,s,labels[s_i]) for s_i, s in sents]
    
    @classmethod
    def search(cls, req):
        # sent: (index, sentence)
        # label: (index, label score)
        sents = cls.get_sents(Macros.sst_datasent_file, Macros.sst_label_file)
        req_obj = SearchOperator(req)
        selected = sorted([(s[0],s[1].strip()[:-1],s[2]) if s[1].strip()[-1]=="." else (s[0],s[1].strip(),s[2]) for s in req_obj.search(sents)], key=lambda x: x[0])
        random.shuffle(selected)
        return selected


class ChecklistTestsuite:

    @classmethod
    def get_sents(cls, testsuite_file):
        tsuite, tsuite_dict = read_testsuite(testsuite_file)
        sents, raw_labels = list(), list()
        for test_name in test_names:
            # sents: List of sent
            # label: 0(neg), 1(neu) and 2(pos)
            sents.extend(tsuite.tests[test_name].data) 
            raw_labels.extend(tsuite.tests[test_name].labels)
        # end for        
        labels = dict()
        for s_i, s in enumerate(raw_labels):
            if s=='0':
                labels[s_i] = "negative"
            elif s=='1':
                labels[s_i] = "neutral"
            else:
                labels[s_i] = "positive"
            # end if
        #end for
        return [(s_i, s, labels[s_i]) for s_i, s in enumerate(sents)]
    
    @classmethod
    def search(cls, req):
        # sent: (index, sentence)
        # label: (index, label score)
        sents = cls.get_sents(Macros.checklist_sa_dataset_file)
        req_obj = SearchOperator(req)
        selected = sorted([(s[0],s[1].strip()[:-1],s[2]) if s[1].strip()[-1]=="." else (s[0],s[1].strip(),s[2]) for s in req_obj.search(sents)], key=lambda x: x[0])
        random.shuffle(selected)
        return selected

    
class DynasentRoundOne:

    @classmethod
    def get_sents(cls, src_file):
        sents = list()
        sent_i = 0
        with open(yelp_src_filename) as f:
            for line in f:
                d = json.loads(line)
                sents.append((sent_i, d['sentence'], d['gold_label']))
                sent_i += 1
            # end for
        # end with
        return [(s_i,s,labels[s_i]) for s_i, s in sents]

    @classmethod
    def search(cls, req):
        # sent: (index, sentence)
        # label: (index, label score) 
        sents = cls.get_labels(Macros.dyna_r1_test_src_file)
        req_obj = SearchOperator(req)
        selected = sorted([(s[0],s[1].strip()[:-1],s[2]) if s[1].strip()[-1]=="." else (s[0],s[1].strip(),s[2]) for s in req_obj.search(sents)], key=lambda x: x[0])
        random.shuffle(selected)
        return selected


class Search:

    SEARCH_FUNC = {
        Macros.sa_task : {
            Macros.datasets[Macros.sa_task][0]: Sst.search,
            Macros.datasets[Macros.sa_task][1]: ChecklistTestsuite.search
        },
        Macros.mc_task : {},
        Macros.qqp_task : {}
    }

    @classmethod
    def get_dataset(cls, task_name, dataset_name):
        pass

    @classmethod
    def search_sentiment_analysis(cls, requirements, dataset):
        func = cls.SEARCH_FUNC[Macros.sa_task][dataset]
        for req in requirements:
            selected = func(req)
            yield {
                "requirement": req,
                "selected_inputs": selected
            }
        # end for
        return


