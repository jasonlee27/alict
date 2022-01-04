# This script searches inputs in datasets that meet requirements

from typing import *

import re, os
import sys
import json
import spacy
import random
import checklist
import numpy as np

from nltk.tokenize import word_tokenize as tokenize
from pathlib import Path

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from .Transform import CONTRACTION_MAP
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
            "label": self.search_by_label,
            "score": self.search_by_score,
        }

    def search(self, sents):
        selected = list()
        for search_reqs in self.search_reqs_list:
            _sents = sents.copy()
            for op, param in search_reqs.items():
                if len(_sents)>0:
                    _sents = self.search_method[op](_sents, search_reqs)
                # end if
            # end for
            selected.extend(_sents)
        # end for
        return list(set(selected))

    def search_by_len(self, sents, search_reqs):
        param = search_reqs["length"]
        match = re.search(r"([<>]=?|==)(\d+)", param)
        op, _len = match.groups()
        results = list()
        if len(sents[0])==4:
            _sents = [(s_i,tokenize(s),l,sc) for s_i, s, l, sc in sents]
            for s_i, s, l, sc in _sents:
                if s[-1]=="." and (eval(f"{len(s)-1} {op} {_len}")):
                    results.append((s_i," ".join(s),l,sc))
                elif s[-1]!="." and (eval(f"{len(s)} {op} {_len}")):
                    results.append((s_i," ".join(s),l,sc))
                # end if
            # end for
        else:
            _sents = [(s_i,tokenize(s),l) for s_i, s, l in sents]
            for s_i, s, l in _sents:
                if s[-1]=="." and (eval(f"{len(s)-1}{op}{_len}")):
                    results.append((s_i," ".join(s),l))
                elif s[-1]!="." and (eval(f"{len(s)}{op}{_len}")):
                    results.append((s_i," ".join(s),l))
                # end if
            # end for
        # end if
        return results

    def search_by_person_name_include(self, sents):
        nlp = spacy.load('en_core_web_sm')
        results = list()
        for sent in sents:
            doc = nlp(sent)
            is_person_name_contained = any([True for x in doc.ents if any([a.ent_type_ == 'PERSON' for a in x])])
            if is_person_name_contained:
                resluts.append(sent)
            # end if
        # end for
        return results

    def search_by_location_name_include(self, sents):
        # location: city names && country names
        nlp = spacy.load('en_core_web_sm')
        results = list()
        for sent in sents:
            doc = nlp(sent)
            is_loc_name_contained = any([True for x in doc.ents if any([a.ent_type_ == 'GPE' for a in x])])
            if is_loc_name_contained:
                resluts.append(sent)
            # end if
        # end for
        return results
    
    def search_by_number_include(self, sents):
        nlp = spacy.load('en_core_web_sm')
        results = list()
        for sent in sents:
            doc = nlp(sent)
            is_number_contained = any([True for x in doc if x.text.isdigit()])
            if is_number_contained:
                resluts.append(sent)
            # end if
        # end for
        return results

    def search_by_punctuation_include(self, sents):
        word_list = CONTRACTION_MAP.keys()+CONTRACTION_MAP.values()
        results = list()
        for s in sents:
            # if len(sents[0])==4: sents = List[(s_i,tokenize(s),label,label_scores)]
            # if len(sents[0])==3: sents = List[(s_i,tokenize(s),label)]
            is_contained = [True for w in word_list if w in s[1]]
            if any(is_contained):
                results.append(s)
            # end if
        # end for
        return results

    def search_by_score(self, sents, search_reqs):
        param = search_reqs["score"]
        # match = re.search(r"([<>]=?|==)(\d+\.?\d+)", param)
        # op, score = match.groups()
        results = list()
        _sents = [(s_i,tokenize(s),l,sc) for s_i, s, l, sc in sents]
        for s_i, s, l, sc in _sents:
            if eval(f"{sc}{param}"):
                results.append((s_i," ".join(s),l,sc))
            # end if
        # end for
        return results

    def search_by_label(self, sents, search_reqs):
        label = search_reqs["label"]
        if label=="neutral" or label=="positive" or label=="negative":
            if len(sents[0])==4:
                _sents = [(s_i,s,l,sc) for s_i, s, l, sc in sents if l==label]
            else:
                _sents = [(s_i,s,l) for s_i, s, l in sents if l==label]
            # end if
            return _sents
        else:
            return sents
        # end if

    def _search_by_word_include(self, sents, word_cond):
        search = re.search("\<([^\<\>]+)\>", word_cond)
        selected = list()
        if search:
            # if word condition searches a specific group of words
            # such as "name of person, location"
            word_group = search.group(1)
            word_list = list()
            if word_group=="punctuation":
                selected = self.search_by_punctuation_include(sents)
            elif word_group=="person_name":
                selected = self.search_by_person_name_include(sents)
            elif word_group=="location_name":
                selected = self.search_by_location_name_include(sents)
            elif word_group=="number":
                selected = self.search_by_number_include(sents)
            # end if
        else:
            selected = [s for s in sents if word_cond in s[1]]
        # end if
        return selected
    
    def _search_by_pos_include(self, sents, cond_key, cond_number):
        # sents: (s_i, tokenizes sentence, label)
        target_words = SENT_DICT[cond_key]
        selected = list()
        for sent in sents:
            # s_i, s, l, sc = sent
            found_w = list()
            for w in sent[1]:
                if w.lower() in target_words:
                    found_w.append(w)
                # end if
            # end for
            if cond_number>0 and len(found_w)==cond_number:
                selected.append(sent)
            elif cond_number<0 and len(found_w)>0:
                selected.append(sent)
            # end if
        # end for
        return selected
        
    def search_by_include(self, sents, search_reqs):
        _sents = sents.copy()
        if len(sents[0])==4:
            _sents = [(s_i,tokenize(s),l,sc) for s_i, s, l, sc in sents]
        else:
            _sents = [(s_i,tokenize(s),l) for s_i, s, l in sents]
        # end if
        params = search_reqs["include"]
        if type(params)==dict:
            params = [params]
        # end if
        selected_indices = list()
        for param in params:
            word_include = param["word"]
            tpos_include = param["POS"]
            
            if word_include is not None:
                temp_sents = list()
                for w in word_include: # AND relationship
                    _sents = self._search_by_word_include(_sents, w)
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
            for sent in _sents:
                if sent[0] not in selected_indices:
                    selected_indices.append(sent[0])
                # end if
            # end for
        # end for
        result = list()
        if len(sents[0])==4:
            result = [(s_i," ".join(s),l,sc) for s_i, s, l, sc in _sents if s_i in selected_indices]
        else:
            result = [(s_i," ".join(s),l) for s_i, s, l in _sents if s_i in selected_indices]
        # end if
        return result

    def _search_by_pos_exclude(self, sents, cond_key):
        # sents: (s_i, tokenizes sentence, label)
        target_words = SENT_DICT[cond_key]
        selected = list()
        for sent in sents:
            found_w = list()
            for w in sent[1]:
                if w.lower() in target_words:
                    found_w.append(w)
                # end if
            # end for
            if len(found_w)==0:
                selected.append(sent)
            # end if
        # end for
        return selected

    def search_by_exclude(self, sents, search_reqs):
        params = search_reqs["exclude"]
        _sents = sents.copy()
        if len(sents[0])==4:
            _sents = [(s_i,tokenize(s),l,sc) for s_i, s, l, sc in sents]
        else:
            _sents = [(s_i,tokenize(s),l) for s_i, s, l in sents]
        # end if
        selected_indices = list()
        if type(params)==dict:
            params = [params]
        # end if
        for param in params:
            word_exclude = param["word"]
            tpos_exclude = param["POS"]
            if word_exclude is not None:
                for w in word_exclude:
                    _sents = [(s_i, s, l, sc) for s_i, s, l, sc in _sents if w not in s]
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

            for sent in _sents:
                if sent[0] not in selected_indices:
                    selected_indices.append(sent[0])
                # end if
            # end for
        # end for
        result = list()
        if len(sents[0])==4:
            result = [(s_i," ".join(s),l, sc) for s_i, s, l, sc in _sents if s_i in selected_indices]
        else:
            result = [(s_i," ".join(s),l) for s_i, s, l in _sents if s_i in selected_indices]
        # end if
        return result        

    
class Sst:

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
    def get_sents(cls, sent_file, label_file, phrase_dict_file):
        # sents: List of [sent_index, sent]
        sents = [(l.split("\t")[0].strip(),l.split("\t")[1].strip()) for l in Utils.read_txt(sent_file)[1:]]
        label_scores = {
            l.split("|")[0].strip():l.split("|")[1].strip() # id: score
            for l in Utils.read_txt(label_file)[1:]
        }
        phrases = {
            l.split("|")[0].strip(): l.split("|")[1].strip() # phrase: id
            for l in Utils.read_txt(phrase_dict_file)
        }
        result = list()
        for s_i, s in sents:
            s = cls.replace_non_english_letter(s)
            if s in phrases.keys():
                phrase_id = phrases[s]
                label_score = float(label_scores[phrase_id])
                label = "neutral"
                if label_score<=0.4:
                    label = "negative"
                elif label_score>0.6:
                    label = "positive"
                # end if
                result.append((s_i,s,label,label_score))
            # end if
        # end for
        return result
    
    @classmethod
    def search(cls, req):
        sents = cls.get_sents(Macros.sst_datasent_file, Macros.sst_label_file, Macros.sst_dict_file)
        req_obj = SearchOperator(req)
        if len(req_obj.search_reqs_list)>0:
            if len(sents[0])==4:
                selected = sorted([(s[0],s[1],s[2],s[3]) for s in req_obj.search(sents)], key=lambda x: x[0])
            else:
                selected = sorted([(s[0],s[1],s[2]) for s in req_obj.search(sents)], key=lambda x: x[0])
            # end if
        else:
            selected = sents
        # end if
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
        selected = sorted([(s[0],s[1],s[2]) for s in req_obj.search(sents)], key=lambda x: x[0])
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
        selected = sorted([(s[0],s[1],s[2]) for s in req_obj.search(sents)], key=lambda x: x[0])
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


