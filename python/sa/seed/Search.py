# This script searches inputs in datasets that meet requirements

from typing import *
from pathlib import Path

import re, os
import sys
import json
import spacy
import random
import checklist
import numpy as np

from itertools import product
from checklist.test_types import MFT, INV, DIR

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..requirement.Requirements import Requirements
from .Transform import TransformOperator
from .sentiwordnet.Sentiwordnet import Sentiwordnet


# get pos/neg/neu words from SentiWordNet
SENT_WORDS = Sentiwordnet.get_sent_words()
SENT_DICT = {
    'positive_adj': [w['word'] for w in SENT_WORDS.values() if w['POS']=='adj' and w['label']=='positive'],
    'negative_adj': [w['word'] for w in SENT_WORDS.values() if w['POS']=='adj' and w['label']=='negative'],
    'neutral_adj': [w['word'] for w in SENT_WORDS.values() if w['POS']=='adj' and w['label']=='pure neutral'],
    'positive_verb': [w['word'] for w in SENT_WORDS.values() if w['POS']=='verb' and w['label']=='positive'],
    'negative_verb': [w['word'] for w in SENT_WORDS.values() if w['POS']=='verb' and w['label']=='negative'],
    'neutral_verb': [w['word'] for w in SENT_WORDS.values() if w['POS']=='verb' and w['label']=='pure neutral'],
    'positive_noun': [w['word'] for w in SENT_WORDS.values() if w['POS']=='noun' and w['label']=='positive'],
    'negative_noun': [w['word'] for w in SENT_WORDS.values() if w['POS']=='noun' and w['label']=='negative'],
    'neutral_noun': [w['word'] for w in SENT_WORDS.values() if w['POS']=='noun' and w['label']=='pure neutral']
}

WORD2POS_MAP = {
    'demonstratives': ['This', 'That', 'These', 'Those'],
    'AUXBE': ['is', 'are']
}

# # get name and location data
# basic = Utils.read_json(Macros.dataset_dir / 'checklist' / 'lexicons' / 'basic.json')
# names = Utils.read_json(Macros.dataset_dir / 'checklist' / 'names.json')
# name_set = { x:set(names[x]) for x in names }
# NAME_LOC_DICT = {
#     'name': names,
#     'name_set': name_set,
#     'city': basic['city'],
#     'country': basic['country'],
# }


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
        if self.search_reqs_list is None:
            return sents
        # end if
        for search_reqs in self.search_reqs_list:
            _sents = sents.copy()
            for op, param in search_reqs.items():
                if len(_sents)>0:
                    _sents = self.search_method[op](_sents, search_reqs)
                # end if
            # end for
            for s in _sents:
                if s not in selected:
                    selected.append(s)
                # end if
            # end for
        # end for
        return selected

    def search_by_len(self, sents, search_reqs):
        param = search_reqs["length"]
        match = re.search(r"([<>]=?|==)(\d+)", param)
        op, _len = match.groups()
        results = list()
        if len(sents[0])==4:
            _sents = [(s_i,Utils.tokenize(s),l,sc) for s_i, s, l, sc in sents]
            for s_i, s, l, sc in _sents:
                if s[-1]=="." and (eval(f"{len(s)-1} {op} {_len}")):
                    results.append((s_i,Utils.detokenize(s),l,sc))
                elif s[-1]!="." and (eval(f"{len(s)} {op} {_len}")):
                    results.append((s_i,Utils.detokenize(s),l,sc))
                # end if
            # end for
        else:
            _sents = [(s_i,Utils.tokenize(s),l) for s_i, s, l in sents]
            for s_i, s, l in _sents:
                if s[-1]=="." and (eval(f"{len(s)-1}{op}{_len}")):
                    results.append((s_i,Utils.detokenize(s),l))
                elif s[-1]!="." and (eval(f"{len(s)}{op}{_len}")):
                    results.append((s_i,Utils.detokenize(s),l))
                # end if
            # end for
        # end if
        return results

    # def search_by_punctuation_include(self, sents):
    #     nlp = spacy.load('en_core_web_sm')
    #     results = list()
    #     for sent in sents:
    #         s = Utils.detokenize(sent[1])
    #         doc = nlp(s)
    #         if len(doc) and doc[-1].pos_ == 'PUNCT':
    #             results.append(sent)
    #         # end if
    #     # end for
    #     return results
    
    # def search_by_person_name_include(self, sents):
    #     nlp = spacy.load('en_core_web_sm')
    #     results = list()
    #     for sent in sents:
    #         s = Utils.detokenize(sent[1])
    #         doc = nlp(s)
    #         is_person_name_contained = any([True for x in doc.ents if any([a.ent_type_ == 'PERSON' for a in x])])
    #         if is_person_name_contained:
    #             ents = [x.text for x in doc.ents if np.all([a.ent_type_ == 'PERSON' for a in x])]
    #             if any([x for x in ents if x in NAME_LOC_DICT['name_set']['women'] or x in NAME_LOC_DICT['name_set']['men']]):
    #                 results.append(sent)
    #             # end if
    #         # end if
    #     # end for
    #     return results

    # def search_by_location_name_include(self, sents):
    #     # location: city names && country names
    #     nlp = spacy.load('en_core_web_sm')
    #     results = list()
    #     for sent in sents:
    #         s = Utils.detokenize(sent[1])
    #         doc = nlp(s)
    #         is_loc_name_contained = any([True for x in doc.ents if any([a.ent_type_ == 'GPE' for a in x])])
    #         if is_loc_name_contained:
    #             ents = [x.text for x in doc.ents if np.all([a.ent_type_ == 'GPE' for a in x])]
    #             if any([x for x in ents if x in NAME_LOC_DICT['city'] or x in NAME_LOC_DICT['country']]):
    #                 results.append(sent)
    #             # end if
    #         # end if
    #     # end for
    #     return results
    
    # def search_by_number_include(self, sents):
    #     nlp = spacy.load('en_core_web_sm')
    #     results = list()
    #     for sent in sents:
    #         s = Utils.detokenize(sent[1])
    #         doc = nlp(s)
    #         nums = [x.text for x in doc if x.text.isdigit()]
    #         if any(nums) and any([x for x in nums if x != '2' and x != '4']):
    #             results.append(sent)
    #         # end if
    #     # end for
    #     return results

    # def search_by_contraction_include(self, sents):
    #     contraction_pattern = re.compile(r'\b({})\b'.format('|'.join(CONTRACTION_MAP.keys())), flags=re.IGNORECASE|re.DOTALL)
    #     reverse_contraction_pattern = re.compile(r'\b({})\b'.format('|'.join(CONTRACTION_MAP.values())), flags=re.IGNORECASE|re.DOTALL)        
    #     results = list()
    #     for sent in sents:
    #         s = Utils.detokenize(sent[1])
    #         if contraction_pattern.search(s) or reverse_contraction_pattern.search(s):
    #             results.append(sent)
    #         # end if
    #     # end for
    #     return results

    def search_by_score(self, sents, search_reqs):
        param = search_reqs["score"]
        # match = re.search(r"([<>]=?|==)(\d+\.?\d+)", param)
        # op, score = match.groups()
        results = list()
        if len(sents[0])==4:
            _sents = [(s_i,Utils.tokenize(s),l,sc) for s_i, s, l, sc in sents]
        else:
            _sents = [(s_i,Utils.tokenize(s),l) for s_i, s, l in sents]
        # end if
        for s_i, s, l, sc in _sents:
            if eval(f"{sc}{param}"):
                results.append((s_i,Utils.detokenize(s),l,sc))
            # end if
        # end for
        return results

    def search_by_label(self, sents, search_reqs):
        label = search_reqs["label"]
        if label=="neutral" or label=="positive" or label=="negative":
            if len(sents[0])==4:
                _sents = [(s_i,Utils.tokenize(s),l,sc) for s_i, s, l, sc in sents]
                _sents = [(s_i,Utils.detokenize(s),l,sc) for s_i, s, l, sc in _sents if l==label]
            else:
                _sents = [(s_i,Utils.tokenize(s),l) for s_i, s, l in sents]
                _sents = [(s_i,Utils.detokenize(s),l) for s_i, s, l in sents if l==label]
            # end if
            return _sents
        elif type(label)==list and ("neutral" in label or "positive" in label or "negative" in label):
            if len(sents[0])==4:
                _sents = [(s_i,Utils.tokenize(s),l,sc) for s_i, s, l, sc in sents]
                _sents = [(s_i,Utils.detokenize(s),l,sc) for s_i, s, l, sc in _sents if l==label]
            else:
                _sents = [(s_i,Utils.tokenize(s),l) for s_i, s, l in sents]
                _sents = [(s_i,Utils.detokenize(s),l) for s_i, s, l in _sents if l==label]
            # end if
            return _sents
        else:
            if len(sents[0])==4:
                _sents = [(s_i,Utils.tokenize(s),l,sc) for s_i, s, l, sc in sents]
                _sents = [(s_i,Utils.detokenize(s),l,sc) for s_i, s, l, sc in _sents]
            else:
                _sents = [(s_i,Utils.tokenize(s),l) for s_i, s, l in sents]
                _sents = [(s_i,Utils.detokenize(s),l) for s_i, s, l in sents]
            # end if
        # end if
        return _sents

    def _search_by_word_include(self, sents, word_cond):
        search = re.search("\<([^\<\>]+)\>", word_cond)
        selected = list()
        if search:
            # if word condition searches a specific group of words
            # such as "name of person, location"
            word_group = search.group(1)
            word_list = list()
            # if word_group=="contraction":
            #     selected = self.search_by_contraction_include(sents)
            # elif word_group=="punctuation":
            #     selected = self.search_by_punctuation_include(sents)
            # elif word_group=="person_name":
            #     selected = self.search_by_person_name_include(sents)
            # elif word_group=="location_name":
            #     selected = self.search_by_location_name_include(sents)
            # elif word_group=="number":
            #     selected = self.search_by_number_include(sents)
            # end if
        else:
            selected = [sent for sent in sents if word_cond in sent[1]]
        # end if
        return selected

    def get_pospat_to_wordproduct(self, pos_pattern):
        results = list()
        pos_dict = {
            p: WORD2POS_MAP[p]
            for p in pos_pattern.split('_')
        }
        # word_product = [dict(zip(d, v)) for v in product(*pos_dict.values())]
        word_product = [dict(zip(pos_dict, v)) for v in product(*pos_dict.values())]
        for wp in word_product:
            results.append(" ".join(list(wp.values())))
        # end for
        return results
            
    def _search_by_pos_include(self, sents, cond_key, cond_number):
        # sents: (s_i, tokenized sentence, label)
        search = re.search(r"\<([^\<\>]+)\>", cond_key)
        selected = list()
        if search:
            selected_ids = list()
            from itertools import product
            # search sents by tag of pos organization
            pos_pat = search.group(1)
            prefix_pat, postfix_pas = '',''
            if pos_pat.startswith('^'):
                pos_pat = pos_pat[1:]
                prefix_pat = '^'
            # end if
            # if pos_pat.endswith('$'):
            #     pos_pat = pos_pat[:-1]
            #     postfix_pat = '$'
            # # end if
            for pat in self.get_pospat_to_wordproduct(pos_pat):
                _pat = prefix_pat+pat
                selected_ids.extends([s[0] for s in sents if re.search(_pat, Utils.detokenize(s[1]))])
            # end for
            selected = [s for s in sents if s[0] in selected_ids]
            return selected
        # end if
        target_words = SENT_DICT[cond_key]
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

    def _get_pospattern_to_wordproduct(self, pos_pattern, value_dict):
        results = list()
        pos_dict = {
            p: value_dict[p]
            for p in pos_pattern.split('_')
        }
        word_product = [dict(zip(pos_dict, v)) for v in product(*pos_dict.values())]
        for wp in word_product:
            results.append(" ".join(list(wp.values())))
        # end for
        return results

    def _search_by_pos_pattern_include(self, sents, pos_pattern):
        prefix_pat, postfix_pas = '',''
        if pos_pattern.startswith('^'):
            pos_pattern = pos_pattern[1:]
            prefix_pat = '^'
        # end if
        res_idx = 0
        selected = list()
        for pat in self._get_pospattern_to_wordproduct(pos_pattern, WORD2POS_MAP):
            _pat = prefix_pat+pat
            for sent in sents:
                if re.search(_pat, Utils.detokenize(sent[1])) and \
                   not re.search(f"{_pat} not ", Utils.detokenize(sent[1])) and \
                   not re.search(f"{_pat} n't ", Utils.detokenize(sent[1])):
                    selected.append(sent)
                # end if
            # end if
        # end for
        return selected
        
    def search_by_include(self, sents, search_reqs):
        params = search_reqs["include"]
        if type(params)==dict:
            params = [params]
        # end if
        _sents = sents.copy()
        if len(sents[0])==4:
            _sents = [(s_i,Utils.tokenize(s),l,sc) for s_i, s, l, sc in _sents]
        else:
            _sents = [(s_i,Utils.tokenize(s),l) for s_i, s, l in _sents]
        # end if
        selected_indices = list()
        for param in params:
            word_include = param["word"]
            tpos_include = param["POS"]
            
            if word_include is not None:
                for w in word_include: # AND relationship
                    _sents = self._search_by_word_include(_sents, w)
                # end for
            # end if
        
            if tpos_include is not None:                
                temp_sents = list()
                for cond in tpos_include:
                    search = re.search(r"\<([^\<\>]+)\>", cond)
                    if search:
                        pos_pat = search.group(1)
                        temp_sents.extend(self._search_by_pos_pattern_include(_sents, pos_pat))
                    else:
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
                    # end if
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
            result = [(s[0],Utils.detokenize(s[1]),s[2],s[3]) for s in _sents if s[0] in selected_indices]
        else:
            result = [(s[0],Utils.detokenize(s[1]),s[2]) for s in _sents if s[0] in selected_indices]
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
            _sents = [(s_i,Utils.tokenize(s),l,sc) for s_i, s, l, sc in sents]
        else:
            _sents = [(s_i,Utils.tokenize(s),l) for s_i, s, l in sents]
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
                    _sents = [sent for sent in _sents if w not in sent[1]]
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
            result = [(s_i,Utils.detokenize(s),l, sc) for s_i, s, l, sc in _sents if s_i in selected_indices]
        else:
            result = [(s_i,Utils.detokenize(s),l) for s_i, s, l in _sents if s_i in selected_indices]
        # end if
        return result


class Sst:

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
            s = Utils.replace_non_english_letter(s)
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
        if req_obj.search_reqs_list is not None:
            if len(sents[0])==4:
                selected = sorted([(s[0],s[1],s[2],s[3]) for s in req_obj.search(sents)], key=lambda x: x[0])
            else:
                selected = sorted([(s[0],s[1],s[2]) for s in req_obj.search(sents)], key=lambda x: x[0])
            # end if
        else:
            if len(sents[0])==4:
                selected = [(s_i,Utils.tokenize(s),l,sc) for s_i, s, l, sc in sents]
                selected = [(s_i,Utils.detokenize(s),l,sc) for s_i, s, l, sc in selected]
            else:
                selected = [(s_i,Utils.tokenize(s),l) for s_i, s, l in sents]
                selected = [(s_i,Utils.detokenize(s),l) for s_i, s, l in selected]
            # end if
        # end if
        random.shuffle(selected)
        return selected


class ChecklistTestsuite:

    @classmethod
    def get_labels(cls, raw_labels, req_desc, num_data):
        labels = dict()
        if type(raw_labels)==int:
            raw_labels = [raw_labels]*num_data
        elif raw_labels is None:
            if req_desc == Macros.OUR_LC_LIST[3] or \
               req_desc == Macros.OUR_LC_LIST[6] or \
               req_desc == Macros.OUR_LC_LIST[10]:
                raw_labels = [['positive', 'neutral']]*num_data
            # end if
        else:
            labels = raw_labels
        # end if
        labels = dict()
        for s_i, s in enumerate(raw_labels):
            # remove multiple spaces in the text
            # checklist label map {'negative': 0, 'positive': 2, 'neutral': 1} 
            if type(s)==list:
                labels[s_i] = [s, 1.]
            elif s==0:
                labels[s_i] = ['negative', 0.]
            elif s==1:
                labels[s_i] = ['neutral', 0.5]
            else: # '2'
                labels[s_i] = ['positive', 1.]
            # end if
        # end for
        return labels
        
    @classmethod
    def get_sents(cls, testsuite_file, req_desc):
        tsuite, tsuite_dict = Utils.read_testsuite(testsuite_file)
        test_names = list(set(tsuite_dict['test_name']))
        sents = list()
        for tn in list(set(tsuite_dict['test_name'])):
            # checklist_test_name = tn.split('::')[-1]
            if tn in Macros.LC_MAP.keys() and req_desc == Macros.LC_MAP[tn]:
                # sents: List of sent
                # label: 0(neg), 1(neu) and 2(pos)
                sents = tsuite.tests[tn].data
                raw_labels = tsuite.tests[tn].labels
                labels = cls.get_labels(raw_labels,
                                        req_desc,
                                        len(tsuite.tests[tn].data))
                sents = [
                    (s_i, re.sub('\\s+', ' ', s), labels[s_i][0], labels[s_i][1])
                    for s_i, s in enumerate(sents)
                ]
                random.shuffle(sents)
                return sents
            # end if
        # end for
        return
        
    
    @classmethod
    def search(cls, req):
        # sent: (index, sentence)
        # label: (index, label score)
        sents = cls.get_sents(Macros.checklist_sa_dataset_file, req['description'])
        # req_obj = SearchOperator(req)
        # selected = sorted([(s[0],s[1],s[2],s[3]) for s in req_obj.search(sents)], key=lambda x: x[0])
        random.shuffle(sents)
        return sents

    
# class AirlineTweets:

#     @classmethod
#     def get_sents(cls, src_file):
#         # src_file: Tweets.csv file
#         import csv
#         rows = csv.DictReader(open(src_file))
#         labels, confs, airlines, sents, reasons = list(), list(), list(), list(), list()
#         for row in rows:
#             labels.append(row['airline_sentiment'])
#             # airlines.append(row['airline'])
#             s = Utils.replace_non_english_letter(row['text'])
#             sents.append(s)
#             reasons.append(row['negativereason'])
#         # end for
#         # labels = [Macros.sa_label_map[x] for x in labels]
#         return [(s_i, s, labels[s_i]) for s_i, s in enumerate(sents)]

#     @classmethod
#     def search(cls, req):
#         sents = cls.get_sents(Macros.tweet_file)
#         req_obj = SearchOperator(req)
#         if req_obj.search_reqs_list is not None:
#             selected = sorted([(s[0],s[1],s[2]) for s in req_obj.search(sents)], key=lambda x: x[0])
#         # end if
#         random.shuffle(selected)
#         return selected

    
# class DynasentRoundOne:

#     @classmethod
#     def get_sents(cls, src_file):
#         sents = list()
#         sent_i = 0
#         with open(yelp_src_filename) as f:
#             for line in f:
#                 d = json.loads(line)
#                 sents.append((sent_i, d['sentence'], d['gold_label']))
#                 sent_i += 1
#             # end for
#         # end with
#         return [(s_i,s,labels[s_i]) for s_i, s in sents]

#     @classmethod
#     def search(cls, req):
#         # sent: (index, sentence)
#         # label: (index, label score)
#         sents = cls.get_labels(Macros.dyna_r1_test_src_file)
#         req_obj = SearchOperator(req)
#         selected = sorted([(s[0],s[1],s[2]) for s in req_obj.search(sents)], key=lambda x: x[0])
#         random.shuffle(selected)
#         return selected


class Search:

    SEARCH_FUNC = {
        Macros.sa_task : {
            Macros.datasets[Macros.sa_task][0]: Sst.search,
            Macros.datasets[Macros.sa_task][1]: ChecklistTestsuite.search,
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
            
            if req["transform"] is not None and \
               dataset!=Macros.datasets[Macros.sa_task][1]:
                transform_obj = TransformOperator(req)
                selected = transform_obj.transform(selected)
            # end if
            yield {
                "requirement": req,
                "selected_inputs": selected
            }
        # end for
        return

    @classmethod
    def search_sentiment_analysis_per_req(cls, req, dataset):
        func = cls.SEARCH_FUNC[Macros.sa_task][dataset]
        selected = func(req)
            
        if req["transform"] is not None and \
           dataset!=Macros.datasets[Macros.sa_task][1]:
            transform_obj = TransformOperator(req)
            selected = transform_obj.transform(selected)
        # end if            
        return {
            "requirement": req,
            "selected_inputs": selected
        }
