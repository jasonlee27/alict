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
from .Transform import TransformOperator
from ..requirement.Requirements import Requirements
from .sentiwordnet.Sentiwordnet import Sentiwordnet


# get pos/neg/neu words from SentiWordNet
# SENT_WORDS = Sentiwordnet.get_sent_words()
SENT_DICT = Sentiwordnet.get_sent_dict()

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

    # def search_by_label(self, sents, search_reqs):
    #     label = search_reqs["label"]
    
    def search_by_label(self, sents, search_reqs):
        label = search_reqs["label"]
        if label=="hate" or label=="positive" or label=="negative":
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
        word_product = [dict(zip(d, v)) for v in product(*pos_dict.values())]
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


class Hatexplain:

    @classmethod
    def get_labels(cls, annotators, is_binary_class=True):
        final_label = [ann['label'] for ann in annotators]
        final_label_id = max(final_label,key=final_label.count)
        if is_binary_class:
            if(final_label.count(final_label_id)==1):
                final_label_id = 'undecided'
            else:
                if(final_label_id in ['hatespeech','offensive']):
                    final_label_id='toxic'
                else:
                    final_label_id='non-toxic'
                # end if
            # end if
        else:
            if(final_label.count(final_label_id)==1):
                final_label_id = 'undecided'
            # end if
        # end if
        return final_label_id
    
    @classmethod
    def get_sents(cls,
                  sent_file: Path = Macros.hatexplain_data_file,
                  is_binary_class=True):
        raw_data_dict = Utils.read_json(sent_file)
        sents = list()
        for key, vals in raw_data_dict.items():
            label = cls.get_labels(
                vals['annotators'],
                is_binary_class=is_binary_class
            )
            if label!='undecided':
                post_id = vals['post_id']
                tokens = vals['post_tokens']
                sents.append({
                    'post_id': vals['post_id'],
                    'tokens': vals['post_tokens']
                    'label': label
                })
            # end if
        # end for
        return sents
    
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


class Search:

    SEARCH_FUNC = {
        Macros.sa_task : {
            Macros.datasets[Macros.sa_task][0]: Sst.search,
            Macros.datasets[Macros.sa_task][1]: ChecklistTestsuite.search,
            Macros.datasets[Macros.sa_task][2]: AirlineTweets.search,
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
            
            if req["transform"] is not None:
                transform_obj = TransformOperator(req)
                selected = transform_obj.transform(selected)
            # end if
            
            yield {
                "requirement": req,
                "selected_inputs": selected
            }
        # end for
        return


