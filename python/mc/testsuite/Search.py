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

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..requirement.Requirements import Requirements
from .Synonyms import Synonyms
# from .sentiwordnet.Sentiwordnet import Sentiwordnet

# get name and location data
basic = Utils.read_json(Macros.dataset_dir / 'checklist' / 'lexicons' / 'basic.json')
names = Utils.read_json(Macros.dataset_dir / 'checklist' / 'names.json')
name_set = { x:set(names[x]) for x in names }
NAME_LOC_DICT = {
    'name': names,
    'name_set': name_set,
    'city': basic['city'],
    'country': basic['country'],
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
        }

    def search(self, sents):
        # sents: list of dictionary with keys of question, answers, context and id
        selected, selected_ids = list(), set()
        for search_reqs in self.search_reqs_list:
            _sents = sents.copy()
            for input_type, reqs in search_reqs.items(): # req for question and context
                if reqs is not None:
                    for op, param in reqs.items():
                        if len(_sents)>0:
                            _sents = self.search_method[op](_sents, reqs, input_type)
                        # end if
                    # end for
                # end if
            # end for
            for s in _sents:
                selected_ids.add(s['id'])
            # end for
        # end for
        return [s for s in sents if s['id'] in selected_ids]

    def search_by_len(self, sents, search_reqs, search_input_type):
        param = search_reqs["length"]
        match = re.search(r"([<>]=?|==)(\d+)", param)
        op, _len = match.groups()
        selected = list()
        if search_input_type=='question':
            _sents = [(s_i,s['id'],Utils.tokenize(s['question'])) for s_i, s in enumerate(sents)]
        else:
            _sents = [(s_i,s['id'],Utils.tokenize(s['context'])) for s_i, s in enumerate(sents)]
        # end for
        for s_i, s_id, s in _sents:
            if eval(f"{len(s)}{op}{_len}"):
                selected.append(sents[s_i])
            # end if
        # end for
        return selected

    def search_by_label(self, sents, search_reqs):
        # label: 0: different, 1: same
        label = search_reqs["label"]
        selected = list()
        for sent in sents:
            answers = [a[0] for a in sent['answers']]
            if label in answers:
                selected.append(sent)
            # end if
        # end for
        return selected

    def _search_by_synonym_existence(self, sents, search_input_type, isinclude=True):
        # sents: (s_i, tokenized sentence, label)
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe('spacy_wordnet', after='tagger', config={'lang': nlp.lang})
        selected = list()
        for sent in sents:
            s = Utils.detokenize(sent[search_input_type])
            doc = nlp(s)
            synonyms = None
            for t in doc:
                synonyms = Synonyms.get_synonyms(nlp, str(t), t.pos_)
                if any(synonyms) and isinclde:
                    selected.append(sent)
                    break
                # end if
            # end for
            if synonyms is None and not isinclude:
                selected.append(sent)
            # end if
        # end for
        return selected

    def _search_by_comparison_existence(self, sents, search_input_type, isinclude=True):
        comp_pat = r"^[Which|Who|When|What|Whose| which| who| when| what| whose].+er(.+?)than(.+?)\?"
        selected = list()
        for sent in sents:
            s = Utils.detokenize(sent[search_input_type])
            if search_input_type=='question':
                # we split by the dot delimeter to get the real question in question
                s = s.split('.')[-1]
            # end if
            search = re.search(comp_pat, s)
            if search:
                selected.append(sent)
            # end if
        # end for
        return selected

    def _search_by_property_existence(self, sents, search_input_type, isinclude=True):
        age_pat = r"[What is the age|How old| what is the age].+\?"
        size_pat = r"[What is the size|What size|Which sized| what is the size| what size].+\?"
        shape_pat = r"[What is the shape|What shape| what is the shape| what shape].+\?"
        color_pat = r"[What is the color|What color|Which color| what is the color| what color].+\?"
        selected = list()
        for sent in sents:
            s = Utils.detokenize(sent[search_input_type])
            if search_input_type=='question':
                # we split by the dot delimeter to get the real question in question
                s = s.split('.')[-1]
            # end if
            age_search = re.search(age_pat, s)
            size_search = re.search(size_pat, s)
            shape_search = re.search(shape_pat, s)
            color_search = re.search(color_pat, s)
            if age_search or size_search or shape_search or color_search:
                selected.append(sent)
            # end if
        # end for
        return selected

    def _search_by_superlative_existence(self, sents, search_input_type, isinclude=True):
        comp_pat = r"^[Which|Who|Whose|Where|When|What|Whose|How many|How much].+[the most|the least](.+?)\?"
        selected = list()
        for sent in sents:
            s = Utils.detokenize(sent[search_input_type])
            if search_input_type=='question':
                # we split by the dot delimeter to get the real question in question
                s = s.split('.')[-1]
            # end if
            search = re.search(comp_pat, s)
            if search:
                selected.append(sent)
            # end if
        # end for
        return selected
    
    def _search_by_word_include(self, sents, word_cond, search_input_type):
        search = re.search("\<([^\<\>]+)\>", word_cond)
        selected = list()
        if search:
            # if word condition searches a specific group of words
            # such as "name of person, location"
            word_group = search.group(1)
            if word_group=="comparison":
                selected = self._search_by_comparison_existence(sents, search_input_type)
            if word_group=="synonym":
                selected = self._search_by_synonym_existence(sents, search_input_type)
            # end if
        else:
            word_or_list = word_cond.split("|")
            for sent in sents:
                isfound = [True for t in sent[search_input_type] if t in word_or_list]
                if any(isfound):
                    selected.append(sent)
                # end if
            # end for
        # end if
        return selected
    
    def _search_by_pos_include(self, sents, pos_cond, search_input_type):
        # sents: (s_i, tokenized sentence, label)
        # pos_cond: e.g. adj, verb, noun
        pos_map = {'adj': 'JJ', 'verb': 'VB', 'noun': 'NN'}
        nlp = spacy.load("en_core_web_sm")
        selected = list()
        for sent in sents:
            doc = nlp(Utils.detokenize(sent[search_input_type]))
            for t in doc:
                if t.tag_.startswith(pos_map[pos_cond]):
                    selected.append(sent)
                    break
                # end if
            # end for
        # end for
        return selected
        
    def search_by_include(self, sents, search_reqs, search_input_type):
        params = search_reqs["include"]
        if type(params)==dict:
            params = [params]
        # end if

        # tokenization for better search
        _sents = sents.copy()
        for s_i in range(len(_sents)):
            _sents[s_i][search_input_type] = Utils.tokenize(_sents[s_i][search_input_type])
        # end for
        
        selected_indices = set()
        for param in params:
            word_include = param["word"]
            pos_include = param["POS"]
            
            if word_include is not None:
                for w in word_include: # AND relationship
                    _sents = self._search_by_word_include(_sents, w, search_input_type)
                # end for
            # end if
        
            if pos_include is not None:
                temp_sents = list()
                for p in pos_include:
                    _sents = self._search_by_pos_include(_sents, p, search_input_type)
                # end for
            # end if
            
            for sent in _sents:
                selected_indices.add(sent['id'])
            # end for
        # end for
        
        result = list()
        # detokenization
        for s_i in range(len(sents)):
            if sents[s_i]['id'] in selected_indices:
                sents[s_i][search_input_type] = Utils.detokenize(sents[s_i][search_input_type])
                result.append(sents[s_i])
            # end if
        # end for
        return result

    def _search_by_word_exclude(self, sents, word_cond, search_input_type):
        search = re.search("\<([^\<\>]+)\>", word_cond)
        selected = list()
        if search:
            # if word condition searches a specific group of words
            # such as "name of person, location"
            word_group = search.group(1)
            if word_group=="comparison":
                selected = self._search_by_comparison_existence(sents, search_input_type)
                ids_comparison = [s['id'] for s in selected]
                selected = [s for s in sents if s['id'] not in ids_comparison]
            if word_group=="synonym":
                selected = self._search_by_synonym_existence(sents, search_input_type)
                ids_comparison = [s['id'] for s in selected]
                selected = [s for s in sents if s['id'] not in ids_comparison]
            # end if
        else:
            word_or_list = word_cond.split("|")
            for sent in sents:
                isfound = [True for t in sent[search_input_type] if t in word_or_list]
                if not any(isfound):
                    selected.append(sent)
                # end if
            # end for
        # end if
        return selected

    def _search_by_pos_exclude(self, sents, pos_cond, search_input_type):
        # sents: (s_i, tokenized sentence, label)
        # pos_cond: e.g. adj, verb, noun
        pos_map = {'adj': 'JJ', 'verb': 'VB', 'noun': 'NN'}
        nlp = spacy.load("en_core_web_sm")
        selected = list()
        for sent in sents:
            pos_found = False
            doc = nlp(Utils.detokenize(sent[search_input_type]))
            for t in doc:
                if t.tag_.startswith(pos_map[pos_cond]):
                    pos_found = True
                    break
                # end if
            # end for
            if not pos_found:
                selected.append(sent)
            # end if
        # end for
        return selected

    def search_by_exclude(self, sents, search_reqs):
        params = search_reqs["exclude"]
        # tokenization for better search
        _sents = sents.copy()
        for s_i in range(len(_sents)):
            _sents[s_i][search_input_type] = Utils.tokenize(_sents[s_i][search_input_type])
        # end for
        
        selected_indices = set()
        if type(params)==dict:
            params = [params]
        # end if
        for param in params:
            word_exclude = param["word"]
            pos_exclude = param["POS"]
            if word_exclude is not None:
                for w in word_exclude:
                    _sents = self._search_by_word_exclude(_sents, w)
                # end for
            # end if

            if pos_exclude is not None:
                for p in pos_exclude:
                    _sents = self._search_by_pos_exclude(_sents, p)
                # end for
            # end if

            for sent in _sents:
                selected_indices.add(sent['id'])
            # end for
        # end for
        
        result = list()
        # detokenization
        for s_i in range(len(sents)):
            if sents[s_i]['id'] in selected_indices:
                sents[s_i][search_input_type] = Utils.detokenize(sents[s_i][search_input_type])
                result.append(sents[s_i])
            # end if
        # end for
        return result

    
class Squad:

    @classmethod
    def get_sents(cls, squad_file):
        data = list()
        f = Utils.read_json(squad_file)
        for t in f['data']:
            for p in t['paragraphs']:
                context = p['context']
                for qa in p['qas']:
                    d = {
                        'id': qa['id'],
                        'context': context,
                        'question': qa['question'],
                        'answers': set([(x['text'], x['answer_start']) for x in qa['answers']])
                    }
                    if any(d['answers']):
                        data.append(d)
                    # end if
                # end for
            # end for
        # end for
        return data
    
    @classmethod
    def search(cls, req):
        # sents: Dict
        req_obj = SearchOperator(req)
        sents = cls.get_sents(Macros.squad_valid_file)
        selected = None
        if req_obj.search_reqs_list is not None:
            selected = sorted([s for s in req_obj.search(sents)], key=lambda x: x['id'])
        else:
            selected = sents
        # end if
        random.shuffle(selected)
        return selected


# class ChecklistTestsuite:

#     @classmethod
#     def get_sents(cls, testsuite_file):
#         tsuite, tsuite_dict = read_testsuite(testsuite_file)
#         sents, raw_labels = list(), list()
#         for test_name in test_names:
#             # sents: List of sent
#             # label: 0(neg), 1(neu) and 2(pos)
#             sents.extend(tsuite.tests[test_name].data) 
#             raw_labels.extend(tsuite.tests[test_name].labels)
#         # end for        
#         labels = dict()
#         for s_i, s in enumerate(raw_labels):
#             if s=='0':
#                 labels[s_i] = "negative"
#             elif s=='1':
#                 labels[s_i] = "neutral"
#             else:
#                 labels[s_i] = "positive"
#             # end if
#         #end for
#         return [(s_i, s, labels[s_i]) for s_i, s in enumerate(sents)]
    
#     @classmethod
#     def search(cls, req):
#         # sent: (index, sentence)
#         # label: (index, label score)
#         sents = cls.get_sents(Macros.checklist_sa_dataset_file)
#         req_obj = SearchOperator(req)
#         selected = sorted([(s[0],s[1],s[2]) for s in req_obj.search(sents)], key=lambda x: x[0])
#         random.shuffle(selected)
#         return selected

    
class Search:

    SEARCH_FUNC = {        
        Macros.datasets[0]: Squad.search,
        Macros.datasets[1]: None
    }

    @classmethod
    def search_mc(cls, requirements, dataset):
        func = cls.SEARCH_FUNC[dataset]
        for req in requirements:
            selected = func(req)
            yield {
                "requirement": req,
                "selected_inputs": selected
            }
        # end for
        return


