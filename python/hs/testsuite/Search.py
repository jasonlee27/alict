# This script searches inputs in datasets that meet requirements

from typing import *
from pathlib import Path

import re, os
import sys
import json
import copy
import spacy
import random
import checklist
import numpy as np

from itertools import product
from checklist.editor import Editor
from checklist.test_types import MFT, INV, DIR
from checklist.test_suite import TestSuite

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from .Transform import TransformOperator
from .Synonyms import Synonyms
from ..requirement.Requirements import Requirements
from .sentiwordnet.Sentiwordnet import Sentiwordnet
from .hurtlex.Hurtlex import Hurtlex


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
        }

    def search(self, sents, nlp):
        selected = list()
        if self.search_reqs_list is None:
            return sents
        # end if
        for search_reqs in self.search_reqs_list:
            _sents = Utils.copy_list_of_dict(sents)
            for op, param in search_reqs.items():
                if len(_sents)>0 and param is not None:
                    _sents = self.search_method[op](_sents, search_reqs, nlp)
                # end if
            # end for
            for s in _sents:
                if s not in selected:
                    selected.append(s)
                # end if
            # end for
        # end for
        return selected

    def search_by_len(self, sents, search_reqs, nlp):
        param = search_reqs["length"]
        match = re.search(r"([<>]=?|==)(\d+)", param)
        op, _len = match.groups()
        results = list()
        for s_i, s in enumerate(sents):
            if eval(f"{len(s)-1} {op} {_len}"):
                results.append((
                    s_i, s['tokens'], s['label']
                ))
            # end if
        # end for
        return results
    
    def search_by_label(self, sents, search_reqs, nlp):
        label = search_reqs["label"]
        return [(s_i,s['tokens'],s['label']) for s_i, s in enumerate(sents) if label==s['label']]

    def get_synonyms(self, nlp, word: str):
        wpos = Synonyms.get_word_pos(nlp, word)
        return Synonyms.get_synonyms(nlp, wpos[0], wpos[1])

    def get_hurtlex_words(self, target_type):
        hurtlex_lex = Hurtlex.read_raw_data()
        for t in target_type.split('&'):
            if t.startswith('pos:'):
                # get target pos for hurtlex
                target_pos = t.split('pos:')[-1]
                words = Hurtlex.get_target_pos_words(hurtlex_lex, target_pos)
            elif t.startswith('cat:'):
                # get target categories for hurtlex
                target_cat = t.split('cat:')[-1]
                words = Hurtlex.get_target_cat_words(hurtlex_lex, target_cat)
            # end if
        # end for        
        words = [w['lemma'] for w in words]
        random.shuffle(words)
        return words
    
    def _search_by_word_include(self, sents, word_cond, nlp):
        # if word condition searches a specific group of words
        # such as "I <hate_syn> <hurtlex_nn>"
        selected = list()
        word_dict = dict()
        target_words = word_cond.split()
        _sents = sents.copy()
        sents_tmp = list()
        for tw in target_words:
            search = re.search("\<([^\<\>]+)\>", tw)
            if search:
                # find pattern
                if tw.startswith('<') and tw.endswith('>'):
                    # search any words in the values in the target template
                    target_template = tw.strip('<>')
                    
                    if target_template.endswith('_syn'):
                        _tw = target_template.split('_syn')[0]
                        _tw_syns = self.get_synonyms(nlp, _tw)
                        _tw_syns.append(_tw)
                        word_dict[tw] = list(set(_tw_syns))
                    elif target_template.endswith('isare'):
                        word_dict[tw] = ['is', 'are']
                    elif target_template.startswith('hurtlex_'):
                        target_type = target_template.split('hurtlex_')[1]
                        hurtlex_words = self.get_hurtlex_words(target_type)
                        word_dict[tw] = list(set(hurtlex_words))
                    # elif target_template.startswith('hatexplain_'):
                    # end if
                # end if
            else:
                # find word
                word_dict[tw] = [tw.lower()]
            # end if
            for w in word_dict[tw]:
                for s in _sents:
                    if w in s['tokens'] and s not in sents_tmp:
                        sents_tmp.append(s)
                    # end if
                # end for
            # end for
            _sents = sents_tmp.copy()
            sents_tmp = list()
        # end for
        # perturb the words for searching in the dataset
        word_product = [dict(zip(word_dict, v)) for v in product(*word_dict.values())]
        search_words = [list(wp.values()) for wp in word_product]

        # find any sentences with any search_words
        for s in _sents:
            for sw in search_words:
                if Utils.is_a_in_x(sw, s['tokens']):
                    selected.append(s)
                # end if
            # end for
        # end for
        return selected

    def get_pospat_to_wordproduct(self, pos_pattern):
        results = list()
        pos_dict = {
            p: WORD2POS_MAP[p]
            for p in pos_pattern.split('_')
        }
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
        
    def search_by_include(self, sents, search_reqs, nlp):
        params = search_reqs["include"]
        if type(params)==dict:
            params = [params]
        # end if
        selected_indices = list()
        _sents = sents.copy()
        for param in params:
            word_include = param["word"]
            tpos_include = param["POS"]
            
            if word_include is not None:
                for w in word_include: # AND relationship
                    _sents = self._search_by_word_include(_sents, w, nlp)
                # end for
            # end if
        
            if tpos_include is not None:
                pass
            # end if
        # end for
        result = _sents
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
                    'tokens': [t.lower() for t in vals['post_tokens']],
                    'label': label
                })
            # end if
        # end for
        return sents
    
    @classmethod
    def search(cls, req, nlp):
        sents = cls.get_sents(Macros.hatexplain_data_file, is_binary_class=True)
        req_obj = SearchOperator(req)

        if req_obj.search_reqs_list is not None:
            # selected = sorted([s for s in req_obj.search(sents, nlp)], key=lambda x: x['post_id'])
            selected = req_obj.search(sents, nlp)
        else:
            selected = sents
        # end if
        random.shuffle(selected)
        return selected

    
class Hatecheck:

    # TODO: complete func mapping
    FUNCTIONALITY_MAP = {
        Macros.OUR_LC_LIST[0]: 'derog_neg_emote_h',
        Macros.OUR_LC_LIST[1]: 'derog_neg_attrib_h',
        Macros.OUR_LC_LIST[2]: 'derog_dehum_h',
        Macros.OUR_LC_LIST[3]: 'derog_neg_impl_h',
        Macros.OUR_LC_LIST[4]: 'threat_dir_h',
        Macros.OUR_LC_LIST[5]: 'threat_norm_h',
        Macros.OUR_LC_LIST[6]: 'slur_h',
        Macros.OUR_LC_LIST[7]: 'slur_homonym_nh',
        Macros.OUR_LC_LIST[8]: 'slur_reclaimed_nh',
        Macros.OUR_LC_LIST[9]: 'profanity_h',
        Macros.OUR_LC_LIST[10]: 'profanity_nh',
        Macros.OUR_LC_LIST[11]: 'ref_subs_clause_h',
        Macros.OUR_LC_LIST[12]: 'ref_subs_sent_h',
        Macros.OUR_LC_LIST[13]: 'negate_pos_h',
        Macros.OUR_LC_LIST[14]: 'negate_neg_nh',
        Macros.OUR_LC_LIST[15]: 'phrase_question_h',
        Macros.OUR_LC_LIST[16]: 'phrase_option_h',
        Macros.OUR_LC_LIST[17]: 'ident_neutral_nh',
        Macros.OUR_LC_LIST[18]: 'ident_pos_nh',
        Macros.OUR_LC_LIST[19]: 'counter_quote_nh',
        Macros.OUR_LC_LIST[20]: 'counter_ref_nh',
        Macros.OUR_LC_LIST[21]: 'target_obj_nh',
        Macros.OUR_LC_LIST[22]: 'target_indiv_nh',
        Macros.OUR_LC_LIST[23]: 'target_group_nh'
    }

    @classmethod
    def get_sents(cls,
                  sent_file: Path = Macros.hatecheck_data_file):
        raw_data_dict = Utils.read_sv(sent_file,
                                      delimeter=',',
                                      is_first_attributes=True)
        atts = raw_data_dict['attributes']
        data = raw_data_dict['lines']
        sents = list()
        func_att_index = atts.index('functionality')
        sent_att_index = atts.index('test_case')
        label_att_index = atts.index('label_gold')
        orig_label_keys = list(Macros.hs_label_map)
        for d in data:
            # attributes: ,functionality,case_id,templ_id,test_case,label_gold,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,label_10,count_label_h,count_label_nh,label_annot_maj
            sents.append({
                'func': d[func_att_index],
                'sent': re.sub('\\s+', ' ', d[sent_att_index].strip().lower()),
                'label': orig_label_keys[0] if d[label_att_index]=='hateful' else orig_label_keys[1]
            })
        # end for
        return sents

    @classmethod
    def add_template(cls, t, editor, template):
        if t is None:
            t = editor.template(template['sent'],
                                labels=template['label'],
                                save=True)
        else:
            t += editor.template(template['sent'],
                                 labels=template['label'],
                                 save=True)
        # end if
        return t

    @classmethod
    def write_testsuites(cls,
                         hatecheck_data_file: Path = Macros.hatecheck_data_file,
                         hatecheck_testsuite_file: Path = Macros.hatecheck_testsuite_file):
        sents = cls.get_sents(hatecheck_data_file)
        editor = Editor()
        suite = TestSuite()
        _sents = dict()
        for s in sents:
            func_desc = None
            for key, val in cls.FUNCTIONALITY_MAP.items():
                if val==s['func']:
                    func_desc = (key, val)
                    break
                # end if
            # end for
            if func_desc is not None:
                sent = s['sent']
                label = 1 if s['label']=='hateful' else 0
                func_key = f"{func_desc[0]}::{func_desc[1]}"
                if func_key not in _sents.keys():
                    _sents[func_key] = [{
                        'sent': sent,
                        'label': label
                    }]
                else:
                    _sents[func_key].append({
                        'sent': sent,
                        'label': label
                    })
                # end if
            # end if
        # end for
        for func_desc in _sents.keys():
            t = None
            for d in _sents[func_desc]:
                t = cls.add_template(t, editor, d)
            # end for
            test = MFT(**t)
            _func_desc, func_label = func_desc.split('::')
            suite.add(test,
                      name=func_desc,
                      capability=func_label,
                      description=_func_desc)
        # end for
        res_dir = Macros.hatecheck_testsuite_file.parent
        res_dir.mkdir(parents=True, exist_ok=True)
        suite.save(hatecheck_testsuite_file)
        return
    
    @classmethod
    def search(cls, req, nlp):
        sents = cls.get_sents(Macros.hatecheck_data_file)
        req_desc = req['description']
        sh_req_desc = cls.FUNCTIONALITY_MAP[req_desc]
        selected = list()
        for s_i, s in enumerate(sents):
            if s['func']==sh_req_desc:
                selected.append((s_i, s['sent'], s['label']))
        # end if
        random.shuffle(selected)
        return selected


class Search:
    
    SEARCH_FUNC = {
        Macros.hs_task : {
            Macros.datasets[Macros.hs_task][0]: Hatexplain.search,
            Macros.datasets[Macros.hs_task][1]: Hatecheck.search
        }
    }
    
    @classmethod
    def get_dataset(cls, task_name, dataset_name):
        pass

    @classmethod
    def search_hatespeech(cls, requirements, dataset, nlp):
        func = cls.SEARCH_FUNC[Macros.hs_task][dataset]
        for req in requirements:
            selected = func(req, nlp)
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

    @classmethod
    def search_hatespeech_per_req(cls, req, dataset, nlp):
        func = cls.SEARCH_FUNC[Macros.hs_task][dataset]
        selected = func(req, nlp)
        if req["transform"] is not None and \
           dataset!=Macros.datasets[Macros.hs_task][1]:
            transform_obj = TransformOperator(req)
            selected = transform_obj.transform(selected)
        # end if            
        return {
            "requirement": req,
            "selected_inputs": selected
        }
