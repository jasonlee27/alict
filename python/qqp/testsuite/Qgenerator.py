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

from checklist.editor import Editor
from checklist.expect import Expect
from nltk.corpus import wordnet
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from pathlib import Path

from ..utils.Macros import Macros
from ..utils.Utils import Utils
# from .sentiwordnet.Sentiwordnet import Sentiwordnet
from .Synonyms import Synonyms
from .Suggest import Suggest

random.seed(27)

WORDNET_TAG_MAP = {
    'n': 'NN',
    's': 'JJ',
    'a': 'JJ',
    'r': 'RB',
    'v': 'VB'
}


class Qgenerator:

    def __init__(self, seed, new_input_results, requirement):
        # new_inputs: List of
        # (masked new input, exp_cfg_from, exp_cfg_to, exp_cfg_pos, exp_word, input with exp_word)
        # if no expansion, then it will be empty
        self.new_inputs = new_input_results
        self.editor = Editor()
        self.nlp = spacy.load('en_core_web_md')
        self.seed = seed
        self.transform_action, self.transform_props = requirement['transform'].split()
    
    def generate_questions(self, is_input_pair=False):
        new_sent_dict = None
        if self.transform_action=='add':
            new_sent_dict = cls.add()
        elif self.transform_action=='replace':
            new_sent_dict = cls.replace(replace_in_pair=is_input_pair)
        elif self.transform_action=='remove':
            new_sent_dict = cls.remove()
        # end if
        return new_sent

    def _add_adj(self, sent):
        # find noun and sample one
        doc = self.nlp(sent)
        tokens = [str(t) for t in doc]
        nouns = [t_i for t_i, t in enumerate(doc) if t.tag_=='NN']
        random.shuffle(nouns)
        
        # insert masked token before the selected noun
        masked_tokens = tokens.insert(nouns[0], Macros.MASK)

        # genereate masked sentence
        masked_sent = Utils.detokenize(masked_tokens)

        # get the suggested word for the masked word
        word_suggestions = Suggest.get_word_suggestion(self.editor, masked_sent, mask_pos=None)
        new_sents = list()
        for w in list(set(word_suggestions)):
            doc = nlp(w)
            if doc[0].tag_=='JJ':
                new_sent = masked_sent.replace(Macros.MASK, f"<{w}>")
                new_sents.append(new_sent)
            # end if
        # end for
        return new_sents
        
    def add(self):
        if self.transform_props=='adj':
            # generating new question by adding word of specified tag of pos in question
            # seed first
            results = {
                self.seed: self._add_adj(self.seed),
                'exp_inputs': dict()
            }
            # exp inputs second
            for s in self.new_inputs:
                results['exp_inputs'][s[5]] = self._add_adj(s[5])
            # end for
            results['label'] = Macros.qqp_label_map['same']
            return results
        else:
            # generating new question by adding word in question
            pass
        # end if

    def _replace_synonyms(self, sent, replace_in_pair=False):
        if replace_in_pair:
            seed1, seed2 = sent.split('::')
            doc1, doc2 = self.nlp(seed1), self.nlp(seed2)
            tokens1, tokens2 = [str(t) for t in doc1], [str(t) for t in doc2]
            tokens_w_tag1, tokens_w_tag2 = [(str(t),t.tag_) for t in doc1], [(str(t),t.tag_) for t in doc2]
            tokens_w_synonyms1, tokens_w_synonyms2 = list(), list()
            for t_i, (t, p) in enumerate(tokens_w_tag1):
                synonyms = Synonyms.get_synonyms(self.nlp,t,p, num_synonyms=2)
                if any(synonyms):
                    tokens_w_synonyms1.append((t_i, t, t.tag_, synonyms))
                # end if
            # end for
            for t_i, (t, p) in enumerate(tokens_w_tag2):
                synonyms = Synonyms.get_synonyms(self.nlp,t,p, num_synonyms=2)
                if any(synonyms):
                    tokens_w_synonyms2.append((t_i, t, t.tag_, synonyms))
                # end if
            # end for
            random.shuffle(tokens_w_synonyms1)
            random.shuffle(tokens_w_synonyms2)
            new_sents = list()
            for t_i, t, t_pos, synonyms in tokens_w_synonyms1[Macros.num_synonyms_for_replace]:
                # replaced_seed1::seed2
                for s in synonyms:
                    new_tokens = tokens1[:t_i]+['<', s, '>']+tokens1[t_i+1:]
                    new_sents.append(f"{Utils.detokenize(new_tokens)}::{seed2}")
                # end for
            # end for
            for t_i, t, t_pos, synonyms in tokens_w_synonyms2[Macros.num_synonyms_for_replace]:
                # seed1::replaced_seed2
                for s in synonyms:
                    new_tokens = tokens2[:t_i]+['<', s, '>']+tokens2[t_i+1:]
                    new_sents.append(f"{seed1}::{Utils.detokenize(new_tokens)}")
                # end for
            # end for
            return new_sents
        # end if
        doc = self.nlp(sent)
        tokens = [str(t) for t in doc]
        tokens_w_tag = [(str(t),t.tag_) for t in doc]
        tokens_w_synonyms = list()
        for t_i, (t, p) in enumerate(tokens_w_tag):
            synonyms = Synonyms.get_synonyms(self.nlp,t,p, num_synonyms=2)
            if any(synonyms):
                tokens_w_synonyms.append((t_i, t, t.tag_, synonyms))
            # end if
        # end for
        random.shuffle(tokens_w_synonyms)
        new_sents = dict()
        for t_i, t, t_pos, synonyms in tokens_w_synonyms[Macros.num_synonyms_for_replace]:
            new_tokens = tokens[:t_i]+['<', tokens[t_i], '>']+tokens[t_i+1:]
            sent_from = Utils.detokenize(new_tokens)
            new_sents[sent_from] = list()
            for s in synonyms:
                new_tokens = tokens[:t_i]+['<', s, '>']+tokens[t_i+1:]
                new_sents[sent_from].append(Utils.detokenize(new_tokens))
            # end for
        # end for
        return new_sents
    
    def _replace_more_less(self, sent, replace_in_pair=False):
        def get_antonyms(self, word, pos):
            antonyms = list()
            for syn in wordnet.synsets(word):
                for lm in syn.lemmas():
                    if lm.antonyms():
                        antonym = lm.antonyms()[0].name()
                        antonym_pos = WORDNET_TAG_MAP[lm.antonyms()[0].synset().pos()]
                        if antonym_pos==pos:
                            antonyms.append(antonym)
                        # end if
                    # end if
                # end for
            # end for
            return list(set(antonyms))
        doc = self.nlp(sent)
        tokens = [str(t) for t in doc]
        tokens_w_tag = [(str(t),t.tag_) for t in doc]
        tokens_w_antonyms = list()
        for t_i, (t, p) in enumerate(tokens_w_tag[:-1]):
            if t.lower()=='more' or t.lower()=='less':
                antonyms = self.get_antonyms(tokens_w_tag[t_i+1][0], tokens_w_tag[t_i+1][1])
                if any(antonyms):
                    tokens_w_antonyms.append((t.lower(), t_i+1, tokens_w_tag[t_i+1][0], tokens_w_tag[t_i+1][1], antonyms))
                # end if
            # end if
        # end for
        random.shuffle(tokens_w_antonyms)
        new_sents = dict()
        for t_type, t_i, t, t_pos, antonyms in tokens_w_antonyms[Macros.num_synonyms_for_replace]:
            new_tokens = tokens[:t_i-1]+['<', t_type, tokens[t_i], '>']+tokens[t_i+1:]
            sent_from = Utils.detokenize(new_tokens)
            new_sents[sent_from] = list()
            for a in antonyms:
                alt_t_type = 'less' if t_type=='more' else 'more'
                new_tokens = tokens[:t_i-1]+['<', alt_t_type, a, '>']+tokens[t_i+1:]
                new_sents[sent_from].append(Utils.detokenize(new_tokens))
            # end for
        # end for
        return new_sents

    def replace(self, replace_in_pair=False):
        func_map = {
            'synonyms': self._replace_synonyms,
            'more_less': self._replace_more_less
        }
        self.nlp.add_pipe('spacy_wordnet', after='tagger', config={'lang': self.nlp.lang})
        new_sents = func_map[self.transform_props](self.seed, replace_in_pair=replace_in_pair)
        if replace_in_pair:
            results = {
                self.seed: new_sents,
                'label': Macros.qqp_label_map['same']
            }
            return results
        # end if
        results = {
            key: new_sent_dict[key]
            for key in new_sents.keys()
        }
        results['exp_inputs'] = dict()
        for s in self.new_inputs:
            new_exp_dict = func_map[self.transform_props](s[5])
            for key in new_exp_dict.keys():
                results['exp_inputs'][key] = new_exp_dict[key]
            # end for
        # end for
        results['label'] = Macros.qqp_label_map['same']
        nlp.remove_pipe('spacy_wordnet')
        return results

    def _remove_semantic_preserving_semantics(self, sent, targets):
        target_list = targets.split('|')
        doc = self.nlp(sent)
        tokens = [str(t) for t in doc]
        target_indices = [t_i for t_i, t in enumerate(doc) if str(t) in target_list]
        infact_indices = [t_i for t_i, t in enumerate(doc[:-1]) if str(t).lower()=='in' and str(doc[t_i+1]).lower()=='fact']
        new_sents = dict()
        if any(infact_indices):
            for ind in infact_indices:
                new_tokens = tokens[:ind]+['<']+tokens[ind]+tokens[ind+1]+['>']+tokens[ind+2:]
                sent_from = Utils.detokenize(new_tokens)
                new_sents[sent_from] = list()
                new_tokens = [str(t) for t_i, t in enumerate(doc) if t_i!=ind or t_i!=ind+1]
                new_sents[sent_from].append(Utils.detokenize(new_tokens))
            # end for
        # end if
        
        for t_i in enumerate(target_indices):
            new_tokens = tokens[:ind]+['<',tokens[ind],'>']+tokens[ind+1:]
            sent_from = Utils.detokenize(new_tokens)
            new_sents[sent_from] = list()
            new_tokens = [str(t) for _t_i, t in enumerate(doc) if _t_i!=t_i]
            new_sents[sent_from].append(Utils.detokenize(new_tokens))
        # end for
        return new_sents
    
    def remove(self):
        # generate seed question pair
        new_sent_dict = self._remove_semantic_preserving_semantics(self.seed, self.transform_props)
        results = {
            key: new_sent_dict[key]
            for key in new_sent_dict.keys()
        }

        # exp inputs second
        results['exp_inputs'] = dict()
        for s in self.new_inputs:
            new_exp_dict = self._remove_semantic_preserving_semantics(s[5], self.transform_props)
            for key in new_exp_dict.keys():
                results['exp_inputs'][key] = new_exp_dict[key]
            # end for
        # end for
        results['label'] = Macros.qqp_label_map['same']
        return results
