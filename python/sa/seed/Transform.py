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
from itertools import product, permutations
from checklist.expect import Expect
from pathlib import Path

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..semexp.Synonyms import Synonyms
from .sentiwordnet.Sentiwordnet import Sentiwordnet
from ...hs.seed.hatecheck.Hatecheck import Hatecheck


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

# CONTRACTION_MAP = {
#     "ain't": "is not", "aren't": "are not", "can't": "cannot",
#     "can't've": "cannot have", "could've": "could have", "couldn't":
#     "could not", "didn't": "did not", "doesn't": "does not", "don't":
#     "do not", "hadn't": "had not", "hasn't": "has not", "haven't":
#     "have not", "he'd": "he would", "he'd've": "he would have",
#     "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y":
#     "how do you", "how'll": "how will", "how's": "how is",
#     "I'd": "I would", "I'll": "I will", "I'm": "I am",
#     "I've": "I have", "i'd": "i would", "i'll": "i will",
#     "i'm": "i am", "i've": "i have", "isn't": "is not",
#     "it'd": "it would", "it'll": "it will", "it's": "it is", "ma'am":
#     "madam", "might've": "might have", "mightn't": "might not",
#     "must've": "must have", "mustn't": "must not", "needn't":
#     "need not", "oughtn't": "ought not", "shan't": "shall not",
#     "she'd": "she would", "she'll": "she will", "she's": "she is",
#     "should've": "should have", "shouldn't": "should not", "that'd":
#     "that would", "that's": "that is", "there'd": "there would",
#     "there's": "there is", "they'd": "they would",
#     "they'll": "they will", "they're": "they are",
#     "they've": "they have", "wasn't": "was not", "we'd": "we would",
#     "we'll": "we will", "we're": "we are", "we've": "we have",
#     "weren't": "were not", "what're": "what are", "what's": "what is",
#     "when's": "when is", "where'd": "where did", "where's": "where is",
#     "where've": "where have", "who'll": "who will", "who's": "who is",
#     "who've": "who have", "why's": "why is", "won't": "will not",
#     "would've": "would have", "wouldn't": "would not",
#     "you'd": "you would", "you'd've": "you would have",
#     "you'll": "you will", "you're": "you are", "you've": "you have"
# }

WORD2POS_MAP = {
    'demonstratives': ['This', 'That', 'These', 'Those'],
    'AUXBE': ['is', 'are']
}

NEG_OF_NEG_AT_THE_END_PHRASE_TEMPLATE = {
    "prefix": ["I agreed that", "I thought that"],
    "sent": [],
    "postfix": ["but it wasn't", "but I didn't"]
}

DISAGREEMENT_PHRASE = {
    "prefix": ["I wouldn't say,", "I do not think,", "I don't agree with,"],
    "middlefix": [],
    # "postfix": ["that"],
    "sent": []
}

CUR_NEG_TEMPORAL_PHRASE_TEMPLATE = {
    "past": [
        "Previously, I used to like it saying that",
        "Last time, I agreed with saying that",
        "I liked it much as to say that"
    ],
    "sent": [],
    "but": ['but', 'although', 'on the other hand'],
    "current": ["now I don't like it.", "now I hate it."]
}

CUR_POS_TEMPORAL_PHRASE_TEMPLATE = {
    "past": [
        "I used to disagree with saying that",
        "Last time, I didn't like it saying that",
        "I hated it much as to say that"
    ],
    "sent": [],
    "but": ['but', 'although', 'on the other hand'],
    "current": ["now I like it."]
}

SRL_PHASE_TEMPLATE = {
    "prefix": [
        "Some people think that",
        "Many people agree with that",
        "They think that",
        "You agree with that"
    ],
    "sent1": [],
    "middlefix": ["but I think that"],
    "sent2": [],
}

QUESTIONIZE_PHRASE_TEMPLATE = {
    "prefix": [
        "Do I think that",
        "Do I agree that"
    ],
    "sent": [],
    "answer": []
}


class TransformOperator:

    def __init__(self,
                 requirements,
                 editor=None
                 ):
        
        self.editor = editor # checklist.editor.Editor()
        self.capability = requirements['capability']
        self.description = requirements['description']
        # self.search_dataset = search_dataset
        self.transform_reqs = requirements['transform']
        # self.inv_replace_target_words = None
        # self.inv_replace_forbidden_words = None
        self.transform_func = self.transform_reqs.split()[0]
        self.transform_props = None
        if len(self.transform_reqs.split())>1:
            self.transform_props = self.transform_reqs.split()[1]
        # end if
        # self.dir_adding_phrases = None

        # # Find INV transformation operations
        # if transform_reqs["INV"] is not None:
        #     self.set_inv_env(transform_reqs["INV"])
        # # end if
        
        # if transform_reqs["DIR"] is not None:
        #     self.set_dir_env(transform_reqs["DIR"])
        # # end if

    # def set_inv_env(self, inv_transform_reqs):
    #     if len(inv_transform_reqs.split())==2:
    #         func = inv_transform_reqs.split()[0]
    #         _property = None
    #         woi = inv_transform_reqs.split()[1]
    #     else:
    #         func = inv_transform_reqs.split()[0]
    #         _property = inv_transform_reqs.split()[1]
    #         woi = inv_transform_reqs.split()[2]
    #     # end if
    #     if func=="replace":
    #         self.inv_replace_target_words = list()
    #         self.inv_replace_forbidden_words = list()
    #         if _property=="neutral" and woi=="word":
    #             self.inv_replace_target_words = set(SENT_DICT[f"{_property}_adj"] + \
    #                                                 SENT_DICT[f"{_property}_verb"] + \
    #                                                 SENT_DICT[f"{_property}_noun"])
    #             self.inv_replace_forbidden_words = set(['No', 'no', 'Not', 'not', 'Nothing', 'nothing', 'without', 'but'] + \
    #                                                    SENT_DICT["positive_adj"] + \
    #                                                    SENT_DICT[f"negative_adj"] + \
    #                                                    SENT_DICT[f"positive_verb"] + \
    #                                                    SENT_DICT[f"negative_verb"] + \
    #                                                    SENT_DICT[f"positive_noun"] + \
    #                                                    SENT_DICT[f"negative_noun"])
    #         # end if
    #     # end if
    #     self.transformation_funcs = f"INV:{func}:{_property}:{woi}"
    #     return

    # def set_dir_env(self, dir_transform_reqs):
    #     if len(dir_transform_reqs.split())==2:
    #         func = dir_transform_reqs.split()[0]
    #         _property = None
    #         woi = dir_transform_reqs.split()[2]
    #     else:
    #         func = dir_transform_reqs.split()[0]
    #         _property = dir_transform_reqs.split()[1]
    #         woi = dir_transform_reqs.split()[2]
    #     # end if
    #     if func=="add":
    #         if _property=="positive" and woi=="phrase":
    #             nlp = spacy.load('en_core_web_sm')
    #             nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
    #             self.editor.add_lexicon('like_VB', ['like']+Synonyms.get_synonyms(nlp, 'like', 'VB', num_synonyms=10))
    #             self.editor.add_lexicon('love_VB', ['love']+Synonyms.get_synonyms(nlp, 'love', 'VB', num_synonyms=10))
    #             self.editor.add_lexicon('great_ADJ', ['great']+Synonyms.get_synonyms(nlp, 'great', 'ADJ', num_synonyms=10))
    #             self.dir_adding_phrases = self.editor.template('I {like_VB} you.').data
    #             self.dir_adding_phrases += self.editor.template('I {love_VB} it.').data
    #             self.dir_adding_phrases += self.editor.template('You are {great_ADJ}.').data
    #             self.dir_expect_func = Expect.pairwise(self.diff_up)
    #         elif _property=="negative" and woi=="phrase":
    #             nlp = spacy.load('en_core_web_sm')
    #             nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
    #             self.editor.add_lexicon('hate_VB', ['hate']+Synonyms.get_synonyms(nlp, 'hate', 'VB', num_synonyms=10))
    #             self.editor.add_lexicon('dislike_VB', ['dislike']+Synonyms.get_synonyms(nlp, 'dislike', 'VB', num_synonyms=10))
    #             self.editor.add_lexicon('bad_ADJ', ['bad']+Synonyms.get_synonyms(nlp, 'bad', 'ADJ', num_synonyms=10))
    #             self.dir_adding_phrases = self.editor.template('I {hate_VB} you.').data
    #             self.dir_adding_phrases += self.editor.template('I {dislike_VB} it.').data
    #             self.dir_adding_phrases += self.editor.template('You are {bad_ADJ}.').data
    #             self.dir_expect_func = Expect.pairwise(self.diff_down)
    #         # end if
    #         self.transformation_funcs = f"DIR:{func}:{_property}:{woi}"
    #     # end if
    #     return
                
    # def replace(self, d):
    #     examples = list()
    #     subs = list()
    #     target_words = set(self.inv_replace_target_words)
    #     forbidden = self.inv_replace_forbidden_words
        
    #     words_in = [x for x in d.split() if x in target_words]
    #     if not words_in:
    #         return None
    #     # end if
    #     for w in words_in:
    #         suggestions = [
    #             x for x in self.editor.suggest_replace(d, w, beam_size=5, words_and_sentences=True)
    #             if x[0] not in forbidden
    #         ]
    #         examples.extend([x[1] for x in suggestions])
    #         subs.extend(['%s -> %s' % (w, x[0]) for x in suggestions])
    #     # end for
    #     if examples:
    #         idxs = np.random.choice(len(examples), min(len(examples), 10), replace=False)
    #         return [examples[i] for i in idxs]#, [subs[i] for i in idxs])
    #     # end if

    # # functions for adding positive/negative phrase
    # def add_phrase(self):
    #     def pert(d):
    #         while d[-1].pos_ == 'PUNCT':
    #             d = d[:-1]
    #         # end while
    #         d = d.text
    #         ret = [d + '. ' + x for x in self.dir_adding_phrases]
    #         idx = np.random.choice(len(ret), min(len(ret),10), replace=False)
    #         ret = [ret[i] for i in idx]
    #         return ret
    #     return pert
    
    # def positive_change(self, orig_conf, conf):
    #     softmax = type(orig_conf) in [np.array, np.ndarray]
    #     if not softmax or orig_conf.shape[0] != 3:
    #         raise(Exception('Need prediction function to be softmax with 3 labels (negative, neutral, positive)'))
    #     # end if
    #     return orig_conf[0] - conf[0] + conf[2] - orig_conf[2]

    # def diff_up(self, orig_pred, pred, orig_conf, conf, labels=None, meta=None):
    #     tolerance = 0.1
    #     change = self.positive_change(orig_conf, conf)
    #     if change + tolerance >= 0:
    #         return True
    #     else:
    #         return change + tolerance
    #     # end if
        
    # def diff_down(self, orig_pred, pred, orig_conf, conf, labels=None, meta=None):
    #     tolerance = 0.1
    #     change = self.positive_change(orig_conf, conf)
    #     if change - tolerance <= 0:
    #         return True
    #     else:
    #         return -(change - tolerance)
    #     # end if

    # def random_string(self, n):
    #     return ''.join(np.random.choice([x for x in string.ascii_letters + string.digits], n))
        
    # def random_url(self, n=6):
    #     return 'https://t.co/%s' % self.random_string(n)
    
    # def random_handle(self, n=6):
    #     return '@%s' % self.random_string(n)

    # def add_irrelevant(self, sentence):
    #     urls_and_handles = [self.random_url(n=6) for _ in range(5)] + [self.random_handle() for _ in range(5)]
    #     irrelevant_before = ['@airline '] + urls_and_handles
    #     irrelevant_after = urls_and_handles 
    #     rets = ['%s %s' % (x, sentence) for x in irrelevant_before ]
    #     rets += ['%s %s' % (sentence, x) for x in irrelevant_after]
    #     return rets

    def transform(self, sents):
        transform_func_map = {
            'add': self.add,
            'negate': self.negate,
            'srl': self.srl,
            'questionize': self.questionize
        }
        new_sents = transform_func_map[self.transform_func](sents, self.transform_props)
        return new_sents

    def _change_temporalness_template(self, sents):
        # sents: List[(s_i, sentence, label)]
        # convert every generate templates into temporal awareness formated templates
        # each template keys: sent, values, label
        results = list()
        res_idx = 0
        for sent in sents:
            new_sents = list()
            word_dict = dict()
            label = sent[2]
            new_label = None
            if label=='positive': # posive previously, but negative now
                word_dict = CUR_NEG_TEMPORAL_PHRASE_TEMPLATE
                word_dict['sent'] = [f"\"{sent[1]}\","]
                new_label = 'negative'
            elif label=='negative':
                word_dict = CUR_POS_TEMPORAL_PHRASE_TEMPLATE
                word_dict['sent'] = [f"\"{sent[1]}\","]
                new_label = 'positive'
            else:
                raise(f"label \"{label}\" is not available")
            # end if
            word_product = [dict(zip(word_dict, v)) for v in product(*word_dict.values())]
            for wp in word_product:
                new_sent = " ".join(list(wp.values()))
                results.append((f'new_{sent[0]}_{res_idx}', new_sent, new_label, None))
                res_idx += 1
            # end for
        # end for
        random.shuffle(results)
        return results

    def add(self, sents, props):
        if props=='temporal_awareness':
            return self._change_temporalness_template(sents)
        # end if
        return sents
    
    def _get_negationpattern_to_wordproduct(self, negation_pattern, value_dict):
        results = list()
        pos_dict = {
            p: value_dict[p]
            for p in negation_pattern.split('_')
        }
        word_product = [dict(zip(pos_dict, v)) for v in product(*pos_dict.values())]
        for wp in word_product:
            results.append(" ".join(list(wp.values())))
        # end for
        return results
    
    def negate(self, sents, negation_pattern):
        # sents: List[(s_i, sentence, label)]
        from itertools import product
        results = list()
        if negation_pattern=='AUXBE$':
            # negation of negative at the end
            negation_pattern = negation_pattern[:-1]
            res_idx = 0
            for sent in sents:
                word_dict = NEG_OF_NEG_AT_THE_END_PHRASE_TEMPLATE
                word_dict['sent'] = [f"\"{sent[1]}\","]
                label = sent[2]
                word_product = [dict(zip(word_dict, v)) for v in product(*word_dict.values())]
                for wp in word_product:
                    new_sent = " ".join(list(wp.values()))
                    new_label = ['positive', 'neutral']
                    results.append((f'new_{sent[0]}_{res_idx}', new_sent, new_label, None))
                    res_idx += 1
                # end for
            # end for
            random.shuffle(results)
            return results
        elif negation_pattern=='positive':
            # negated of positive with neutral content in the middle
            # first, search neutral sentences
            positive_sents = [s for s in sents if s[2]=='positive']
            random.shuffle(positive_sents)
            neutral_sents = [s[1] for s in sents if s[2]=='neutral']
            random.shuffle(neutral_sents)
            neutral_selected = [f"given {s}," for s in neutral_sents[:3]]
            
            word_dict = DISAGREEMENT_PHRASE
            word_dict['middlefix'] = neutral_selected
            res_idx = 0
            for sent in positive_sents:
                word_dict['sent'] = [sent[1]]
                label = sent[2]
                word_product = [dict(zip(word_dict, v)) for v in product(*word_dict.values())]
                for wp in word_product:
                    new_sent = " ".join(list(wp.values()))
                    new_label = 'negative'
                    results.append((f'new_{sent[0]}_{res_idx}', new_sent, new_label, None))
                    res_idx += 1
                # end for
            # end for
            random.shuffle(results)
            return results
        # end if

        # negated neutral should still be neutral &
        # negated negative should be positive or neutral
        # search sents by tag of pos organization
        prefix_pat, postfix_pas = '',''
        if negation_pattern.startswith('^'):
            negation_pattern = negation_pattern[1:]
            prefix_pat = '^'
        # end if

        res_idx = 0
        for pat in self._get_negationpattern_to_wordproduct(negation_pattern, WORD2POS_MAP):
            _pat = prefix_pat+pat
            for sent in sents:
                if re.search(_pat, sent[1]):
                    new_sent = re.sub(_pat, f"{pat} not", sent[1])
                    label = sent[2]
                    new_label = None
                    if label=='negative':
                        new_label = ['positive', 'neutral']
                    elif label=='neutral':
                        new_label = 'neutral'
                    # end if
                    results.append((f'new_{sent[0]}_{res_idx}', new_sent, new_label, None))
                    res_idx += 1
                # end if
            # end for
        # end for
        random.shuffle(results)
        return results

    def srl(self, sents, na_param):
        positive_sents = [s[1] for s in sents if s[2]=='positive']
        random.shuffle(positive_sents)
        negative_sents = [s[1] for s in sents if s[2]=='negative']
        random.shuffle(negative_sents)
        
        word_dict = SRL_PHASE_TEMPLATE
        res_idx = 0
        results = list()
        for sent in sents:
            label = sent[2]
            if label=='positive':
                word_dict['sent1'] = [f"\"{s}\"," for s in negative_sents[:3]]
            elif label=='negative':
                word_dict['sent1'] = [f"\"{s}\"," for s in positive_sents[:3]]
            # end if
            word_dict['sent2'] = [f"\"{sent[1]}\""]
            
            word_product = [dict(zip(word_dict, v)) for v in product(*word_dict.values())]
            for wp in word_product:
                new_sent = " ".join(list(wp.values()))
                results.append((f'new_{sent[0]}_{res_idx}', new_sent, label, None))
                res_idx += 1
            # end for
        # end for
        random.shuffle(results)
        return results

    def questionize(self, sents, answer):
        word_dict = QUESTIONIZE_PHRASE_TEMPLATE
        res_idx = 0
        results = list()
        for sent in sents:
            word_dict['sent'] = [f"\"{sent[1]}\"?"]
            word_dict['answer'] = [answer]
            label = sent[2]
            if label=='positive' and answer=='yes':
                new_label = 'positive'
            elif label=='positive' and answer=='no':
                new_label = 'negative'
            elif label=='negative' and answer=='yes':
                new_label = 'negative'
            elif label=='negative' and answer=='no':
                new_label = ['positive', 'neutral']
            # end if
            word_product = [dict(zip(word_dict, v)) for v in product(*word_dict.values())]
            for wp in word_product:
                new_sent = " ".join(list(wp.values()))
                results.append((f'new_{sent[0]}_{res_idx}', new_sent, new_label, None))
                res_idx += 1
            # end for
        # end for
        random.shuffle(results)
        return results


class TransformOperatorForFairness:

    def __init__(
        self,
        requirements,
        editor=None
    ):    
        self.editor = editor # checklist.editor.Editor()
        self.capability = requirements['capability']
        self.description = requirements['description']
        # self.search_dataset = search_dataset
        self.transform_reqs = requirements['transform']
        # self.inv_replace_target_words = None
        # self.inv_replace_forbidden_words = None
        self.transform_func = self.transform_reqs.split()[0]
        self.transform_props = None
        self.identity_groups: Dict = Hatecheck.get_placeholder_values()

        if len(self.transform_reqs.split())>1:
            self.transform_props = self.transform_reqs.split()[1]
        # end if
    
    def transform(self, sents):
        transform_func_map = {
            'replace': self.replace
        }
        new_sents = transform_func_map[self.transform_func](sents, self.transform_props)
        return new_sents

    def replace_pronouns_to_identity_groups(
        self,
        sents
    ) -> List:
        pronouns_dict = {
            'y': ['you', 'your', 'yours'],
            'h': ['he', 'his', 'him'],
            's': ['she', 'her', 'hers'],
            't': ['they', 'their', 'them']
        }
        pronouns_with_apostrophes = [
            pronouns_dict[key][1]
            for key in pronouns_dict.keys()
        ]
        results = list()

        for s in sents:
            # first find how many pronouns is used in the sentence
            pronouns_used = list()
            tokens = Utils.tokenize(s[1])
            label = s[2]
            res_idx = 0
            for t_i, t in enumerate(tokens):
                if t.lower() in pronouns_dict['h']:
                    pronouns_used.append((t_i, 'h'))
                elif t.lower() in pronouns_dict['s']:
                    pronouns_used.append((t_i, 's'))
                elif t.lower() in pronouns_dict['t']:
                    pronouns_used.append((t_i, 't'))
                elif t.lower() in pronouns_dict['y']:
                    pronouns_used.append((t_i, 'y'))
                # end if
            # end for

            # generate map for the pronouns used and identity words
            unique_pronouns = set([pr[1] for pr in pronouns_used])
            num_pronouns_used = len(unique_pronouns)
            if num_pronouns_used>0:
                pronouns_to_identity_map = dict()
                for pr in unique_pronouns:
                    if pr=='t':
                        pronouns_to_identity_map[pr] = self.identity_groups['IDENTITY_P']
                    else:
                        pronouns_to_identity_map[pr] = self.identity_groups['IDENTITY_S'] + self.identity_groups['IDENTITY_A']
                    # end if
                # end for

                if num_pronouns_used>1:
                    for num_repl in range(1, num_pronouns_used+1):
                        target_pronouns_generator = permutations(
                            list(pronouns_to_identity_map.keys()), 
                            num_repl
                        )
                        for target_pronouns in target_pronouns_generator:
                            _pronouns_used = [
                                pr for pr in pronouns_used
                                if pr[1] in target_pronouns
                            ]
                            _pronouns_to_identity_map = {
                                key: pronouns_to_identity_map[key]
                                for key in pronouns_to_identity_map.keys()
                                if key in target_pronouns
                            }

                            word_product: List[Dict] = [
                                dict(zip(_pronouns_to_identity_map, v))
                                for v in product(*_pronouns_to_identity_map.values())
                            ]
                            for wp in word_product:
                                _tokens = tokens.copy()
                                for _pr_i, _pr in _pronouns_used:
                                    if _tokens[_pr_i] in pronouns_with_apostrophes:
                                        _tokens[_pr_i] = f"{wp[_pr]}'s"
                                    else:
                                        _tokens[_pr_i] = wp[_pr]
                                    # end for
                                # end for
                                new_sent = Utils.detokenize(_tokens)
                                results.append((f'new_{s[0]}_{res_idx}', new_sent, label, None))
                                res_idx += 1
                            # end for
                        # end for
                    # end for
                else:
                    pr_i, pr = pronouns_used[0][0], pronouns_used[0][1]
                    for w in pronouns_to_identity_map[pr]:
                        _tokens = tokens.copy()
                        if _tokens[pr_i] in pronouns_with_apostrophes:
                            _tokens[pr_i] = f"{w}'s"
                        else:
                            _tokens[pr_i] = w
                        # end if
                        new_sent = Utils.detokenize(_tokens)
                        results.append((f'new_{s[0]}_{res_idx}', new_sent, label, None))
                        res_idx += 1
                    # end for
                # end if
            # end if
        # end for
        return results

    def replace(
        self, 
        sents,
        replace_prop
    ):
        results = list()
        if replace_prop=='pronouns_with_<hatecheck_identity>':
            results = self.replace_pronouns_to_identity_groups(sents)
        # end if
        return results
