# This script is to perturb cfg of sentences
# given a seed input and reference CFGs.

from typing import *

import re, os
import sys
import nltk
import copy
import spacy
import random

from pathlib import Path
from nltk.grammar import Nonterminal
from nltk.parse import generate
from nltk import CFG

from ...utils.Macros import Macros
from ...utils.Utils import Utils
from .CFG import BeneparCFG # , TreebankCFG
# from .RefPCFG import RefPCFG


COMP_LENGTH = 3
NUM_TOPK = 5
PHRASE_LEVEL_WORD_MAX_LEN = 5

CLAUSE_LEVEL_POS = [
    'S', 'SBAR', 'SBARQ', 'SINV', 'SQ'
]

PHRASE_LEVEL_POS = [
    'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ',
    'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN',
    'PRT', 'QP', 'RRC', 'UCP', 'VP',
    'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X',
]

WORD_LEVEL_POS = [
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ',
    'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS',
    'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',
    'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO',
    'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
    'VBZ', 'WDT', 'WP', 'WP$', 'WRB'
]

class CFGDiff:

    def __init__(self, 
                 pcfg_ref: dict, 
                 cfg_seed: dict,
                 tree_seed
    ):
        self.cfg_diff = self.get_cfg_diff(
            cfg_seed=cfg_seed,
            tree_seed=tree_seed,
            pcfg_ref=pcfg_ref,
            comp_length=COMP_LENGTH,
        )

    
    def check_list_inclusion(self, a_list, b_list):
        temp = -1
        a_is = list()
        for a in a_list:
            try:
                a_i = b_list.index(a)
                if temp<a_i:
                    temp = a_i
                    a_is.append(a)
                else:
                    a_is.append(None)
                # end if
            except ValueError:
                a_is.append(None)
            # end try
        # end for
        if all(a_is):
            return True
        # end if
        return False

    def get_rule_key(self, seed_tree):
        lhs = None
        rhs = None
        words = None
        if any(seed_tree._.labels):
            lhs = seed_tree._.labels[0]
            rhs = list()
            words = list()
            if any(list(seed_tree._.children)):
                for ch in seed_tree._.children:
                    if any(ch._.labels):
                        rhs.append(ch._.labels[0])
                    else:
                        search = re.search(r'\((\S+)\s(\S+)\)', ch._.parse_string)
                        if search:
                            rhs.append(search.group(1))
                        # end if
                    # end if
                    words.append(str(ch))
                # end for
                return lhs, tuple(rhs), tuple(words)
            else:
                # search = re.search(f"\(([^\s\(]+)\s([^\s\)]+)\)", seed_tree._.parse_string)
                search = re.search(f"\(([^s\(]+)\s([^\(\)]+)\)", seed_tree._.parse_string)
                rhs.append(search.group(1))
                return lhs, tuple(rhs), str(seed_tree)
            # end if            
        else:
            search = re.search(r'\((\S+)\s(\S+)\)', seed_tree._.parse_string)
            # return f"{search.group(1)} -> {tuple([search.group(2)])}"
            return search.group(1), tuple([search.group(2)]), str(seed_tree)
        # end if

    def get_target_rule_parents(self,
                                pcfg_ref,
                                seed_tree,
                                target_lhs,
                                target_rhs,
                                target_words,
                                sent_prob_wo_target,
                                parent_rule_list):
        if seed_tree is None:
            return parent_rule_list, sent_prob_wo_target
        # end if
        # traverse seed_tree and search target rule
        rule_lhs, rule_rhs, rule_words = self.get_rule_key(seed_tree)
        # rule_key = f"{rule_lhs} -> {rule_rhs}"
        if rule_lhs==target_lhs and rule_rhs==target_rhs and rule_words==target_words:
            # when we find the target rule in seed tree,
            # reverse tree to get parents of target rule
            if not any(parent_rule_list):
                parent = seed_tree._.parent
                parent_lhs = None
                if parent is None:
                    return parent_rule_list, sent_prob_wo_target
                # end if
                while parent is not None:
                    parent_lhs, parent_rhs, _ = self.get_rule_key(parent)
                    parent_key = f"{parent_lhs} -> {parent_rhs}"
                    parent_rule_list.append(parent_key)
                    parent = parent._.parent
                    sent_prob_wo_target = sent_prob_wo_target/self.get_rule_prob(pcfg_ref, parent_lhs, parent_rhs)
                # end while
                parent_rule_list.append(f"<SOS> -> {parent_lhs}")
                parent_rule_list.reverse()
            # end if
            # sent_prob_wo_target = sent_prob_wo_target*self.get_rule_prob(pcfg_ref, rule_lhs, rule_rhs)
        else:
            sent_prob_wo_target = sent_prob_wo_target*self.get_rule_prob(pcfg_ref, rule_lhs, rule_rhs)
            for ch in seed_tree._.children:
                parent_rule_list, sent_prob_wo_target = self.get_target_rule_parents(
                    pcfg_ref, ch,
                    target_lhs, target_rhs, target_words,
                    sent_prob_wo_target, parent_rule_list
                )
            # end for
        # end if
        return parent_rule_list, sent_prob_wo_target

    def get_rule_prob(self, pcfg_ref, lhs:str, rhs:tuple):
        for r in pcfg_ref[lhs]:
            if rhs==tuple(r['rhs']):
                return r['prob']
            # end if
        # end for
        return 1.
    
    def get_parent_rules_prob(self, pcfg_ref, parent_rules):
        prob = 1.
        for r in parent_rules:
            lhs, rhs = r.split(' -> ')
            if type(rhs)==str: rhs = tuple([rhs])
            prob = prob*self.get_rule_prob(pcfg_ref, lhs, rhs)
        # end for
        return prob

    def get_token_pos(self, nlp, token):
        doc = nlp(token)
        # for key in self.rule_dict_ref.keys():
        #     if key.split('-')[0] in WORD_LEVEL_POS:
        #         for v in self.rule_dict_ref[key]:
        #             if token in v['rhs']:
        #                 return key
        #             # end if
        #         # end for
        #     # end if
        # # end for
        return doc[0].tag_
    
    def get_sent_pos(self, nlp, sent):
        return [self.get_token_pos(nlp, t) for t in sent]

    def generate_phrase(self, grammar, start_symbol):
        nonterminal = Nonterminal(start_symbol)
        sents = list()
        sys.setrecursionlimit(100)
        try:
            for s in generate.generate(grammar, start=nonterminal, depth=5):
                sents.append(s)
            # end for
            random.shuffle(sents)
            return sents[0]
        except Exception:
            if any(sents):
                random.shuffle(sents)
                return sents[0]
            else:
                return sents
            # end if
        # end try
        
    def phrase_to_word_pos(self, pcfg_ref, pos_from, pos_list):
        contained = [
            (p_i,p)
            for p_i, p in enumerate(pos_list)
            if (p.split('-')[0] in PHRASE_LEVEL_POS+CLAUSE_LEVEL_POS) and
            (p.split('-')[0] not in pos_from)
        ]
        # nlp = spacy.load('en_core_web_md')
        if any(contained):
            return None
            # for c_i, c in contained:
            #     sent = self.generate_phrase(pcfg_ref.grammar, c)
            #     if any(sent):
            #         sent_pos = self.get_sent_pos(nlp, sent)
            #         if all(sent_pos) and len(sent_pos)<PHRASE_LEVEL_WORD_MAX_LEN:
            #             pos_list = pos_list[:c_i]+sent_pos+pos_list[c_i+1:]
            #         # end if
            #     else:
            #         return None
            #     # end if
            # # end for
        # end if
        return pos_list

    def check_rhs_availability(self, pos_from, pos_to):
        # for now, we only take pos expansion
        recursion_pos_contained = [
            (p_i,p)
            for p_i, p in enumerate(pos_to)
            if (p.split('-')[0] in PHRASE_LEVEL_POS+CLAUSE_LEVEL_POS) and
            (p.split('-')[0] not in pos_from)
        ]
        non_maskable_pos_contained = [
            (p_i,p)
            for p_i, p in enumerate(pos_to)
            if (p.split('-')[0] not in PHRASE_LEVEL_POS+CLAUSE_LEVEL_POS+WORD_LEVEL_POS) and
            (p.split('-')[0] not in pos_from)
        ]
        if any(recursion_pos_contained) or any(non_maskable_pos_contained):
            return False
        # end if
        return True

    def get_exp_syntax_probs(self, pcfg_ref, seed_cfg, lhs_seed, rhs_seed, target_words, rhs_to_candid):
        # rule_candid: Dict.
        # key is the rule_from, and values are the list of rule_to to be replaced with.
        prob_list = list()

        # compute prob of start of sentence
        sent_prob_wo_target = self.get_rule_prob(
            pcfg_ref.pcfg, '<SOS>', tuple([seed_cfg._.labels[0]])
        )
        
        parent_rules, sent_prob_wo_target = self.get_target_rule_parents(
            pcfg_ref.pcfg, seed_cfg,
            lhs_seed, tuple(rhs_seed), target_words,
            sent_prob_wo_target, list()
        )
       
        parents_prob = self.get_parent_rules_prob(pcfg_ref.pcfg, parent_rules)
        for rhs_to, rhs_to_prob in rhs_to_candid:
            if self.check_rhs_availability(rhs_seed, rhs_to):
                rhs_to = self.phrase_to_word_pos(pcfg_ref, rhs_seed, rhs_to)
                prob_list.append({
                    'parents': parent_rules,
                    'rule': rhs_to,
                    'prob': rhs_to_prob,
                    'prob_w_parents': parents_prob*rhs_to_prob,
                    'sent_prob_wo_target': sent_prob_wo_target
                })
            # end if
        # end for
        if any(prob_list):
            prob_list = sorted(prob_list, key=lambda x: x['prob_w_parents'], reverse=True)
            prob_list = prob_list[:NUM_TOPK]
        # end if
        return prob_list
    
    def get_cfg_diff(self,
                     cfg_seed,
                     pcfg_ref,
                     tree_seed,
                     comp_length):
        cfg_diff = dict()
        for seed_lhs, seed_rhs in cfg_seed.items():
            try:
                for _sr in seed_rhs:
                    sr = _sr['pos']
                    sr = [sr] if type(sr)==str else list(sr)
                    rule_from_ref = list()
                    for rhs_dict in pcfg_ref.pcfg[seed_lhs]:
                        rr = rhs_dict['rhs']
                        rr_prob = rhs_dict['prob']
                        if self.check_list_inclusion(sr, rr) and \
                           (rr, rr_prob) not in rule_from_ref and \
                           len(sr)<len(rr):
                            # len(rr)<comp_length+len(sr)
                            rule_from_ref.append((rr, rr_prob))
                        # end if
                    # end for

                    if any(rule_from_ref):
                        # Get syntax prob
                        rhs_syntax_probs = self.get_exp_syntax_probs(
                            pcfg_ref, tree_seed, seed_lhs, sr, _sr['word'], rule_from_ref
                        )
                        if any(rhs_syntax_probs):
                            # Get top-k prob elements
                            if seed_lhs not in cfg_diff.keys():
                                cfg_diff[seed_lhs] = {
                                    str(sr): ([
                                        (r['rule'], r['prob'], r['sent_prob_wo_target'])
                                        for r in rhs_syntax_probs
                                    ], _sr['word'])
                                }
                            elif str(sr) not in cfg_diff[seed_lhs].keys():
                                cfg_diff[seed_lhs][str(sr)] = ([
                                    (r['rule'], r['prob'], r['sent_prob_wo_target'])
                                    for r in rhs_syntax_probs
                                ], _sr['word'])
                            # end if
                        # end if
                    elif any(rule_from_ref):
                        # randomly select the syntax expansion suggestion
                        random.shuffle(rule_from_ref)
                        if seed_lhs not in cfg_diff.keys():
                            cfg_diff[seed_lhs] = {
                                str(sr): ([
                                    (rhs_to, None, None)
                                    for rhs_to, rhs_to_prob in rule_from_ref
                                ], _sr['word'])
                            }
                        elif str(sr) not in cfg_diff[seed_lhs].keys():
                            cfg_diff[seed_lhs][str(sr)] = ([
                                (rhs_to, None, None)
                                for rhs_to, rhs_to_prob in rule_from_ref
                            ], _sr['word'])
                        # end if
                    # end if
                # end for
            except KeyError:
                continue
            # end try
        # end for
        return cfg_diff
