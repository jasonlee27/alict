# This script is to perturb cfg of sentences
# given a seed input and reference CFGs.

from typing import *

import re, os
import nltk
import copy
import random
import numpy

from pathlib import Path
from nltk.grammar import Nonterminal
from nltk.parse import generate
from nltk import CFG

from ...utils.Macros import Macros
from ...utils.Utils import Utils
from .CFG import BeneparCFG # , TreebankCFG
from .RefPCFG import RefPCFG


random.seed(Macros.SEED)
COMP_LENGTH = 3
NUM_TOPK = 5
PHRASE_LEVEL_WORD_MAX_LEN = 5
PHRASE_LEVEL_POS = [
    'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ',
    'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN',
    'PRT', 'QP', 'RRC', 'UCP', 'VP',
    'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X'
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
                 tree_seed):
        self.cfg_diff = self.get_cfg_diff(
            cfg_seed=cfg_seed,
            tree_seed=tree_seed,
            pcfg_ref=pcfg_ref,
            comp_length = COMP_LENGTH
        )
        self.grammar_ref, self.rule_dict_ref = pcfg_ref.get_pcfg()
    
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

    def get_target_rule_parents(self,
                                pcfg_ref,
                                seed_tree,
                                target_rule,
                                target_words,
                                sent_prob_wo_target,
                                parent_rule_list):
        if seed_tree is None:
            return parent_rule_list, sent_prob_wo_target
        # end if
        
        # traverse seed_tree and search target rule
        print(seed_tree._.labels, target_rule, target_words)
        rule_lhs = seed_tree._.labels[0]
        rule_rhs = tuple([ch._.labels[0] if any(ch._.labels) else str(ch) for ch in seed_tree._.children])
        rule_key = f"{rule_lhs} -> {rule_rhs}"
        if rule_key==target_rule and str(seed_tree)==target_words:
            # when we find the target rule in seed tree,
            # reverse tree to get parents of target rule
            if not any(parent_rule_list):
                while True:
                    parent = seed_tree._.parent
                    parent_lhs = parent._.labels[0]
                    parent_rhs = tuple([ch._.labels[0] if any(ch._.labels) else str(ch) for ch in parent._.children])
                    parent_key = f"{parent_lhs} -> {parent_rhs}"
                    parent_rule_list.append(parent_key)
                    if parent._.labels[0]=='S':
                        parent_rule_list.reverse()
                        break
                    # end if
                # end while
            else:
                sent_prob_wo_target = sent_prob_wo_target*self.get_rule_prob(pcfg_ref, rule_lhs, rule_rhs)
            # end if
        else:
            sent_prob_wo_target = sent_prob_wo_target*self.get_rule_prob(pcfg_ref, rule_lhs, rule_rhs)
            for ch in seed_tree._.children:
                parent_rule_list, sent_prob_wo_target = self.get_target_rule_parents(pcfg_ref,ch, target_rule, str(ch), sent_prob_wo_target, parent_rule_list)
            # end for
        # end if
        return parent_rule_list, sent_prob_wo_target

    def get_rule_prob(self, pcfg_ref, lhs:str, rhs:str):
        for r in pcfg_ref[lhs]:
            if rhs==str(tuple(r['rhs'])):
                return r['prob']
            # end if
        # end for
        return 0.
    
    def get_parent_rules_prob(self, parent_rules):
        prob = 1.
        for r in parent_rules:
            lhs, rhs = r.split(' -> ')
            prob = prob*self.get_rule_prob(lhs, rhs)
        # end for
        return prob

    def get_token_pos(self, token):
        for key in self.rule_dict_ref.keys():
            if key.split('-')[0] in WORD_LEVEL_POS:
                for v in self.rule_dict_ref[key]:
                    if token in v['rhs']:
                        return key
                    # end if
                # end for
            # end if
        # end for
        return None
    
    def get_sent_pos(self, sent):
        return [self.get_token_pos(t) for t in sent]

    def phrase_to_word_pos(self, pos_list):
        contained = [(p_i,p) for p_i, p in enumerate(pos_list) if p.split('-')[0] in PHRASE_LEVEL_POS]
        if any(contained):
            for c_i, c in contained:
                nonterminal = Nonterminal(c)
                for s in generate.generate(self.grammar_ref, start=nonterminal, n=20):
                    sent_pos = self.get_sent_pos(s)
                    if all(sent_pos) and len(sent_pos)<PHRASE_LEVEL_WORD_MAX_LEN:
                        pos_list = pos_list[:c_i]+sent_pos+pos_list[c_i+1:]
                    # end if
            return contained
        # end if
        return pop_list

    def get_exp_syntax_probs(self, pcfg_ref, seed_cfg, lhs_seed, rhs_seed, target_words, rhs_to_candid):
        # rule_candid: Dict.
        # key is the rule_from, and values are the list of rule_to to be replaced with.
        prob_list = list()
        rule_key = f"{lhs_seed} -> {tuple(rhs_seed)}"
        parent_rules, sent_prob_wo_target = self.get_target_rule_parents(
            pcfg_ref, seed_cfg, rule_key, target_words, 1., list()
        )

        parents_prob = self.get_parent_rules_prob(parent_rules)
        for rhs_to, rhs_to_prob in rule_to_candid:
            rhs_to = self.phrase_to_word_pos(rhs_to)
            prob_list.append({
                'parents': parent_rules,
                'rule': rule_to,
                'prob': rhs_to_prob,
                'sent_prob_wo_target': sent_prob_wo_target
            })
        # end for
        prob_list = sorted(prob_list, key=lambda x: x['prob_w_parents'])
        prob_list.reverse()
        return prob_list[:NUM_TOPK]
    
    def get_cfg_diff(self,
                     cfg_seed,
                     pcfg_ref,
                     tree_seed,
                     comp_length=COMP_LENGTH):
        cfg_diff = dict()
        for seed_lhs, seed_rhs in cfg_seed.items():
            try:
                for _sr in seed_rhs:
                    sr = _sr['pos']
                    sr = [sr] if type(sr)==str else list(sr)
                    rule_from_ref = list()
                    for rhs_dict in pcfg_ref[seed_lhs]:
                        rr = rhs_dict['rhs']
                        rr_prob = rhs_dict['prob']
                        if self.check_list_inclusion(sr, rr) and \
                           (rr, rr_prob) not in rule_from_ref and \
                           len(sr)<len(rr):
                            # len(rr)<comp_length+len(sr)
                            rule_from_ref.append((rr, rr_prob))
                        # end if
                    # end for
                    
                    # rhs_syntax_probs = list()
                    if any(rule_from_ref):
                        # Get syntax prob
                        rhs_syntax_probs = self.get_exp_syntax_probs(
                            pcfg_ref, tree_seed, seed_lhs, sr, _sr['word'], rule_from_ref
                        )
                        
                        # Get top-k prob elements
                        if seed_lhs not in cfg_diff.keys():
                            cfg_diff[seed_lhs] = {
                                sr: ([
                                    (r['rule'], r['prob'], r['sent_prob_wo_target'])
                                    for r in rhs_syntax_probs
                                ], _sr['word'])
                            }
                        elif sr not in cfg_diff[seed_lhs].keys():
                            cfg_diff[seed_lhs][sr] = ([
                                (r['rule'], r['prob'], r['sent_prob_wo_target'])
                                for r in rhs_syntax_probs
                            ], _sr['word'])
                        # end if
                    # end if
                # end for
            except KeyError:
                continue
            # end try
        # end for
        return cfg_diff
