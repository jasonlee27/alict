# This script generates nlp grammar rule set 
# given sentence input set.

import re, os
import sys
import json
import nltk
import benepar
import spacy

from pathlib import Path
from nltk.corpus import treebank
from nlpk import Nonterminal

from ...utils.Macros import Macros
from ...utils.Utils import Utils

class RefPCFG:

    def __init__(self, corpus_name='treebank'):
        self.corpus_name = corpus_name
        
        # self.pcfg = Dict[rule_string, Dict[lhs, rhs, prob]]
        self.pcfg = None
        if self.corpus_name!='treebank':
            rules = self.get_rules()
            self.pcfg = self.get_pcfg(rule_dict=rules)
        else:
            self.pcfg = self.get_pcfg()
        # end if

    def get_rules(self):
        if self.corpus_name=='treebank':
            rule_dict = dict()
            for tree in treebank.parsed_sents():
                rule_dict = cls.get_treebank_rules(tree, rule_dict)
            # end for
            return rule_dict
        # end if

    def get_treebank_rules(self, tree, rule_dict):
        if type(tree)==str:
            return rule_dict
        # end if
        rule = tree.productions()[0]
        corr_terminal_pos = [pos[1] for pos in tree.pos()]
        if str(rule) in rule_dict.keys():
            if (rule, corr_terminal_pos) in rule_dict[str(rule)]:
                rule_dict[str(rule)].append((rule,corr_terminal_pos))
            # end if
        else:
            rule_dict[str(rule)] = [(rule,corr_terminal_pos)]
        # end if
        for ch in tree:
            rule_dict = cls.get_treebank_rules(ch, rule_dict)
        # end for
        return rule_dict
        
    def get_pcfg(self, rule_dict=None):
        if self.corpus_name=='treebank':
            return get_treebank_pcfg()
        # end if

    def get_treebank_pcfg(self):
        rule_dict = dict()
        productions = list()
        for s in treebank.parsed_sents():
            productions += s.productions()
        # end for
        S = Noneterminal('S')
        grammar = nltk.induce_pcfg(S, productions)
        for prod in grammar.productions():
            rule_key = f"{prod._lhs} -> {prod._rhs}"
            rule_dict[rule_key] = {
                'lhs': str(prod._lhs)
                'rhs': [str(r) for r in prod._rhs]
                'prob': prod.prob()
            }
        # end for
        return rule_dict

    def get_target_rule_parents(self, seed_tree, target_rule, parent_rule_list):
        # traverse seed_tree and search target rule
        rule_lhs = seed_tree._.labels[0]
        rule_rhs = tuple([ch._.labels[0] if any(ch._.labels) else str(ch) for ch in seed_tree._.children])
        rule_key = f"{rule_lhs} -> {rule_rhs}"
        if rule_key==target_rule:
            # reverse tree to get parents of target rule
            parent_rule_list = list()
            while True:
                parent = seed_tree._.parent
                parent_lhs = parent._.labels[0]
                parent_rhs = tuple([ch._.labels[0] if any(ch._.labels) else str(ch) for ch in parent._.children])
                parent_key = f"{parent_lhs} -> {parent_rhs}"
                parent_rule_list.append(parent_key)
                if parent._.labels[0]=='S':
                    parent_rule_list.reverse()
                    return parent_rule_list
                # end if
            # end while
            return
        else:
            for ch in seed_tree._.children:
                self.get_target_rule_parents(ch, target_rule)
            # end for
        # end if
        return

    def get_parent_rules_prb(parent_rules):
        prob = 1.
        for r in parent_rules:
            prob = prob*self.pcfg[r]['prob']
        # end for
        return prob

    def get_exp_syntax_probs(self, seed_cfg, rule_candid):
        # rule_candid: Dict.
        # key is the rule_from, and values are the list of rule_to to be replaced with.
        prob_list = list()
        for rule_from in rule_candid.keys():
            parent_rules = self.get_target_rule_parents(seed_cfg, rule_from)
            parents_prob = self.get_parent_rules_prob(parent_rules)
            rule_to_candids = rule_candid[rule_from]
            for r in rule_to_candids:
                prob_list.append({
                    'parents': parent_rules,
                    'rule': r
                    'prob': parents_prob*self.pcfg[r]['prob']
                })
            # end for
        # end for
        return prob_list
    
