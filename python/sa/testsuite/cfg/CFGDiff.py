# This script is to perturb cfg of sentences
# given a seed input and reference CFGs.

from typing import *

import re, os
import nltk
import copy
import random
import numpy

from pathlib import Path
from nltk.parse.generate import generate, demo_grammar
from nltk import CFG

from ...utils.Macros import Macros
from ...utils.Utils import Utils
from .CFG import BeneparCFG # , TreebankCFG
from RefPCFG import RefPCFG


random.seed(Macros.SEED)
COMP_LENGTH = 3
NUM_TOPK = 3

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

    def get_target_rule_parents(self, seed_tree, target_rule, parent_rule_list):
        # traverse seed_tree and search target rule
        rule_lhs = seed_tree._.labels[0]
        rule_rhs = tuple([ch._.labels[0] if any(ch._.labels) else str(ch) for ch in seed_tree._.children])
        rule_key = f"{rule_lhs} -> {rule_rhs}"
        if rule_key==target_rule:
            # when we find the target rule in seed tree,
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

    def get_rule_prob(self, lhs:str, rhs:str):
        for r in self.pcfg[lhs]:
            if rhs==str(tuple(_r['rhs'])):
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

    def get_exp_syntax_probs(self, seed_cfg, lhs_seed, rhs_seed, rhs_to_candid):
        # rule_candid: Dict.
        # key is the rule_from, and values are the list of rule_to to be replaced with.
        prob_list = list()
        rule_key = f"{lhs_seed} -> {rhs_seed}"
        parent_rules = self.get_target_rule_parents(seed_cfg, rule_key, list())
        parents_prob = self.get_parent_rules_prob(parent_rules)
        for rhs_to, rhs_to_prob in rule_to_candid:
            prob_list.append({
                'parents': parent_rules,
                'rule': rule_to,
                'prob': parents_prob*rhs_to_prob
            })
        # end for
        prob_list = sorted(prob_list, key=lambda x: x['prob'])
        prob_list.reverse()
        return prob_list
    
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
                        if self.check_list_inclusion(sr, rr) and len(sr)<len(rr):
                            # len(rr)<comp_length+len(sr)
                            if (rhs_dict['rhs'], rhs_dict['prob']) not in rule_from_ref[str(rr)]:
                                rule_from_ref.append((rhs_dict['rhs'], rhs_dict['prob']))
                            # end if
                        # end if
                    # end for

                    rhs_syntax_probs = list()
                    if any(rule_from_ref):
                        # Get syntax prob
                        rhs_syntax_probs = get_exp_syntax_probs(
                            tree_seed, seed_lhs, sr, rule_from_ref
                        )
                        
                        # Get top-k prob elements
                        rhs_syntax_probs = rhs_syntax_probs[:NUM_TOPK]
                    # end if
                    
                    if seed_lhs not in cfg_diff.keys() and any(rhs_syntax_probs):
                        cfg_diff[seed_lhs] = {
                            sr: ([
                                (r['rule'], r['prob'])
                                for r in rhs_syntax_probs
                            ], _sr["word"])
                        }
                    elif sr not in cfg_diff[seed_lhs].keys() and any(rule_from_ref):
                        cfg_diff[seed_lhs][sr] = ([
                            (r['rule'], r['prob'])
                            for r in rhs_syntax_probs
                        ], _sr["word"])
                    # end if
                # end for
            except KeyError:
                continue
            # end try
        # end for
        return cfg_diff
