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
from nltk import Nonterminal

from ...utils.Macros import Macros
from ...utils.Utils import Utils

class RefPCFG:

    def __init__(self, corpus_name='treebank'):
        self.corpus_name = corpus_name
        self.pcfg_dir = Macros.result_dir / "ref_corpus" 
        self.pcfg_file = self.pcfg_dir / f"ref_pcfg_{corpus_name}.json"
        # self.pcfg = Dict[rule_string, Dict[lhs, rhs, prob]]
        self.grammar, self.pcfg = None, None
        if self.corpus_name!='treebank':
            rules = self.get_rules()
            self.grammar, self.pcfg = self.get_pcfg(rule_dict=rules)
        else:
            self.grammar, self.pcfg = self.get_pcfg()
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

    def get_treebank_pcfg(self):
        rule_dict = dict()
        productions = list()
        for s in treebank.parsed_sents():
            productions += s.productions()
        # end for
        S = Nonterminal('S')
        grammar = nltk.induce_pcfg(S, productions)
        if os.path.exists(str(self.pcfg_file)):
            rule_dict = Utils.read_json(self.pcfg_file)
            return grammar, rule_dict
        # end if
        for prod in grammar.productions():
            lhs_key = str(prod._lhs)
            if lhs_key not in rule_dict:
                rule_dict[lhs_key] = list()
            # end if
            rule_dict[lhs_key].append({
                'rhs': [str(r) for r in prod._rhs],
                'prob': prod.prob()
            })
        # end for

        # relaxing rhs
        self.pcfg_dir.mkdir(parents=True, exist_ok=True)
        Utils.write_json(rule_dict, self.pcfg_file)
        return grammar, rule_dict

    def get_rules(self):
        if self.corpus_name=='treebank':
            rule_dict = dict()
            for tree in treebank.parsed_sents():
                rule_dict = cls.get_treebank_rules(tree, rule_dict)
            # end for
            return rule_dict
        # end if

    def get_pcfg(self, rule_dict=None):
        if self.corpus_name=='treebank':
            grammar, rule_dict = self.get_treebank_pcfg()
            return grammar, rule_dict
        # end if
