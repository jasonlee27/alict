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

    def _get_treebank_pcfg(self, pos_of_sent_start, sent_start_prob, productions, rule_dict):
        S = Nonterminal(pos_of_sent_start)
        grammar = nltk.induce_pcfg(S, productions)
        if os.path.exists(str(self.pcfg_file)):
            rule_dict = Utils.read_json(self.pcfg_file)
            return grammar, rule_dict
        # end if

        # first count the lhs function tag 
        # for applying lhs tag prob distribution
        lhs_func_cnt_dict = dict()
        for prod in grammar.productions():
            lhs_key = str(prod._lhs).split('-')
            lhs_func = "<empty>"
            if len(lhs_key)>1:
                lhs_func = '-'.join(lhs_key[1:])
            # end if
            lhs_key = lhs_key[0]
            if lhs_key not in lhs_func_cnt_dict.keys():
                lhs_func_cnt_dict[lhs_key] = { lhs_func: 1 }
            elif lhs_key in lhs_func_cnt_dict.keys() and \
                 lhs_func not in lhs_func_cnt_dict[lhs_key].keys():
                lhs_func_cnt_dict[lhs_key][lhs_func] = 1
            else:
                lhs_func_cnt_dict[lhs_key][lhs_func] += 1
            # end if
        # end for

        # second get the prob of production rule
        for prod in grammar.productions():
            lhs_key = str(prod._lhs).split('-')
            lhs_func = "<empty>"
            if len(lhs_key)>1:
                lhs_func = '-'.join(lhs_key[1:])
            # end if
            lhs_key = lhs_key[0]

            # compute lhs_key prob
            lhs_tot_cnt = sum([
                lhs_func_cnt_dict[lhs_key][key]
                for key in lhs_func_cnt_dict[lhs_key].keys()
            ])
            lhs_key_cnt = lhs_func_cnt_dict[lhs_key][lhs_func]
            lhs_key_prob = lhs_key_cnt*1./lhs_tot_cnt

            if lhs_key not in rule_dict.keys():
                rule_dict[lhs_key] = [{
                    'rhs': [str(r).split('-')[0] for r in prod._rhs],
                    'prob': sent_start_prob*lhs_key_prob*prod.prob()
                }]
            else:
                rhs_query = [str(r).split('-')[0] for r in prod._rhs]
                is_query_found = False
                for rhs_i in range(len(rule_dict[lhs_key])):
                    if rhs_query==rule_dict[lhs_key][rhs_i]['rhs']:
                        is_query_found = True
                        rule_dict[lhs_key][rhs_i]['prob'] += sent_start_prob*lhs_key_prob*prod.prob()
                        break
                    # end if
                # end for
                if not is_query_found:
                    rule_dict[lhs_key].append({
                        'rhs': rhs_query,
                        'prob': sent_start_prob*lhs_key_prob*prod.prob()
                    })
                # end if
            # end if
        # end for
        return grammar, rule_dict

    def get_treebank_pcfg(self):
        raw_rule_dict = dict()
        rule_dict = dict()
        grammars = dict()
        productions = list()
        pos_of_sent_start = list()
        for s in treebank.parsed_sents():
            productions += s.productions()
            pos_of_sent_start.append(s.label())
        # end for

        rule_dict['<SOS>'] = list()
        for s in set(pos_of_sent_start):
            start_prob = pos_of_sent_start.count(s)*1./len(pos_of_sent_start)
            rule_dict['<SOS>'].append({
                'rhs': [s],
                'prob': start_prob
            })
            grammar, rule_dict = self._get_treebank_pcfg(s, start_prob, productions, rule_dict)
            grammars[s] = grammar
        # end if
                
        self.pcfg_dir.mkdir(parents=True, exist_ok=True)
        Utils.write_json(rule_dict, self.pcfg_file)
        return grammars, rule_dict

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
