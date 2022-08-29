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

from ...utils.Macros import Macros
from ...utils.Utils import Utils

class BeneparCFG:
    benepar_parser_model = 'benepar_en3'
    
    @classmethod
    def load_parser(cls):
        nlp = spacy.load('en_core_web_md')
        if spacy.__version__.startswith('2'):
            nlp.add_pipe(benepar.BeneparComponent(cls.benepar_parser_model))
        else:
            nlp.add_pipe('benepar', config={'model': cls.benepar_parser_model})
        #end if
        return nlp

    @classmethod
    def get_tree(cls, parser, sent):
        doc = parser(sent.strip())
        tree = list(doc.sents)[0]
        return tree

    @classmethod
    def get_cfg_per_tree(cls, tree, rule_dict):
        left = tree._.labels
        
        if len(left)==0: # if terminal
            re_search = re.search(r'\((\:|\,|\$|\'\'|\`\`|\.|\-?[A-Z]+\-?|[A-Z]+\$)\s(.+)\)', tree._.parse_string)
            rlabel = re_search.group(1)
            rword = re_search.group(2)
            plabel = tree._.parent._.labels[0]

            comp = {
                "pos": f'{rword}',
                "word": tuple([str(tree)])
            }
            if rlabel not in rule_dict.keys():
                rule_dict[rlabel] = [comp]
            elif comp not in rule_dict[rlabel]:
                rule_dict[rlabel].append(comp)
            # end if
        else:
            llabel = left[0]
            if len(left)==2:
                llabel = left[0]
                rlabel = left[1]
                if llabel not in rule_dict.keys():
                    rule_dict[llabel] = [{
                        "pos": rlabel,
                        "word": tuple([str(tree)])
                    }]
                else:
                    rule_dict[llabel].append({
                        "pos": rlabel,
                        "word": tuple([str(tree)])
                    })
                # end if
                llabel = left[1]
            # end if
            
            if llabel not in rule_dict.keys():
                rule_dict[llabel] = list()
            # end if
                
            if len(list(tree._.children))>0:
                non_terminals = list()
                non_terminal_words = list()
                for r in tree._.children:
                    if len(list(r._.labels))>0:
                        non_terminals.append(r._.labels[0])
                        non_terminal_words.append(str(r))
                    else:
                        re_search = re.search(r'\((\:|\,|\$|\'\'|\`\`|\.|\-?[A-Z]+\-?|[A-Z]+\$)\s(.+)\)', r._.parse_string)
                        rlabel = re_search.group(1)
                        rword = re_search.group(2)
                        non_terminals.append(rlabel)
                        non_terminal_words.append(str(r))
                    # end if
                    rule_dict = cls.get_cfg_per_tree(r, rule_dict)
                # end for
                
                _rule_dict = {
                    "pos": tuple(non_terminals),
                    "word": tuple(non_terminal_words)
                }
                if len(non_terminals)>0 and (_rule_dict not in rule_dict[llabel]):
                    rule_dict[llabel].append(_rule_dict)
                # end if
            else:
                re_search = re.search(f'\({llabel}\s\((\:|\,|\$|\'\'|\`\`|\.|\-?[A-Z]+\-?|[A-Z]+\$)\s(.+)\)\)$', tree._.parse_string)
                rlabel = re_search.group(1)
                rword = re_search.group(2)
                if (rlabel) not in rule_dict[llabel]:
                    rule_dict[llabel].append({
                        "pos": (rlabel),
                        "word": tuple([str(tree)])
                    })
                # end if

                comp = {
                    "pos": f'{rword}',
                    "word": tuple([str(tree)])
                }
                if rlabel not in rule_dict.keys():
                    rule_dict[rlabel] = [comp]
                elif comp not in rule_dict[rlabel]:
                    rule_dict[rlabel].append(comp)
                # end if
            # end if
        # end if
        return rule_dict

    @classmethod
    def get_cfg_dict_per_sent(cls, parser, sent, rule_dict):
        tree = cls.get_tree(parser,sent)
        return {
            "tree": tree,
            "rule": cls.get_cfg_per_tree(tree, rule_dict)
        }
 
    @classmethod
    def get_seed_cfg(cls, seed_input):
        cfg_dict = None
        parser = cls.load_parser()
        cfg_dict = cls.get_cfg_dict_per_sent(parser,seed_input,{})
        return cfg_dict

    @classmethod
    def get_seed_cfgs(cls, tokenized_seed_inputs):
        cfg_dict = None
        parser = benepar.Parser(cls.benepar_parser_model)
        input_sents = list()
        cfg_dicts = list()
        for s in tokenized_seed_inputs:
            input_sents.append(benepar.InputSentence(s))
        # end for
        docs = parser.parse_sents(input_sents)
        for d in docs:
            tree = list(d.sents)[0]
            cfg = {
                "tree": tree,
                "rule": cls.get_cfg_per_tree(tree, {})
            }
            cfg_dicts.append(cfg)
        # end for
        return cfg_dict

    @classmethod
    def get_word_pos(cls, word):
        parser = cls.load_parser()
        tree = cls.get_tree(parser, word)
        return tree

    @classmethod
    def get_words_pos(cls, words):
        inp = ". ".join(words)+"."
        parser = cls.load_parser()
        doc = parser(inp)
        tree = list(doc.sents)
        return tree


# class TreebankCFG:

#     @classmethod
#     def get_treebank_rules(cls, pcfg=False):
#         if 'treebank' not in sys.modules:
#             from nltk.corpus import treebank
#         # end if
#         rule_dict = dict()
#         for tree in treebank.parsed_sents():
#             rule_dict = cls._get_treebank_rules(tree, rule_dict)
#         # end for
#         return rule_dict

#     @classmethod
#     def _get_treebank_rules(cls, tree, rule_dict):
#         if type(tree)==str:
#             return rule_dict
#         # end if
#         rule = tree.productions()[0]
#         corr_terminal_pos = [pos[1] for pos in tree.pos()]
#         if str(rule) in rule_dict.keys():
#             if (rule,corr_terminal_pos) in rule_dict[str(rule)]:
#                 rule_dict[str(rule)].append((rule,corr_terminal_pos))
#             # end if
#         else:
#             rule_dict[str(rule)] = [(rule,corr_terminal_pos)]
#         # end if
#         for ch in tree:
#             rule_dict = cls._get_treebank_rules(ch, rule_dict)
#         # end for
#         return rule_dict

#     @classmethod
#     def get_treebank_word_tags(cls):
#         if 'treebank' not in sys.modules:
#             from nltk.corpus import treebank
#         # end if
#         return list(set([pos for _,pos in treebank.tagged_words()]))

#     @classmethod
#     def convert_ruleset_to_dict(cls, ruleset: dict, prob=False):
#         cfg_dict = dict()
#         for r_key in ruleset.keys():
#             for r, trmnls in ruleset[r_key]:
#                 lhs = str(r.lhs())
#                 r_tuple = tuple([f'\'{r}\'' if type(r) is str else str(r) for r in r.rhs()])
#                 if lhs not in cfg_dict.keys():
#                     cfg_dict[lhs] = list()
#                 # end if

#                 if prob:
#                     cfg_dict[lhs].append((r_tuple,trmnls))
#                 else:
#                     if (r_tuple,trmnls) not in cfg_dict[lhs]:
#                         cfg_dict[lhs].append((r_tuple,trmnls))
#                     # end if
#                 # end if
#             # end for
#         # end for
#         if prob:
#             _cfg_dict = dict()
#             tot = sum([True for r_key in ruleset.keys() for _ in ruleset[r_key]])
#             for lhs in cfg_dict.keys():
#                 rhs_w_prob = list()
#                 for rhs, trmnls in cfg_dict[lhs]:
#                     num_occ = sum([True for _rhs, _trmnls in cfg_dict[lhs] if rhs==_rhs])
#                     rhs_w_prob.append((rhs, trmnls, num_occ*1./tot))
#                 # end for
#                 _cfg_dict[lhs] = rhs_w_prob
#             # end for
#             return _cfg_dict
#         # end if
#         return cfg_dict

#     # @classmethod
#     # def convert_cfg_dict_to_str(cls, cfg_dict):
#     #     cfg_str = ''
#     #     for left, rights in cfg_dict.items():
#     #         cfg_elem = f'{left} -> '
#     #         for r_i, right in enumerate(rights):
#     #             if type(right) is tuple:
#     #                 r_str = ' '.join([f'\'{r}\'' if type(r) is str else str(r) for r in right])
#     #                 cfg_elem += r_str
#     #             else:
#     #                 cfg_elem += f'\'{r_str}\''
#     #             # end if
#     #             if r_i+1<len(rights):
#     #                 cfg_elem += ' | '
#     #             # end if
#     #         # end for
#     #         cfg_str += f'{cfg_elem}\n'
#     #     # end for
#     #     return cfg_str

#     # @classmethod
#     # def write_cfg(cls, cfg_str, cfg_file):
#     #     with open(cfg_file, 'w') as f:
#     #         f.write(cfg_str)
#     #     # end with

#     # @classmethod
#     # def write_cfg(cls, cfg_dict: dict, cfg_file: Path, pretty_format=False):
#     #     with open(cfg_file, 'w') as f:
#     #         if pretty_format:
#     #             json.dump(cfg_dict, f, indent=4)
#     #         else:
#     #             json.dump(cfg_dict, f)
#     #         # end if
#     #     # end with

#     @classmethod
#     def get_cfgs(cls, cfg_file: Path, pretty_format=False):
#         if not os.path.exists(cfg_file):
#             rulesets: dict = cls.get_treebank_rules()
#             cfg_dict = cls.convert_ruleset_to_dict(rulesets)
#             # cfg_str = cls.convert_cfg_dict_to_str(cfg_dict)
#             Utils.write_json(cfg_dict, cfg_file, pretty_format=pretty_format)
#             return cfg_dict
#         # end if
#         return Utils.read_json(cfg_file)

#     @classmethod
#     def get_pcfgs(cls, pcfg_file: Path, pretty_format=False):
#         if not os.path.exists(pcfg_file):
#             rulesets = cls.get_treebank_rules(pcfg=True)
#             pcfg_dict = cls.convert_ruleset_to_dict(rulesets, prob=True)
#             Utils.write_json(pcfg_dict, pcfg_file, pretty_format=pretty_format)
#             return pcfg_dict
#         # end if
#         return Utils.read_json(pcfg_file)

