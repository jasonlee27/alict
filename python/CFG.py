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

from Macros import Macros
from Utils import Utils

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
            re_search = re.search('\(([A-Z]+)\s(.+)\)', tree._.parse_string)
            rlabel = re_search.group(1)
            rword = re_search.group(2)
            plabel = tree._.parent._.labels[0]
            if rlabel not in rule_dict.keys():
                rule_dict[rlabel] = [f'terminal::{rword}']
            elif (rlabel, rword) not in rule_dict[rlabel]:
                rule_dict[rlabel].append(f'terminal::{rword}')
            # end if
        else:
            llabel = left[0]
            if llabel not in rule_dict.keys():
                rule_dict[llabel] = list()
            # end if

            if len(list(tree._.children))>0:
                non_terminals = list()
                for r in tree._.children:
                    if len(list(r._.labels))>0:
                        non_terminals.append(r._.labels[0])
                    else:
                        re_search = re.search('\(([A-Z]+)\s(.+)\)', r._.parse_string)
                        rlabel = re_search.group(1)
                        rword = re_search.group(2)
                        non_terminals.append(rlabel)
                    # end if
                    rule_dict = cls.get_cfg_per_tree(r, rule_dict)
                # end for
                if len(non_terminals)>0 and (tuple(non_terminals) not in rule_dict[llabel]):
                    rule_dict[llabel].append(tuple(non_terminals))
                # end if
            else:
                re_search = re.search(f'\({llabel}\s\(([A-Z]+)\s(.+)\)\)$', tree._.parse_string)
                rlabel = re_search.group(1)
                rword = re_search.group(2)
                if (rlabel) not in rule_dict[llabel]:
                    rule_dict[llabel].append((rlabel))
                # end if
                if rlabel not in rule_dict.keys():
                    rule_dict[rlabel] = [f'terminal::{rword}']
                elif f'terminal::{rword}' not in rule_dict[rlabel]:
                    rule_dict[rlabel].append(f'terminal::{rword}')
                # end if
            # end if
        # end if
        return rule_dict

    @classmethod
    def get_cfg_dict_per_sent(cls, parser, sent, rule_dict):
        tree = cls.get_tree(parser,sent.strip())
        return cls.get_cfg_per_tree(tree, rule_dict)

    @classmethod
    def get_cfg_dict(cls, parser, sents, rule_dict):
        for s in sents:
            # tree = cls.get_tree(parser,s.strip())
            # rule_dict = cls.get_cfg_per_tree(tree, rule_dict)
            rule_dict = cls.get_cfg_dict_per_sent(parser, s, rule_dict)
        # end for
        return rule_dict

    # @classmethod
    # def convert_cfg_dict_to_str(cls, cfg_dict):
    #     cfg_str = ''
    #     for left, rights in cfg_dict.items():
    #         cfg_elem = f'{left} -> '
    #         for r_i, right in enumerate(rights):
    #             if type(right) is tuple:
    #                 r_str = ' '.join(right)
    #                 cfg_elem += f'{r_str}'
    #             elif right.startswith('terminal::'):
    #                 r_str = right.split('terminal::')[-1]
    #                 cfg_elem += f'\'{r_str}\''
    #             else:
    #                 cfg_elem += right
    #             # end if
    #             if r_i+1<len(rights):
    #                 cfg_elem += ' | '
    #             # end if
    #         # end for
    #         cfg_str += f'{cfg_elem}\n'
    #     # end for
    #     return cfg_str
    
    # @classmethod
    # def write_cfg(cls, cfg_str, cfg_file):
    #     with open(cfg_file, 'w') as f:
    #         f.write(cfg_str)
    #     # end with

    @classmethod
    def trim_cfg_dict(cls, cfg_dict):
        _cfg_dict = dict()
        for lhs, rhs in cfg_dict.copy().items():
            _rhs = list()
            for r in rhs:
                _r = list()
                if type(r) is tuple:
                    for x in r:
                        if x.startswith('terminal::'):
                            _r.append(f"\'{x.split('terminal::')[-1]}\'")
                        else:
                            _r.append(x)
                        # end if
                    # end for
                    _rhs.append(tuple(_r))
                else:
                    if r.startswith('terminal::'):
                        _r.append(f"\'{r.split('terminal::')[-1]}\'")
                    else:
                        _r.append(r)
                    # end if
                # end if
                if tuple(_r) not in _rhs:
                    _rhs.append(tuple(_r))
                # end if
            # end for
            _cfg_dict[lhs] = _rhs
        # end for
        return _cfg_dict

    @classmethod
    def get_cfgs(cls, data_file, cfg_file, pretty_format=False):
        parser = cls.load_parser()
        sents: List = Utils.read_txt(data_file)
        cfg_dict = cls.get_cfg_dict(parser,sents,{})
        # cfg_str = cls.convert_cfg_dict_to_str(cfg_dict)
        Utils.write_json(cls.trim_cfg_dict(cfg_dict), cfg_file, pretty_format=pretty_format)

    @classmethod
    def get_seed_cfg(cls, seed_input):
        parser = cls.load_parser()
        return cls.get_cfg_dict_per_sent(parser,seed_input,{})


class TreebankCFG:

    @classmethod
    def get_treebank_rules(cls):
        if 'treebank' not in sys.modules:
            from nltk.corpus import treebank
        # end if
        return list(set(rule for tree in treebank.parsed_sents() for rule in tree.productions()))

    @classmethod
    def convert_ruleset_to_dict(cls, ruleset):
        cfg_dict = dict()
        for r in ruleset:
            lhs = str(r.lhs())
            if lhs not in cfg_dict.keys():
                cfg_dict[lhs] = list()
            # end if
            r_tuple = tuple([f'\'{r}\'' if type(r) is str else str(r) for r in r.rhs()])
            if r_tuple not in cfg_dict[lhs]:
                cfg_dict[lhs].append(r_tuple)
            # end if
        # end for
        return cfg_dict

    # @classmethod
    # def convert_cfg_dict_to_str(cls, cfg_dict):
    #     cfg_str = ''
    #     for left, rights in cfg_dict.items():
    #         cfg_elem = f'{left} -> '
    #         for r_i, right in enumerate(rights):
    #             if type(right) is tuple:
    #                 r_str = ' '.join([f'\'{r}\'' if type(r) is str else str(r) for r in right])
    #                 cfg_elem += r_str
    #             else:
    #                 cfg_elem += f'\'{r_str}\''
    #             # end if
    #             if r_i+1<len(rights):
    #                 cfg_elem += ' | '
    #             # end if
    #         # end for
    #         cfg_str += f'{cfg_elem}\n'
    #     # end for
    #     return cfg_str

    # @classmethod
    # def write_cfg(cls, cfg_str, cfg_file):
    #     with open(cfg_file, 'w') as f:
    #         f.write(cfg_str)
    #     # end with

    @classmethod
    def write_cfg(cls, cfg_dict, cfg_file, pretty_format=False):
        with open(cfg_file, 'w') as f:
            if pretty_format:
                json.dump(cfg_dict, f, indent=4)
            else:
                json.dump(cfg_dict, f)
            # end if
        # end with

    @classmethod
    def get_cfgs(cls, cfg_file, pretty_format=False):
        rulesets = cls.get_treebank_rules()
        cfg_dict = cls.convert_ruleset_to_dict(rulesets)
        # cfg_str = cls.convert_cfg_dict_to_str(cfg_dict)
        Utils.write_json(cfg_dict, cfg_file, pretty_format=pretty_format)

def main():
    cfg_ref_file = Macros.result_dir / 'treebank_cfg.json'
    input_file = Macros.this_dir / 'ex.txt'
    cfg_ut_file = Macros.result_dir / 'ex_cfg.json'
    cfg_diff_file = Macros.result_dir / 'ex_treebank_cfg_diff.json'

    # generate grammars
    if not os.path.exists(cfg_ref_file):
        TreebankCFG.get_cfgs(cfg_ref_file, pretty_format=True)
    # end if
    if not os.path.exists(cfg_ut_file):
        BeneparCFG.get_cfgs(input_file, cfg_ut_file, pretty_format=True)
    # end if
    
    # compare the loded grammars
    cfg_diff = CFGDiff(
        cfg_ref_file=cfg_ref_file,
        cfg_ut_file=cfg_ut_file,
        write_diff=True, 
        diff_file=cfg_diff_file
    )


if __name__=='__main__':
    main()