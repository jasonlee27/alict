# this script generates nlp grammar rule set 
# given sentence input set

import re
import nltk
import benepar
import spacy
from nltk.corpus import treebank

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
    def read_input_sents(cls, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        #end with
        return lines

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
    def get_cfg_dict(cls, parser, sents, rule_dict):
        for s in sents:
            tree = CFG.get_tree(parser,s.strip())
            rule_dict = CFG.get_cfg_per_tree(tree, rule_dict)
            # print(rule_dict)
            # print()
        # end for
        return rule_dict

    @classmethod
    def convert_cfg_dict_to_str(cls, cfg_dict):
        cfg_str = ''
        for left, rights in cfg_dict.items():
            cfg_elem = f'{left} -> '
            for r_i, right in enumerate(rights):
                if type(right) is tuple:
                    r_str = ' '.join(right)
                    cfg_elem += f'{r_str}'
                elif right.startswith('terminal::'):
                    r_str = right.split('terminal::')[-1]
                    cfg_elem += f'\'{r_str}\''
                else:
                    cfg_elem += right
                # end if
                if r_i+1<len(rights):
                    cfg_elem += ' | '
                # end if
            # end for
            cfg_str += f'{cfg_elem}\n'
        # end for
        return cfg_str
    
    @classmethod
    def write_cfg(cls, cfg_str, cfg_file):
        with open(cfg_file, 'w') as f:
            f.write(cfg_str)
        # end with


class TreebankCFG:

    @classmethod
    def get_treebank_cfgs(cls):
        rulesets = set(rule for tree in treebank.parsed_sents() for rule in tree.productions())
        print(f"Treebank #parsed sents: {len(treebank.parsed_sents())}")
        print(f"Treebank #parsed ruleset: {len(rulesets)}")
        for r in rulesets:
            print(r)
        # end for

def main():
    # parser = BeneparCFG.load_parser()
    # sents: List = BeneparCFG.read_input_sents('./ex.txt')
    # cfg_dict = BeneparCFG.get_cfg_dict(parser,sents,{})
    # cfg_str = BeneparCFG.convert_cfg_dict_to_str(cfg_dict)
    # BeneparCFG.write_cfg(cfg_str, './ex_cfg.txt')
    TreebankCFG.get_treebank_cfgs()

if __name__=='__main__':
    main()