
# This script is for generating all tables used in paper

from typing import *

import collections
from pathlib import Path

# from seutil import LoggingUtils, IOUtils
from seutil import latex

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger


class Tables:

    METRICS = ["selfbleu", "acc", "f1"]
    METRICS_THEADS = {
        "selfbleu": r"\tBleu",
        "acc": r"\tAccuracy",
        "f1": r"\tF1-score",
    }

    FMT_INT = "{:,d}"
    FMT_PER = "{:.1%}"
    FMT_FLOAT = "{:,.2f}"

    LATEX_SYMBOL_MAP = {
        '&': '\&', '{': '\{', '}': '\}', '_': '\_'
    }

    @classmethod
    def make_tables(cls, which):
        paper_dir: Path = Macros.paper_dir
        tables_dir: Path = paper_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(which, list):
            which = [which]
        # end if

        for item in which: 
            if item == "lc-req":
                cls.make_numbers_lc_requirement(Macros.result_dir, tables_dir)
                cls.make_table_lc_requirement(Macros.result_dir, tables_dir)
            elif item == "selfbleu":
                cls.make_numbers_selfbleu(Macros.result_dir, tables_dir)
                cls.make_table_selfbleu(Macros.result_dir, tables_dir)
            else:
                cls.logger.warning(f"Unknown table {item}")
            # end if
        # end for
        return

    @classmethod
    def replace_latex_symbol(cls, string):
        for symbol in cls.LATEX_SYMBOL_MAP.keys():
            string = string.replace(symbol, cls.LATEX_SYMBOL_MAP[symbol])
        # end for
        return string
    
    @classmethod
    def make_numbers_lc_requirement(cls, results_dir: Path, tables_dir: Path):
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        output_file = latex.File(tables_dir / 'lc-requirement-numbers.tex')
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            l_split = l.strip().split('::')
            if len(l_split)>3:
                desc, search, exclude, transform = l_split[0], l_split[1], l_split[2], l_split[3]
                desc = cls.replace_latex_symbol(desc)
                search = r"\textbf{Search}~"+cls.replace_latex_symbol(search.split('<SEARCH>')[-1])
                search += "~exlcude: "+cls.replace_latex_symbol(exclude.split('<EXCLUDE>')[-1])
                transform = r"\textbf{Transform}~"+cls.replace_latex_symbol(transform.split('<TRANSFORM>')[-1])
            else:
                desc, search, exclude, transform = l_split[0], l_split[1], None, l_split[2]
                desc = cls.replace_latex_symbol(desc)
                search = r"\textbf{Search}~"+cls.replace_latex_symbol(search.split('<SEARCH>')[-1])
                transform = r"\textbf{Transform}~"+cls.replace_latex_symbol(transform.split('<TRANSFORM>')[-1])
            # end if
            output_file.append_macro(latex.Macro(f"lc_{l_i+1}_desc", desc))
            output_file.append_macro(latex.Macro(f"lc_{l_i+1}_search", search))
            output_file.append_macro(latex.Macro(f"lc_{l_i+1}_transform", transform))
        # end for
        output_file.save()
        return

    @classmethod
    def make_table_lc_requirement(cls, results_dir: Path, tables_dir: Path):
        output_file = latex.File(tables_dir / "lc-requirement-table.tex")

        # Header
        output_file.append(r"\begin{table*}[t]")
        output_file.append(r"\begin{small}")
        output_file.append(r"\begin{center}")
        output_file.append(r"\caption{\ReqTableCaption}")
        output_file.append(r"\begin{tabular}{p{5cm}||p{9cm}}")
        output_file.append(r"\toprule")

        # Content
        output_file.append(r"\tLc & \tRules \\")
        output_file.append(r"\midrule")

        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            l_split = l.split('::')
            len_l = len(l_split) # 3 or 4
            output_file.append("\multirow{3}{*}{\parbox{5cm}{" + \
                               f"LC{l_i+1}: " + latex.Macro(f"lc_{l_i+1}_desc").use() + "}}")
            output_file.append(" & " + latex.Macro(f"lc_{l_i+1}_search").use() + r"\\")
            output_file.append(" & " + latex.Macro(f"lc_{l_i+1}_transform").use() + r"\\")
            output_file.append(r"\\")
            output_file.append(r"\hline")
        # end for

        # Footer
        output_file.append(r"\bottomrule")
        output_file.append(r"\end{tabular}")
        output_file.append(r"\end{center}")
        output_file.append(r"\end{small}")
        output_file.append(r"\vspace{\ReqTableVSpace}")
        output_file.append(r"\end{table*}")
        output_file.save()
        return

    @classmethod
    def make_numbers_selfbleu(cls, results_dir: Path, tables_dir: Path):
        output_file = latex.File(tables_dir / 'selfbleu-numbers.tex')
        selfbleu_file = Macros.selfbleu_result_dir / "{task}_{search_dataset_name}_{selection_method}_testcase_selfbleu.json"
        selfbleu_res = Utils.read_json(selfbleu_file)
        our_num_data = selfbleu_res['num_data']
        our_score = selfbleu_res['score']
        bl_name = selfbleu_res['baseline_name']
        bl_num_data = selfbleu_res['baseline_num_data']
        bl_score = selfbleu_res['baseline_score']
        output_file.append_macro(latex.Macro("selfbleu_our_num_data", our_num_data))
        output_file.append_macro(latex.Macro("selfbleu_our_score", FMT_FLOAT.format(our_socre)))
        output_file.append_macro(latex.Macro("selfbleu_bl_name", bl_name))
        output_file.append_macro(latex.Macro("selfbleu_bl_num_data", bl_num_data))
        output_file.append_macro(latex.Macro("selfbleu_bl_score", FMT_FLOAT.format(bl_score)))

        output_file.save()
        return

    @classmethoe
    def make_table_selfbleu(cls, results_dir: Path, tables_dir: Path):
        
        return
