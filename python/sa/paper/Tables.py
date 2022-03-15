
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
        '&': '\&', '{': '\{', '}': '\}'
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
                search = cls.replace_latex_symbol(search)
                exclude = cls.replace_latex_symbol(exclude)
                transform = cls.replace_latex_symbol(transform)
            else:
                desc, search, exclude, transform = l_split[0], l_split[1], None, l_split[2]
                desc = cls.replace_latex_symbol(desc)
                search = cls.replace_latex_symbol(search)
                transform = cls.replace_latex_symbol(transform)
            # end if
            output_file.append_macro(latex.Macro(f"lc_{l_i+1}_desc", desc))
            output_file.append_macro(latex.Macro(f"lc_{l_i+1}_search", search))
            if exclude is not None:
                output_file.append_macro(latex.Macro(f"lc_{l_i+1}_exclude", exclude))
            # end if
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
            output_file.append("\multirow{" + str(len_l-1) + "}{*}{\parbox{5cm}{" + \
                               f"LC{l_i+1}: " + latex.Macro(f"lc_{l_i+1}_desc").use() + "}}")
            output_file.append(" & " + latex.Macro(f"lc_{l_i+1}_search").use() + r"\\")
            if len_l>3:
                output_file.append(" & " + latex.Macro(f"lc_{l_i+1}_exclude").use() + r"\\")
            # end if
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
