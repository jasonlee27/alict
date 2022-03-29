
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
    def make_tables(cls, **options):
        paper_dir: Path = Macros.paper_dir
        tables_dir: Path = paper_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        which = options['which']
        
        if not isinstance(which, list):
            which = [which]
        # end if

        for item in which: 
            if item == "lc-req":
                cls.make_numbers_lc_requirement(Macros.result_dir, tables_dir)
                cls.make_table_lc_requirement(Macros.result_dir, tables_dir)
            elif item == "selfbleu":
                task = options['task']
                search_dataset = options['search_dataset_name']
                selection_method = options['selection_method']
                print(task, search_dataset, selection_method)
                cls.make_numbers_selfbleu(Macros.result_dir, tables_dir, task, search_dataset, selection_method)
                cls.make_table_selfbleu(Macros.result_dir, tables_dir, task, search_dataset, selection_method)
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
    def make_numbers_selfbleu(cls,
                              results_dir: Path,
                              tables_dir: Path,
                              task: str,
                              search_dataset_name: str,
                              selection_method: str):
        output_file = latex.File(tables_dir / 'selfbleu-numbers.tex')
        selfbleu_file = Macros.selfbleu_result_dir / "{task}_{search_dataset_name}_{selection_method}_testcase_selfbleu.json"
        selfbleu_res = Utils.read_json(selfbleu_file)
        our_scores = selfbleu_res['ours']
        for lc_i, lc in enumerate(our_scores.keys()):
            output_file.append_macro(latex.Macro(f"selfbleu_our_lc_{lc_i}_desc", lc))
            output_file.append_macro(latex.Macro(f"selfbleu_our_lc_{lc_i}_num_data", our_scores[lc]['num_data']))
            output_file.append_macro(latex.Macro(f"selfbleu_our_lc_{lc_i}_score", our_scores[lc]['score']))
        # end for

        checklist_scores = selfbleu_res['checklist']
        ours_lc_list = list(our_scores.keys())
        for lc in checklist_scores.keys():
            if lc==Macros.CHECKLIST_LC_LIST[8]:
                our_lc_i = ours_lc_list.index(Macros.CHECKLIST_LC_LIST[8])
                output_file.append_macro(latex.Macro("selfbleu_checklist_lc_{our_lc_i}_num_data", checklist_scores[lc]['num_data']))
                output_file.append_macro(latex.Macro("selfbleu_checklist_lc_{our_lc_i}_score", checklist_scores[lc]['score']))
            elif lc==Macros.CHECKLIST_LC_LIST[10]:
                our_lc_i = ours_lc_list.index('Parsing sentiment in (question, no) form')
                output_file.append_macro(latex.Macro("selfbleu_checklist_lc_{our_lc_i}_num_data", checklist_scores[lc]['num_data']))
                output_file.append_macro(latex.Macro("selfbleu_checklist_lc_{our_lc_i}_score", checklist_scores[lc]['score']))
            else:
                our_lc_i = ours_lc_list.index(Macros.LC_MAP[lc])
                output_file.append_macro(latex.Macro("selfbleu_checklist_lc_{our_lc_i}_num_data", checklist_scores[lc]['num_data']))
                output_file.append_macro(latex.Macro("selfbleu_checklist_lc_{our_lc_i}_score", checklist_scores[lc]['score']))
            # end if
        # end for
        output_file.save()
        return

    @classmethoe
    def make_table_selfbleu(cls,
                            results_dir: Path,
                            tables_dir: Path,
                            task: str,
                            search_dataset_name: str,
                            selection_method: str):
        output_file = latex.File(tables_dir / "selfbleu-table.tex")
        
        # Header
        output_file.append(r"\begin{table*}[t]")
        output_file.append(r"\begin{small}")
        output_file.append(r"\begin{center}")
        output_file.append(r"\caption{\SelfBleuTableCaption}")
        output_file.append(r"\begin{tabular}{p{5cm}||p{3cm}||p{3cm}}||p{3cm}||p{3cm}}")
        output_file.append(r"\toprule")

        # Content
        output_file.append(r"\tLc & \tOurNumData & \tOurSelfBleuScore & \tChecklistNumData & \tChecklistSelfBleuScore \\")
        output_file.append(r"\midrule")

        selfbleu_file = Macros.selfbleu_result_dir / "{task}_{search_dataset_name}_{selection_method}_testcase_selfbleu.json"
        selfbleu_res = Utils.read_json(selfbleu_file)
        our_scores = selfbleu_res['ours']
        for lc_i, lc in enumerate(our_scores.keys()):
            output_file.append("\multirow{3}{*}{\parbox{5cm}{" + \
                               f"LC{lc_i+1}: " + latex.Macro(f"selfbleu_our_lc_{lc_i}_desc").use() + "}}")
            output_file.append(" & " + latex.Macro(f"selfbleu_our_lc_{lc_i}_num_data").use() + r"\\")
            output_file.append(" & " + latex.Macro(f"selfbleu_our_lc_{lc_i}_score").use() + r"\\")
            output_file.append(" & " + latex.Macro(f"selfbleu_checklist_lc_{lc_i}_num_data").use() + r"\\")
            output_file.append(" & " + latex.Macro(f"selfbleu_checklist_lc_{lc_i}_score").use() + r"\\")
            output_file.append(r"\\")
            output_file.append(r"\hline")
        # end for

        # Footer
        output_file.append(r"\bottomrule")
        output_file.append(r"\end{tabular}")
        output_file.append(r"\end{center}")
        output_file.append(r"\end{small}")
        output_file.append(r"\vspace{\SelfBleuTableVSpace}")
        output_file.append(r"\end{table*}")
        output_file.save()
        return
