
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
        '&': '\&',
        '{': '\{',
        '}': '\}',
        '_': '\_',
        ' < ': ' $<$ ',
        ' > ': ' $>$ ',
        '|': '$|$',
        '->': '$\\to$'
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

        task = options.pop('task', 'sa')
        search_dataset = options.pop('search_dataset_name', 'sst')
        selection_method = options.pop('selection_method', 'random')
        num_seeds = options.pop('num_seeds', 50)
        num_trials = options.pop('num_trials', 3)
        for item in which: 
            if item == "lc-req":
                cls.make_numbers_lc_requirement(Macros.result_dir, tables_dir)
                cls.make_table_lc_requirement(Macros.result_dir, tables_dir)
            # elif item == "selfbleu": # deprecated
            #     task = options.pop('task', 'sa')
            #     search_dataset = options.pop('search_dataset_name', 'sst')
            #     selection_method = options.pop('selection_method', 'random')
            #     cls.make_numbers_selfbleu(Macros.result_dir, tables_dir, task, search_dataset, selection_method)
            #     cls.make_table_selfbleu(Macros.result_dir, tables_dir, task, search_dataset, selection_method)
            # elif item == "retrain-debug": # deprecated
            #     task = options.pop('task', 'sa')
            #     search_dataset = options.pop('search_dataset_name', 'sst')
            #     selection_method = options.pop('selection_method', 'random')
            #     epochs = options.pop('epochs', 5)
            #     model_name = options.pop('model_name')
            #     cls.make_numbers_retrain(Macros.result_dir, tables_dir, task, search_dataset, selection_method, epochs, model_name)
            #     cls.make_table_retrain(Macros.result_dir, tables_dir, task, search_dataset, selection_method, epochs, model_name)
            elif item == "manual-study":
                cls.make_numbers_manual_study(Macros.result_dir, tables_dir, task, search_dataset, selection_method)
                cls.make_table_manual_study(Macros.result_dir, tables_dir, task, search_dataset, selection_method)
            elif item == "test-results":
                
                cls.make_numbers_test_results(Macros.result_dir, tables_dir, task, search_dataset, selection_method, num_seeds, num_trials)
                cls.make_table_test_results(Macros.result_dir, tables_dir, task, search_dataset, selection_method, num_seeds, num_trials)
            elif item == "test-results-all":
                cls.make_numbers_test_results_all(Macros.result_dir, tables_dir, task, search_dataset, selection_method)
                cls.make_table_test_results_all(Macros.result_dir, tables_dir, task, search_dataset, selection_method)
            elif item == "test-results-bl":
                cls.make_numbers_test_results_baseline(Macros.result_dir, tables_dir, task, search_dataset, selection_method)
                cls.make_table_test_results_baseline(Macros.result_dir, tables_dir, task, search_dataset, selection_method)
            elif item == 'mtnlp-comparison':
                cls.make_numbers_mtnlp_comparison(Macros.result_dir, tables_dir)
                cls.make_table_mtnlp_comparison(Macros.result_dir, tables_dir)
            elif item == 'adv-comparison':
                cls.make_numbers_adv_comparison(Macros.result_dir, tables_dir)
                cls.make_table_adv_comparison(Macros.result_dir, tables_dir)
            else:
                raise(f"Unknown table {item}")
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
            desc, search, transform = l_split[0], l_split[1], l_split[2]
            desc = cls.replace_latex_symbol(desc)
            search = r"\textbf{Search-based:}~"+cls.replace_latex_symbol(search.split('<SEARCH>')[-1])
            transform = r"\textbf{Enumerative:}~"+cls.replace_latex_symbol(transform.split('<TRANSFORM>')[-1])
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
            output_file.append("\multirow{2}{*}{\parbox{5cm}{" + \
                               f"LC{l_i+1}: " + latex.Macro(f"lc_{l_i+1}_desc").use() + "}}")
            output_file.append(" & " + latex.Macro(f"lc_{l_i+1}_search").use() + r"\\")
            output_file.append(" & " + latex.Macro(f"lc_{l_i+1}_transform").use() + r"\\")
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

    # @classmethod
    # def make_table_selfbleu(cls,
    #                         results_dir: Path,
    #                         tables_dir: Path,
    #                         task: str,
    #                         search_dataset_name: str,
    #                         selection_method: str):
    #     output_file = latex.File(tables_dir / "selfbleu-table.tex")
        
    #     # Header
    #     output_file.append(r"\begin{table*}[t]")
    #     output_file.append(r"\begin{small}")
    #     output_file.append(r"\begin{center}")
    #     output_file.append(r"\caption{\SelfBleuTableCaption}")
    #     output_file.append(r"\begin{tabular}{p{5cm}||p{3cm}||p{3cm}}||p{3cm}||p{3cm}}")
    #     output_file.append(r"\toprule")

    #     # Content
    #     output_file.append(r"\tLc & \tOurNumData & \tOurSelfBleuScore & \tChecklistNumData & \tChecklistSelfBleuScore \\")
    #     output_file.append(r"\midrule")

    #     selfbleu_file = Macros.selfbleu_result_dir / "{task}_{search_dataset_name}_{selection_method}_testcase_selfbleu.json"
    #     selfbleu_res = Utils.read_json(selfbleu_file)
    #     our_scores = selfbleu_res['ours']
    #     for lc_i, lc in enumerate(our_scores.keys()):
    #         output_file.append("\multirow{3}{*}{\parbox{5cm}{" + \
    #                            f"LC{lc_i+1}: " + latex.Macro(f"selfbleu_our_lc_{lc_i}_desc").use() + "}}")
    #         output_file.append(" & " + latex.Macro(f"selfbleu_our_lc_{lc_i}_num_data").use() + r"\\")
    #         output_file.append(" & " + latex.Macro(f"selfbleu_our_lc_{lc_i}_score").use() + r"\\")
    #         output_file.append(" & " + latex.Macro(f"selfbleu_checklist_lc_{lc_i}_num_data").use() + r"\\")
    #         output_file.append(" & " + latex.Macro(f"selfbleu_checklist_lc_{lc_i}_score").use() + r"\\")
    #         output_file.append(r"\\")
    #         output_file.append(r"\hline")
    #     # end for

    #     # Footer
    #     output_file.append(r"\bottomrule")
    #     output_file.append(r"\end{tabular}")
    #     output_file.append(r"\end{center}")
    #     output_file.append(r"\end{small}")
    #     output_file.append(r"\vspace{\SelfBleuTableVSpace}")
    #     output_file.append(r"\end{table*}")
    #     output_file.save()
    #     return

    # @classmethod
    # def make_numbers_retrain(cls,
    #                          results_dir: Path,
    #                          tables_dir: Path,
    #                          task: str,
    #                          search_dataset_name: str,
    #                          selection_method: str,
    #                          epochs: int,
    #                          retrain_model_name: str):
    #     CH_TO_OUR_MAP = {
    #         Macros.CHECKLIST_LC_LIST[0].replace(',', ' '): Macros.OUR_LC_LIST[0],
    #         Macros.CHECKLIST_LC_LIST[1].replace(',', ' '): Macros.OUR_LC_LIST[1],
    #         Macros.CHECKLIST_LC_LIST[2].replace(',', ' '): Macros.OUR_LC_LIST[2],
    #         Macros.CHECKLIST_LC_LIST[4].replace(',', ' '): Macros.OUR_LC_LIST[4],
    #         Macros.CHECKLIST_LC_LIST[5].replace(',', ' '): Macros.OUR_LC_LIST[5],
    #         Macros.CHECKLIST_LC_LIST[7].replace(',', ' '): Macros.OUR_LC_LIST[7],
    #         str(Macros.CHECKLIST_LC_LIST[8:10]).replace(',', ' '): Macros.OUR_LC_LIST[8]
    #     }

    #     # OUR_TO_CH_MAP = {
    #     #     Macros.OUR_LC_LIST[0].replace(',', ' '): Macros.CHECKLIST_LC_LIST[0],
    #     #     Macros.OUR_LC_LIST[1].replace(',', ' '): Macros.CHECKLIST_LC_LIST[1],
    #     #     Macros.OUR_LC_LIST[2].replace(',', ' '): Macros.CHECKLIST_LC_LIST[2],
    #     #     Macros.OUR_LC_LIST[4].replace(',', ' '): Macros.CHECKLIST_LC_LIST[4],
    #     #     Macros.OUR_LC_LIST[5].replace(',', ' '): Macros.CHECKLIST_LC_LIST[5],
    #     #     Macros.OUR_LC_LIST[7].replace(',', ' '): Macros.CHECKLIST_LC_LIST[7],
    #     #     Macros.OUR_LC_LIST[8].replace(',', ' '): [Macros.CHECKLIST_LC_LIST[8],
    #     #                                               Macros.CHECKLIST_LC_LIST[9]]
    #     # }
        
    #     output_file = latex.File(tables_dir / 'retrain-debug-numbers.tex')

    #     retrain_model_name = retrain_model_name.replace("/", "_")
    #     retrain_debug_file = Macros.home_result_dir / 'retrain' / f"debug_comparison_{retrain_model_name}.csv"
    #     retrain_debug_res = Utils.read_sv(retrain_debug_file, delimeter=',', is_first_attributes=True)
        
    #     attributes = retrain_debug_res['attributes']
    #     empty_l_i = retrain_debug_res['lines'].index(['', '', '', '', '', '', ''])
    #     for l_i, l in enumerate(retrain_debug_res['lines']):
    #         if l_i<empty_l_i:
    #             for att_i, att_val in enumerate(l):
    #                 att = attributes[att_i].strip()
    #                 if att=='eval_lc':
    #                     _att_val = CH_TO_OUR_MAP[att_val]
    #                     output_file.append_macro(latex.Macro(f"retrain_debug_{att}_{l_i}", _att_val))
    #                 else:
    #                     output_file.append_macro(latex.Macro(f"retrain_debug_{att}_{l_i}", att_val))
    #                 # end if
    #             # end for
    #         elif l_i>empty_l_i:
    #             for att_i, att_val in enumerate(l):
    #                 att = attributes[att_i].strip()
    #                 if att=='retrained_lc':
    #                     _att_val = CH_TO_OUR_MAP[att_val]
    #                     output_file.append_macro(latex.Macro(f"retrain_debug_{att}_{l_i}", _att_val))
    #                 else:
    #                     output_file.append_macro(latex.Macro(f"retrain_debug_{att}_{l_i}", att_val))
    #                 # end if
    #             # end for
    #         # end if
    #     # end for
    #     output_file.save()
    #     return

    # @classmethod
    # def make_table_retrain(cls,
    #                        results_dir: Path,
    #                        tables_dir: Path,
    #                        task: str,
    #                        search_dataset_name: str,
    #                        selection_method: str,
    #                        epochs: int,
    #                        retrain_model_name: str):
    #     output_file = latex.File(tables_dir / "retrain-debug-table.tex")

    #     retrain_model_name = retrain_model_name.replace("/", "_")
    #     retrain_debug_file = Macros.home_result_dir / 'retrain' / f"debug_comparison_{retrain_model_name}.csv"
    #     retrain_debug_res = Utils.read_sv(retrain_debug_file, delimeter=',', is_first_attributes=True)
        
    #     # Header
    #     output_file.append(r"\begin{table*}[t]")
    #     output_file.append(r"\begin{small}")
    #     output_file.append(r"\begin{center}")
    #     output_file.append(r"\caption{\RetrainDebugTableCaption}")
    #     output_file.append(r"\begin{tabular}{p{2cm}|p{2cm}|p{2cm}|c|c|c|c|c}")
    #     output_file.append(r"\toprule")

    #     # Content
    #     output_file.append(r"\tApproach & \tRetrainlc & \tEvallc & \tNumfailpass & \tNumfailbeforeretrain & \tNumpassbeforeretrain & \tNumfailafterretrain & \tNumpassafterretrain \\")
    #     output_file.append(r"\midrule")

    #     attributes = retrain_debug_res['attributes']
    #     empty_l_i = retrain_debug_res['lines'].index(['', '', '', '', '', '', ''])
    #     for l_i, l in enumerate(retrain_debug_res['lines']):
    #         if l_i!=empty_l_i:
    #             for att_i, att_val in enumerate(l):
    #                 att = attributes[att_i].strip()
    #                 if att_i==0:
    #                     output_file.append("\multirow{2}{*}{\parbox{2cm}{" + \
    #                                        latex.Macro(f"retrain_debug_{att}_{l_i}").use()+ "}}")
    #                 elif att_i in [1,2]:
    #                     output_file.append(" & \multirow{2}{*}{\parbox{2cm}{" + \
    #                                        latex.Macro(f"retrain_debug_{att}_{l_i}").use() + "}}")
    #                 else:
    #                     output_file.append(" & " + latex.Macro(f"retrain_debug_{att}_{l_i}").use())
    #                 # end if
    #             # end for
    #             output_file.append(r"\\")
    #             output_file.append(r"& & & & & \\")
    #             output_file.append(r"\hline")
    #         # end if
    #     # end for

    #     # Footer
    #     output_file.append(r"\bottomrule")
    #     output_file.append(r"\end{tabular}")
    #     output_file.append(r"\end{center}")
    #     output_file.append(r"\end{small}")
    #     output_file.append(r"\vspace{\RetrainDebugTableVSpace}")
    #     output_file.append(r"\end{table*}")
    #     output_file.save()
    #     return
    
    @classmethod
    def make_numbers_manual_study(cls,
                                  result_dir,
                                  tables_dir,
                                  task,
                                  search_dataset,
                                  selection_method):
        output_file = latex.File(tables_dir / 'manual-study-numbers.tex')
        tasks = ['sa', 'hs']
        for task in tasks:
            if task=='sa':
                search_dataset = 'sst'
                selection_method = 'random'
            elif task=='hs':
                search_dataset = 'hatexplain'
                selection_method = 'random'
            # end if
            result_file = result_dir / 'human_study' / f"{task}_{search_dataset}_{selection_method}" / "human_study_results.json"
            result = Utils.read_json(result_file)
            seed_res = result['agg']['seed']
            exp_res = result['agg']['exp']
            output_file.append_macro(latex.Macro(f"manual-study-{task}-seed-num-sents",
                                                 seed_res['num_sents']))
            output_file.append_macro(latex.Macro(f"manual-study-{task}-seed-label-consistency",
                                                 cls.FMT_FLOAT.format(seed_res['avg_label_score'],2)))
            output_file.append_macro(latex.Macro(f"manual-study-{task}-seed-lc-relevancy",
                                                 cls.FMT_FLOAT.format(seed_res['avg_lc_score'],2)))
            output_file.append_macro(latex.Macro(f"manual-study-{task}-seed-validity", '-'))
            output_file.append_macro(latex.Macro(f"manual-study-{task}-exp-num-sents",
                                                 exp_res['num_sents']))
            output_file.append_macro(latex.Macro(f"manual-study-{task}-exp-label-consistency",
                                                 cls.FMT_FLOAT.format(exp_res['avg_label_score'],2)))
            output_file.append_macro(latex.Macro(f"manual-study-{task}-exp-lc-relevancy",
                                                 cls.FMT_FLOAT.format(exp_res['avg_lc_score'],2)))
            output_file.append_macro(latex.Macro(f"manual-study-{task}-exp-validity",
                                                 cls.FMT_FLOAT.format(exp_res['avg_val_score'],2)))
        # end for
        output_file.save()
        return

    @classmethod
    def make_table_manual_study(cls,
                                result_dir,
                                tables_dir,
                                task,
                                search_dataset,
                                selection_method):
        tasks = ['sa', 'hs']
        output_file = latex.File(tables_dir / 'manual-study-table.tex')

        # Header
        output_file.append(r"\begin{table}[htbp]")
        output_file.append(r"\begin{small}")
        output_file.append(r"\begin{center}")
        output_file.append(r"\caption{\ManualStudyTableCaption}")
        output_file.append(r"\resizebox{0.47\textwidth}{!}{")
        output_file.append(r"\begin{tabular}{l|l|cccc}")
        output_file.append(r"\toprule")
        
        # Content
        output_file.append(r"\tTask & \tSentType & \tNumTestCases & \tAvgLabelCons & \tAvgLCRel & \tAvgExpVal \\")
        output_file.append(r"\midrule")
        
        for t_i, task in enumerate(tasks):
            task_latex = 'SA' if task=='sa' else 'HSD'
            output_file.append("\multirow{2}{*}{" + task_latex + "} ")
            output_file.append(" & SEED & " + latex.Macro(f"manual-study-{task}-seed-num-sents").use())
            output_file.append(" & " + latex.Macro(f"manual-study-{task}-seed-label-consistency").use())
            output_file.append(" & " + latex.Macro(f"manual-study-{task}-seed-lc-relevancy").use())
            output_file.append(" & " + latex.Macro(f"manual-study-{task}-seed-validity").use() + r"\\")
            output_file.append(" & EXP & " + latex.Macro(f"manual-study-{task}-exp-num-sents").use())
            output_file.append(" & " + latex.Macro(f"manual-study-{task}-exp-label-consistency").use())
            output_file.append(" & " + latex.Macro(f"manual-study-{task}-exp-lc-relevancy").use())
            output_file.append(" & " + latex.Macro(f"manual-study-{task}-exp-validity").use() + r"\\")
            if t_i+1<len(tasks):
                output_file.append(r"\hline")
            # end if
        # end for

        # Footer
        output_file.append(r"\bottomrule")
        output_file.append(r"\end{tabular}}")
        output_file.append(r"\end{center}")
        output_file.append(r"\end{small}")
        output_file.append(r"\vspace{\ManualStudyTableVSpace}")
        output_file.append(r"\end{table}")
        output_file.save()
        return

    
    @classmethod
    def make_numbers_test_results(cls,
                                  result_dir,
                                  tables_dir,
                                  task,
                                  search_dataset,
                                  selection_method,
                                  num_seeds=50,
                                  num_trials=3):
        lc_descs = dict()
        num_seeds_tot = dict()
        num_exps_tot = dict()
        num_seed_fail = dict()
        num_exp_fail = dict()
        num_all_fail = dict()
        num_seed_fail_rate = dict()
        num_exp_fail_rate = dict()
        num_all_fail_rate = dict()
        num_pass2fail = dict()

        req_dir = result_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        output_file = latex.File(tables_dir / f"test-results-{num_trials}-{num_seeds}-numbers.tex")
        lcs = Utils.read_txt(req_file)
        lc_ids = dict()
        for l_i, l in enumerate(lcs):
            desc = l.split('::')[0].strip()
            lc_ids[desc.lower()] = (desc,l_i)
            desc = cls.replace_latex_symbol(desc)
            lc_descs[l_i] = desc
        # end for

        # baseline_result_file = res_dir / 'test_result_checklist_analysis.json'
        # baseline_result = Utils.read_json(baseline_result_file)
        
        for num_trial in range(num_trials):
            _num_trial = '' if num_trial==0 else str(num_trial+1)
            res_dir = result_dir / f"test_results{_num_trial}_{task}_{search_dataset}_{selection_method}_{num_seeds}seeds"
            result_file = res_dir / 'test_result_analysis.json'
            result = Utils.read_json(result_file)
            for m_i, model_name in enumerate(result.keys()):
                if f"model{m_i}" not in num_seeds_tot.keys():
                    num_seeds_tot[f"model{m_i}"] = dict()
                    num_exps_tot[f"model{m_i}"] = dict()
                    num_seed_fail[f"model{m_i}"] = dict()
                    num_exp_fail[f"model{m_i}"] = dict()
                    num_all_fail[f"model{m_i}"] = dict()
                    num_seed_fail_rate[f"model{m_i}"] = dict()
                    num_exp_fail_rate[f"model{m_i}"] = dict()
                    num_all_fail_rate[f"model{m_i}"] = dict()
                    num_pass2fail[f"model{m_i}"] = dict()
                # end if
                
                # model_name = model_name.replace('/', '-')
                temp_num_seeds, temp_num_exps = 0,0
                temp_num_seed_fail, temp_num_exp_fail = 0,0
                temp_num_pass2fail = 0
                for res_i, res in enumerate(result[model_name]):
                    desc, _res_lc_i = lc_ids[res['req'].lower()]
                    if _res_lc_i not in num_seeds_tot[f"model{m_i}"].keys():
                        num_seeds_tot[f"model{m_i}"][_res_lc_i] = list()
                        num_exps_tot[f"model{m_i}"][_res_lc_i] = list()
                        num_seed_fail[f"model{m_i}"][_res_lc_i] = list()
                        num_exp_fail[f"model{m_i}"][_res_lc_i] = list()
                        num_all_fail[f"model{m_i}"][_res_lc_i] = list()
                        num_seed_fail_rate[f"model{m_i}"][_res_lc_i] = list()
                        num_exp_fail_rate[f"model{m_i}"][_res_lc_i] = list()
                        num_all_fail_rate[f"model{m_i}"][_res_lc_i] = list()
                        num_pass2fail[f"model{m_i}"][_res_lc_i] = list()
                    # end if
                    num_seeds_tot[f"model{m_i}"][_res_lc_i].append(res['num_seeds'])
                    num_exps_tot[f"model{m_i}"][_res_lc_i].append(res['num_exps'] if res['is_exps_exist'] else 0)
                    num_seed_fail[f"model{m_i}"][_res_lc_i].append(res['num_seed_fail'])
                    num_exp_fail[f"model{m_i}"][_res_lc_i].append(res['num_exp_fail'] if res['is_exps_exist'] else 0.)

                    num_all_fail[f"model{m_i}"][_res_lc_i].append(res['num_seed_fail']+res['num_exp_fail'] if res['is_exps_exist'] else res['num_seed_fail'])
                    
                    num_seed_fail_rate[f"model{m_i}"][_res_lc_i].append(res['num_seed_fail']*100./res['num_seeds'])
                    num_exp_fail_rate[f"model{m_i}"][_res_lc_i].append(res['num_exp_fail']*100./res['num_exps'] if res['is_exps_exist'] else 0.)
                    num_all_fail_rate[f"model{m_i}"][_res_lc_i].append((res['num_seed_fail']+res['num_exp_fail'])*100./(res['num_seeds']+res['num_exps']) if res['is_exps_exist'] else res['num_seed_fail']*100./res['num_seeds'])
                    num_pass2fail[f"model{m_i}"][_res_lc_i].append(res['num_pass2fail'] if res['is_exps_exist'] else 0.)
                # end for
            # end for
        # end for

        for m_i, m_name in enumerate(num_seeds_tot.keys()):
            temp_num_seeds, temp_num_exps = 0,0
            temp_num_seed_fail, temp_num_exp_fail = 0,0
            temp_num_pass2fail = 0

            # num_seeds_tot[f"model{m_i}"] = dict()
            # num_exps_tot[f"model{m_i}"] = dict()
            # num_seed_fail_rate[f"model{m_i}"] = dict()
            # num_exp_fail_rate[f"model{m_i}"] = dict()
            # num_pass2fail[f"model{m_i}"] = dict()

            for lc_i in num_seeds_tot[m_name].keys():
                if m_i==0:
                    output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-lc{lc_i}", lc_descs[lc_i]))
                    output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-lc{lc_i}-num-seeds",
                                                         num_seeds_tot[m_name][lc_i][0]))

                    output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-lc{lc_i}-num-exps-avg",
                                                         Utils.avg(num_exps_tot[m_name][lc_i], decimal=2)))
                    output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-lc{lc_i}-num-exps-med",
                                                         Utils.median(num_exps_tot[m_name][lc_i], decimal=2)))
                    output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-lc{lc_i}-num-exps-std",
                                                         str(float(Utils.stdev(num_exps_tot[m_name][lc_i], decimal=2))/2)))
                # end if
                # output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-seed-failrate-avg",
                #                                      Utils.avg(num_seed_fail_rate[m_name][lc_i], decimal=2)))
                # output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-seed-failrate-med",
                #                                      Utils.median(num_seed_fail_rate[m_name][lc_i], decimal=2)))
                # output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-seed-failrate-std",
                #                                      str(float(Utils.stdev(num_seed_fail_rate[m_name][lc_i], decimal=2))/2)))

                # output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-exp-failrate-avg",
                #                                      Utils.avg(num_exp_fail_rate[m_name][lc_i], decimal=2)))
                # output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-exp-failrate-med",
                #                                      Utils.median(num_exp_fail_rate[m_name][lc_i], decimal=2)))
                # output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-exp-failrate-std",
                #                                      str(float(Utils.stdev(num_exp_fail_rate[m_name][lc_i], decimal=2))/2)))

                output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-all-fail-avg",
                                                     Utils.avg(num_all_fail[m_name][lc_i], decimal=2)))
                output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-all-fail-med",
                                                     Utils.median(num_all_fail[m_name][lc_i], decimal=2)))
                output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-all-fail-std",
                                                     str(float(Utils.stdev(num_all_fail[m_name][lc_i], decimal=2))/2)))

                output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-all-failrate-avg",
                                                     Utils.avg(num_all_fail_rate[m_name][lc_i], decimal=2)))
                output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-all-failrate-med",
                                                     Utils.median(num_all_fail_rate[m_name][lc_i], decimal=2)))
                output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-all-failrate-std",
                                                     str(float(Utils.stdev(num_all_fail_rate[m_name][lc_i], decimal=2))/2)))

                
                output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-pass-to-fail-avg",
                                                     Utils.avg(num_pass2fail[m_name][lc_i], decimal=2)))
                output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-pass-to-fail-med",
                                                     Utils.median(num_pass2fail[m_name][lc_i], decimal=2)))
                output_file.append_macro(latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-pass-to-fail-std",
                                                     str(float(Utils.stdev(num_pass2fail[m_name][lc_i], decimal=2))/2)))
            # end for
        # end_for
        output_file.save()
        return
    
    @classmethod
    def make_table_test_results(cls,
                                result_dir,
                                tables_dir,
                                task,
                                search_dataset,
                                selection_method,
                                num_seeds,
                                num_trials):
        output_file = latex.File(tables_dir / f"test-results-{num_trials}-{num_seeds}-table.tex")
        res_dir = result_dir / f"test_results_{task}_{search_dataset}_{selection_method}_{num_seeds}seeds"
        result_file = res_dir / 'test_result_analysis.json'
        
        # result_file = result_dir / f"test_results_{task}_{search_dataset}_{selection_method}" / 'test_result_analysis.json'
        
        # baseline_result_file = result_dir / f"test_results_{task}_{search_dataset}_{selection_method}" / 'test_result_checklist_analysis.json'
        
        result = Utils.read_json(result_file)
        # baseline_result = Utils.read_json(baseline_result_file)
        model_names = list(result.keys())
        lcs_len = len(result[model_names[0]])
        
        # Header
        output_file.append(r"\begin{table*}[t]")
        output_file.append(r"\begin{small}")
        output_file.append(r"\begin{center}")
        output_file.append(r"\caption{\TestResultsTableCaption}")
        output_file.append(r"\resizebox{0.9\textwidth}{!}{")
        # output_file.append(r"\begin{tabular}{p{4cm}||p{1cm}p{2cm}p{1cm}p{2cm}p{1cm}p{2cm}p{2cm}}")
        output_file.append(r"\begin{tabular}{p{8cm}||cclll}")
        output_file.append(r"\toprule")
        
        # output_file.append(r"\tLc & \parbox{1cm}{\tNumBlSents} & \parbox{1cm}{\tNumBlFail} & \parbox{1cm}{\tNumSeeds} & \parbox{1.5cm}{\tNumSeedFail} & \parbox{1cm}{\tNumExps} & \parbox{1.5cm}{\tNumExpFail} & \parbox{1.5cm}{\tNumPasstoFail}\\")
        output_file.append(r"\tLc & \parbox{1cm}{\tNumSeeds} & \parbox{1cm}{\tNumExps} & \parbox{1.5cm}{\centering\tNumFail} & \parbox{1.5cm}{\centering\tFailRate} & \parbox{1.5cm}{\centering\tNumPasstoFail}\\")
        output_file.append(r"\midrule")
        
        # Content
        for lc_i in range(lcs_len):
            lc_prefix_str = f"LC{lc_i+1}: "
            # end if
            output_file.append("\multirow{"+str(len(model_names))+"}{*}{\parbox{8cm}{" + \
                               lc_prefix_str + latex.Macro(f"test-results-{num_trials}-{num_seeds}-lc{lc_i}").use() + "}}")
            # output_file.append(" & \multirow{"+str(len(model_names))+"}{*}{" + \
            #                    latex.Macro(f"test-results-bl-lc{lc_i}-num-sents").use() + "}")
            # output_file.append(r" & BERT$\colon$" + latex.Macro(f"test-results-bl-model0-lc{lc_i}-num-fail").use())

            output_file.append(" & \multirow{"+str(len(model_names))+"}{*}{\centering" + \
                               latex.Macro(f"test-results-{num_trials}-{num_seeds}-lc{lc_i}-num-seeds").use() + "}")            
            output_file.append(" & \multirow{"+str(len(model_names))+"}{*}{\centering" + \
                               latex.Macro(f"test-results-{num_trials}-{num_seeds}-lc{lc_i}-num-exps-med").use() + "}")
            
            output_file.append(r" & BERT$\colon$" + latex.Macro(f"test-results-{num_trials}-{num_seeds}-model0-lc{lc_i}-num-all-fail-med").use())
            output_file.append(r" & BERT$\colon$" + latex.Macro(f"test-results-{num_trials}-{num_seeds}-model0-lc{lc_i}-num-all-failrate-med").use())
            output_file.append(r" & BERT$\colon$" + latex.Macro(f"test-results-{num_trials}-{num_seeds}-model0-lc{lc_i}-num-pass-to-fail-med").use() + r"\\")
            
            for m_i in range(1,len(model_names)):
                if m_i==1:
                    m_name = 'RoBERTa'
                else:
                    m_name = 'dstBERT'
                # output_file.append(f" & & {m_name}$\colon$" + latex.Macro(f"test-results-bl-model{m_i}-lc{lc_i}-num-fail").use())
                output_file.append(f" & & & {m_name}$\colon$" + latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-all-fail-med").use())
                output_file.append(f" & {m_name}$\colon$" + latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-all-failrate-med").use())
                output_file.append(f" & {m_name}$\colon$" + latex.Macro(f"test-results-{num_trials}-{num_seeds}-model{m_i}-lc{lc_i}-num-pass-to-fail-med").use() + r"\\")
            # end for
            output_file.append(r"\hline")
        # end for
        # Footer
        output_file.append(r"\bottomrule")
        output_file.append(r"\end{tabular}}")
        output_file.append(r"\end{center}")
        output_file.append(r"\end{small}")
        output_file.append(r"\vspace{\TestResultsTableVSpace}")
        output_file.append(r"\end{table*}")
        output_file.save()
        return
    
    @classmethod
    def make_numbers_test_results_all(cls,
                                      result_dir,
                                      tables_dir,
                                      task,
                                      search_dataset,
                                      selection_method):
        lc_descs = dict()
        num_seeds_tot = dict()
        num_exps_tot = dict()
        num_seed_fail = dict()
        num_exp_fail = dict()
        num_all_fail = dict()
        num_seed_fail_rate = dict()
        num_exp_fail_rate = dict()
        num_all_fail_rate = dict()
        num_pass2fail = dict()

        bl_num_tc_tot = dict()
        bl_num_tc_fail = dict()
        bl_num_tc_fail_rate = dict()

        req_dir = result_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        output_file = latex.File(tables_dir / f"test-results-all-numbers.tex")
        lcs = Utils.read_txt(req_file)
        lc_ids = dict()
        for l_i, l in enumerate(lcs):
            desc = l.split('::')[0].strip()
            lc_ids[desc.lower()] = (desc,l_i)
            desc = cls.replace_latex_symbol(desc)
            lc_descs[l_i] = desc
        # end for

        res_dir = result_dir / f"test_results_{task}_{search_dataset}_{selection_method}"
        result_file = res_dir / 'test_result_analysis.json'
        bl_result_file = res_dir / 'test_result_checklist_analysis.json'
        result = Utils.read_json(result_file)
        bl_result = Utils.read_json(bl_result_file)
        for m_i, model_name in enumerate(result.keys()):
            if f"model{m_i}" not in num_seeds_tot.keys():
                num_seeds_tot[f"model{m_i}"] = dict()
                num_exps_tot[f"model{m_i}"] = dict()
                num_seed_fail[f"model{m_i}"] = dict()
                num_exp_fail[f"model{m_i}"] = dict()
                num_all_fail[f"model{m_i}"] = dict()
                num_seed_fail_rate[f"model{m_i}"] = dict()
                num_exp_fail_rate[f"model{m_i}"] = dict()
                num_all_fail_rate[f"model{m_i}"] = dict()
                num_pass2fail[f"model{m_i}"] = dict()

                bl_num_tc_tot[f"model{m_i}"] = dict()
                bl_num_tc_fail[f"model{m_i}"] = dict()
                bl_num_tc_fail_rate[f"model{m_i}"] = dict()
            # end if
            
            # model_name = model_name.replace('/', '-')
            temp_num_seeds, temp_num_exps = 0,0
            temp_num_seed_fail, temp_num_exp_fail = 0,0
            temp_num_pass2fail = 0
            for res_i, res in enumerate(result[model_name]):
                desc, _res_lc_i = lc_ids[res['req'].lower()]
                bl_res = [bl_r for bl_r in bl_result[model_name] if bl_r['req'].lower()==desc.lower()]
                bl_res = bl_res[0]
                if _res_lc_i not in num_seeds_tot[f"model{m_i}"].keys():
                    num_seeds_tot[f"model{m_i}"][_res_lc_i] = list()
                    num_exps_tot[f"model{m_i}"][_res_lc_i] = list()
                    num_seed_fail[f"model{m_i}"][_res_lc_i] = list()
                    num_exp_fail[f"model{m_i}"][_res_lc_i] = list()
                    num_all_fail[f"model{m_i}"][_res_lc_i] = list()
                    num_seed_fail_rate[f"model{m_i}"][_res_lc_i] = list()
                    num_exp_fail_rate[f"model{m_i}"][_res_lc_i] = list()
                    num_all_fail_rate[f"model{m_i}"][_res_lc_i] = list()
                    num_pass2fail[f"model{m_i}"][_res_lc_i] = list()
                    
                    bl_num_tc_tot[f"model{m_i}"][_res_lc_i] = list()
                    bl_num_tc_fail[f"model{m_i}"][_res_lc_i] = list()
                    bl_num_tc_fail_rate[f"model{m_i}"][_res_lc_i] = list()
                # end if
                num_seeds_tot[f"model{m_i}"][_res_lc_i].append(res['num_seeds'])
                num_exps_tot[f"model{m_i}"][_res_lc_i].append(res['num_exps'] if res['is_exps_exist'] else 0)
                num_seed_fail[f"model{m_i}"][_res_lc_i].append(res['num_seed_fail'])
                num_exp_fail[f"model{m_i}"][_res_lc_i].append(res['num_exp_fail'] if res['is_exps_exist'] else 0.)
                
                num_all_fail[f"model{m_i}"][_res_lc_i].append(res['num_seed_fail']+res['num_exp_fail'] if res['is_exps_exist'] else res['num_seed_fail'])
                    
                num_seed_fail_rate[f"model{m_i}"][_res_lc_i].append(res['num_seed_fail']*100./res['num_seeds'])
                num_exp_fail_rate[f"model{m_i}"][_res_lc_i].append(res['num_exp_fail']*100./res['num_exps'] if res['is_exps_exist'] else 0.)
                num_all_fail_rate[f"model{m_i}"][_res_lc_i].append((res['num_seed_fail']+res['num_exp_fail'])*100./(res['num_seeds']+res['num_exps']) if res['is_exps_exist'] else res['num_seed_fail']*100./res['num_seeds'])
                num_pass2fail[f"model{m_i}"][_res_lc_i].append(res['num_pass2fail'] if res['is_exps_exist'] else 0.)

                bl_num_tc_tot[f"model{m_i}"][_res_lc_i].append(bl_res['num_tcs'])
                bl_num_tc_fail[f"model{m_i}"][_res_lc_i].append(bl_res['num_tc_fail'])
                bl_num_tc_fail_rate[f"model{m_i}"][_res_lc_i].append(bl_res['num_tc_fail']*100./bl_res['num_tcs'])
            # end for
        # end for
        
        for m_i, m_name in enumerate(num_seeds_tot.keys()):
            temp_num_seeds, temp_num_exps = 0,0
            temp_num_seed_fail, temp_num_exp_fail = 0,0
            temp_num_pass2fail = 0

            # num_seeds_tot[f"model{m_i}"] = dict()
            # num_exps_tot[f"model{m_i}"] = dict()
            # num_seed_fail_rate[f"model{m_i}"] = dict()
            # num_exp_fail_rate[f"model{m_i}"] = dict()
            # num_pass2fail[f"model{m_i}"] = dict()

            for lc_i in num_seeds_tot[m_name].keys():
                if m_i==0:
                    output_file.append_macro(latex.Macro(f"test-results-all-lc{lc_i}", lc_descs[lc_i]))
                    output_file.append_macro(latex.Macro(f"test-results-all-lc{lc_i}-num-seeds",
                                                         cls.FMT_INT.format(num_seeds_tot[m_name][lc_i][0])))
                    output_file.append_macro(latex.Macro(f"test-results-all-lc{lc_i}-num-exps",
                                                         cls.FMT_INT.format(num_exps_tot[m_name][lc_i][0])))
                    
                    output_file.append_macro(latex.Macro(f"test-results-bl-lc{lc_i}-num-tcs",
                                                         cls.FMT_INT.format(bl_num_tc_tot[m_name][lc_i][0])))
                # end if
                output_file.append_macro(latex.Macro(f"test-results-all-model{m_i}-lc{lc_i}-num-fail",
                                                     cls.FMT_INT.format(num_all_fail[m_name][lc_i][0])))
                output_file.append_macro(latex.Macro(f"test-results-all-model{m_i}-lc{lc_i}-num-failrate",
                                                     cls.FMT_FLOAT.format(num_all_fail_rate[m_name][lc_i][0])))
                output_file.append_macro(latex.Macro(f"test-results-all-model{m_i}-lc{lc_i}-num-pass-to-fail",
                                                     cls.FMT_INT.format(num_pass2fail[m_name][lc_i][0])))
                
                output_file.append_macro(latex.Macro(f"test-results-bl-model{m_i}-lc{lc_i}-num-fail",
                                                     cls.FMT_INT.format(bl_num_tc_fail[m_name][lc_i][0])))
                output_file.append_macro(latex.Macro(f"test-results-bl-model{m_i}-lc{lc_i}-num-failrate",
                                                     cls.FMT_FLOAT.format(bl_num_tc_fail_rate[m_name][lc_i][0])))
            # end for
        # end_for
        output_file.save()
        return
    
    @classmethod
    def make_table_test_results_all(cls,
                                    result_dir,
                                    tables_dir,
                                    task,
                                    search_dataset,
                                    selection_method):
        output_file = latex.File(tables_dir / f"test-results-all-table.tex")
        res_dir = result_dir / f"test_results_{task}_{search_dataset}_{selection_method}"
        result_file = res_dir / 'test_result_analysis.json'
        bl_result_file = res_dir / 'test_result_checklist_analysis.json'
        result = Utils.read_json(result_file)
        bl_result = Utils.read_json(bl_result_file)
        # baseline_result = Utils.read_json(baseline_result_file)
        model_names = list(result.keys())
        lcs_len = len(result[model_names[0]])
        
        # Header
        output_file.append(r"\begin{table*}[t]")
        output_file.append(r"\begin{small}")
        output_file.append(r"\begin{center}")
        output_file.append(r"\caption{\TestResultsAllTableCaption}")
        output_file.append(r"\resizebox{0.9\textwidth}{!}{")
        output_file.append(r"\begin{tabular}{p{8cm}||ccclll}")
        output_file.append(r"\toprule")
        
        output_file.append(r"\tLc & \parbox{1.5cm}{\tNumBlTcs} & \parbox{1cm}{\tNumSeeds} & \parbox{1cm}{\tNumExps} & \parbox{1.5cm}{\centering\tNumFail} & \parbox{1.5cm}{\centering\tFailRate} & \parbox{1.5cm}{\centering\tNumPasstoFail}\\")
        output_file.append(r"\midrule")
        
        # Content
        for lc_i in range(lcs_len):
            lc_prefix_str = f"LC{lc_i+1}: "
            # end if
            output_file.append("\multirow{"+str(len(model_names))+"}{*}{\parbox{8cm}{" + \
                               lc_prefix_str + latex.Macro(f"test-results-all-lc{lc_i}").use() + "}}")
            output_file.append(" & \multirow{"+str(len(model_names))+"}{*}{\centering" + \
                               latex.Macro(f"test-results-bl-lc{lc_i}-num-tcs").use() + "}")       
            output_file.append(" & \multirow{"+str(len(model_names))+"}{*}{\centering" + \
                               latex.Macro(f"test-results-all-lc{lc_i}-num-seeds").use() + "}")
            output_file.append(" & \multirow{"+str(len(model_names))+"}{*}{\centering" + \
                               latex.Macro(f"test-results-all-lc{lc_i}-num-exps").use() + "}")
            
            output_file.append(r" & BERT$\colon$" + latex.Macro(f"test-results-all-model0-lc{lc_i}-num-fail").use() + '/' + \
                               latex.Macro(f"test-results-bl-model0-lc{lc_i}-num-fail").use())
            output_file.append(r" & BERT$\colon$" + latex.Macro(f"test-results-all-model0-lc{lc_i}-num-failrate").use() + '/' + \
                               latex.Macro(f"test-results-bl-model0-lc{lc_i}-num-failrate").use())
            output_file.append(r" & BERT$\colon$" + latex.Macro(f"test-results-all-model0-lc{lc_i}-num-pass-to-fail").use() + r"\\")
            
            for m_i in range(1,len(model_names)):
                if m_i==1:
                    m_name = 'RoBERTa'
                else:
                    m_name = 'dstBERT'
                # output_file.append(f" & & {m_name}$\colon$" + latex.Macro(f"test-results-bl-model{m_i}-lc{lc_i}-num-fail").use())
                output_file.append(f" & & & & {m_name}$\colon$" + latex.Macro(f"test-results-all-model{m_i}-lc{lc_i}-num-fail").use() + '/' + \
                                   latex.Macro(f"test-results-bl-model{m_i}-lc{lc_i}-num-fail").use())
                output_file.append(f" & {m_name}$\colon$" + latex.Macro(f"test-results-all-model{m_i}-lc{lc_i}-num-failrate").use() + '/' + \
                                   latex.Macro(f"test-results-bl-model{m_i}-lc{lc_i}-num-failrate").use())
                output_file.append(f" & {m_name}$\colon$" + latex.Macro(f"test-results-all-model{m_i}-lc{lc_i}-num-pass-to-fail").use() + r"\\")
            # end for
            output_file.append(r"\hline")
        # end for
        # Footer
        output_file.append(r"\bottomrule")
        output_file.append(r"\end{tabular}}")
        output_file.append(r"\end{center}")
        output_file.append(r"\end{small}")
        output_file.append(r"\vspace{\TestResultsTableVSpace}")
        output_file.append(r"\end{table*}")
        output_file.save()
        return


    @classmethod
    def make_numbers_test_results_baseline(cls,
                                           result_dir,
                                           tables_dir,
                                           task,
                                           search_dataset,
                                           selection_method,
                                           num_seeds=50,
                                           num_trials=3):
        lc_descs = dict()
        num_seeds_tot = dict()
        num_seed_fail = dict()
        num_seed_fail_rate = dict()

        req_dir = result_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        output_file = latex.File(tables_dir / f"test-results-bl-numbers.tex")
        lcs = Utils.read_txt(req_file)
        lc_ids = dict()
        for l_i, l in enumerate(lcs):
            desc = l.split('::')[0].strip()
            lc_ids[desc.lower()] = (desc,l_i)
            desc = cls.replace_latex_symbol(desc)
            lc_descs[l_i] = desc
        # end for
        
        res_dir = result_dir / f"test_results_{task}_{search_dataset}_{selection_method}_{num_seeds}seeds"
        result_file = res_dir / 'test_result_checklist_analysis.json'
        result = Utils.read_json(result_file)
        for m_i, model_name in enumerate(result.keys()):
            if f"model{m_i}" not in num_seeds_tot.keys():
                num_seeds_tot[f"model{m_i}"] = dict()
                num_seed_fail[f"model{m_i}"] = dict()
                num_seed_fail_rate[f"model{m_i}"] = dict()
            # end if
                
            # model_name = model_name.replace('/', '-')
            temp_num_seeds = 0
            temp_num_seed_fail = 0
            for res_i, res in enumerate(result[model_name]):
                if res['req'].lower() == "['parsing positive sentiment in (question, no) form', 'parsing negative sentiment in (question, no) form']":
                    req = "parsing sentiment in (question, no) form"
                    desc, _res_lc_i = lc_ids[req]
                else:
                    desc, _res_lc_i = lc_ids[res['req'].lower()]
                # end if
                if _res_lc_i not in num_seeds_tot[f"model{m_i}"].keys():
                    num_seeds_tot[f"model{m_i}"][_res_lc_i] = list()
                    num_seed_fail[f"model{m_i}"][_res_lc_i] = list()
                    num_seed_fail_rate[f"model{m_i}"][_res_lc_i] = list()
                # end if
                num_seeds_tot[f"model{m_i}"][_res_lc_i].append(res['num_tcs'])
                num_seed_fail[f"model{m_i}"][_res_lc_i].append(res['num_tc_fail'])                    
                num_seed_fail_rate[f"model{m_i}"][_res_lc_i].append(res['num_tc_fail']*100./res['num_tcs'])
            # end for
        # end for

        for m_i, m_name in enumerate(num_seeds_tot.keys()):
            temp_num_seeds = 0
            temp_num_seed_fail = 0
            temp_num_pass2fail = 0

            # num_seeds_tot[f"model{m_i}"] = dict()
            # num_exps_tot[f"model{m_i}"] = dict()
            # num_seed_fail_rate[f"model{m_i}"] = dict()
            # num_exp_fail_rate[f"model{m_i}"] = dict()
            # num_pass2fail[f"model{m_i}"] = dict()

            for lc_i in num_seeds_tot[m_name].keys():
                if m_i==0:
                    output_file.append_macro(latex.Macro(f"test-results-bl-lc{lc_i}", lc_descs[lc_i]))
                    output_file.append_macro(latex.Macro(f"test-results-bl-lc{lc_i}-num-seeds",
                                                         cls.FMT_INT.format(num_seeds_tot[m_name][lc_i][0])))
                # end if
                output_file.append_macro(latex.Macro(f"test-results-bl-model{m_i}-lc{lc_i}-num-seed-fail",
                                                     cls.FMT_INT.format(num_seed_fail[m_name][lc_i][0])))
                output_file.append_macro(latex.Macro(f"test-results-bl-model{m_i}-lc{lc_i}-num-seed-failrate",
                                                     cls.FMT_FLOAT.format(num_seed_fail_rate[m_name][lc_i][0])))
            # end for
        # end_for
        output_file.save()
        return

    @classmethod
    def make_table_test_results_baseline(cls,
                                         result_dir,
                                         tables_dir,
                                         task,
                                         search_dataset,
                                         selection_method,
                                         num_seeds=50,
                                         num_trials=3):
        output_file = latex.File(tables_dir / f"test-results-bl-table.tex")
        res_dir = result_dir / f"test_results_{task}_{search_dataset}_{selection_method}_{num_seeds}seeds"
        result_file = res_dir / 'test_result_checklist_analysis.json'

        result = Utils.read_json(result_file)
        # baseline_result = Utils.read_json(baseline_result_file)
        model_names = list(result.keys())
        lcs_len = len(result[model_names[0]])
        
        # Header
        output_file.append(r"\begin{table*}[t]")
        output_file.append(r"\begin{small}")
        output_file.append(r"\begin{center}")
        output_file.append(r"\caption{\TestResultsBlTableCaption}")
        output_file.append(r"\resizebox{0.9\textwidth}{!}{")
        output_file.append(r"\begin{tabular}{p{8cm}||cll}")
        output_file.append(r"\toprule")
        
        output_file.append(r"\tLc & \parbox{1cm}{\tNumSeeds} & \parbox{1.5cm}{\centering\tNumFail} & \parbox{1.5cm}{\centering\tFailRate}\\")
        output_file.append(r"\midrule")
        
        # Content
        for lc_i in range(lcs_len):
            lc_prefix_str = f"LC{lc_i+1}: "
            # end if
            output_file.append("\multirow{"+str(len(model_names))+"}{*}{\parbox{8cm}{" + \
                               lc_prefix_str + latex.Macro(f"test-results-bl-lc{lc_i}").use() + "}}")
            output_file.append(" & \multirow{"+str(len(model_names))+"}{*}{\centering" + \
                               latex.Macro(f"test-results-bl-lc{lc_i}-num-seeds").use() + "}")
            output_file.append(r" & BERT$\colon$" + latex.Macro(f"test-results-bl-model0-lc{lc_i}-num-seed-fail").use())
            output_file.append(r" & BERT$\colon$" + latex.Macro(f"test-results-bl-model0-lc{lc_i}-num-seed-failrate").use()+ r"\\")
            
            for m_i in range(1,len(model_names)):
                if m_i==1:
                    m_name = 'RoBERTa'
                else:
                    m_name = 'dstBERT'
                # output_file.append(f" & & {m_name}$\colon$" + latex.Macro(f"test-results-bl-model{m_i}-lc{lc_i}-num-fail").use())
                output_file.append(f" & & {m_name}$\colon$" + latex.Macro(f"test-results-bl-model{m_i}-lc{lc_i}-num-seed-fail").use())
                output_file.append(f" & {m_name}$\colon$" + latex.Macro(f"test-results-bl-model{m_i}-lc{lc_i}-num-seed-failrate").use()+ r"\\")
            # end for
            output_file.append(r"\hline")
        # end for
        # Footer
        output_file.append(r"\bottomrule")
        output_file.append(r"\end{tabular}}")
        output_file.append(r"\end{center}")
        output_file.append(r"\end{small}")
        output_file.append(r"\vspace{\TestResultsTableVSpace}")
        output_file.append(r"\end{table*}")
        output_file.save()
        return

    @classmethod
    def make_numbers_mtnlp_comparison(cls,
                                      result_dir,
                                      tables_dir):
        tasks = ['sa', 'hs']
        output_file = latex.File(tables_dir / f"mtnlp-comp-numbers.tex")
        for task in tasks:
            if task == 'sa':
                search_dataset_name = 'sst'
                selection_method = 'random'
            else:
                search_dataset_name = 'hatexplain'
                selection_method = 'random'
            # end if
            mtnlp_res_dir = result_dir / 'mtnlp' / f"{task}_{search_dataset_name}_{selection_method}_sample"
            pdr_res_file = mtnlp_res_dir / f"mtnlp_sample_{task}_{search_dataset_name}_{selection_method}_pdrcov.json"
            sb_res_file = mtnlp_res_dir / f"mtnlp_sample_{task}_{search_dataset_name}_{selection_method}_selfbleu.json"
            
            # Self-Bleu scores
            sb_res = Utils.read_json(sb_res_file)
            pdr_res = Utils.read_json(pdr_res_file)
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-ours-selfbleu-num-mutate",
                                                 cls.FMT_INT.format(sb_res['ours_exp']['num_data'])))
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-ours-selfbleu-avg",
                                                 Utils.avg(sb_res['ours_exp']['scores'], decimal=2)))
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-ours-selfbleu-med",
                                                 Utils.median(sb_res['ours_exp']['scores'], decimal=2)))
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-ours-selfbleu-std",
                                                 Utils.stdev(sb_res['ours_exp']['scores'], decimal=2)))
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-mtnlp-selfbleu-num-mutate",
                                                 cls.FMT_INT.format(sb_res['mtnlp']['num_data'])))
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-mtnlp-selfbleu-avg",
                                                 Utils.avg(sb_res['mtnlp']['scores'], decimal=2)))
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-mtnlp-selfbleu-med",
                                                 Utils.median(sb_res['mtnlp']['scores'], decimal=2)))
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-mtnlp-selfbleu-std",
                                                 Utils.stdev(sb_res['mtnlp']['scores'], decimal=2)))
            # PDR cov scores
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-ours-pdr-num-mutate",
                                                 cls.FMT_INT.format(pdr_res['ours_exp']['num_data'])))
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-ours-pdr-avg",
                                                 Utils.avg(pdr_res['ours_exp']['scores'], decimal=2)))
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-ours-pdr-med",
                                                 Utils.median(pdr_res['ours_exp']['scores'], decimal=2)))
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-ours-pdr-std",
                                                 Utils.stdev(pdr_res['ours_exp']['scores'], decimal=2)))
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-mtnlp-pdr-num-mutate",
                                                 cls.FMT_INT.format(pdr_res['mtnlp']['num_data'])))
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-mtnlp-pdr-avg",
                                                 Utils.avg(pdr_res['mtnlp']['scores'], decimal=2)))
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-mtnlp-pdr-med",
                                                 Utils.median(pdr_res['mtnlp']['scores'], decimal=2)))
            output_file.append_macro(latex.Macro(f"mtnlp-comp-{task}-mtnlp-pdr-std",
                                                 Utils.stdev(pdr_res['mtnlp']['scores'], decimal=2)))
        # end for
        output_file.save()
        return

    @classmethod
    def make_table_mtnlp_comparison(cls,
                                    result_dir,
                                    tables_dir):
        # mtnlp_res_dir = result_dir / 'mtnlp' / f"{task}_{search_dataset_name}_{selection_method}_sample"
        # prd_res_file = mtnlp_res_dir / f"mtnlp_sample_{task}_{search_dataset_name}_{selection_method}_pdrcov.json"
        # sb_res_file = mtnlp_res_dir / f"mtnlp_sample_{task}_{search_dataset_name}_{selection_method}_selfbleu.json"
        tasks = ['sa', 'hs']
        output_file = latex.File(tables_dir / f"mtnlp-comp-table.tex")
        
        # Header
        output_file.append(r"\begin{table}[htbp]")
        output_file.append(r"\begin{small}")
        output_file.append(r"\begin{center}")
        output_file.append(r"\caption{\MtnlpCompTableCaption}")
        output_file.append(r"\resizebox{0.47\textwidth}{!}{")
        output_file.append(r"\begin{tabular}{l|l|ccc}")
        output_file.append(r"\toprule")
        
        # Content
        output_file.append(r"\tTask & \tApproach & \tNummut & \tSelfbleu & \tPdrcov \\")
        output_file.append(r"\midrule")
        
        for t_i, task in enumerate(tasks):
            task_latex = 'SA' if task=='sa' else 'HSD'
            output_file.append("\multirow{2}{*}{" + task_latex + "} ")
            output_file.append(r" & S^{2}LCT & " + latex.Macro(f"mtnlp-comp-{task}-ours-pdr-num-mutate").use())
            output_file.append(r" & " + latex.Macro(f"mtnlp-comp-{task}-ours-selfbleu-avg").use() + '\pm')
            output_file.append(latex.Macro(f"mtnlp-comp-{task}-ours-selfbleu-std").use())
            output_file.append(r" & " + latex.Macro(f"mtnlp-comp-{task}-ours-pdr-avg").use() + '\pm'+
                               latex.Macro(f"mtnlp-comp-{task}-ours-pdr-std").use() + r"\\")
            output_file.append(r" & MT-NLP & " + latex.Macro(f"mtnlp-comp-{task}-mtnlp-pdr-num-mutate").use())
            output_file.append(r" & " + latex.Macro(f"mtnlp-comp-{task}-mtnlp-selfbleu-avg").use() + '\pm')
            output_file.append(latex.Macro(f"mtnlp-comp-{task}-mtnlp-selfbleu-std").use())
            output_file.append(r" & " + latex.Macro(f"mtnlp-comp-{task}-mtnlp-pdr-avg").use() + '\pm'+
                               latex.Macro(f"mtnlp-comp-{task}-mtnlp-pdr-std").use() + r"\\")
            if t_i+1<len(tasks):
                output_file.append(r"\hline")
            # end if
        # end for

        # Footer
        output_file.append(r"\bottomrule")
        output_file.append(r"\end{tabular}}")
        output_file.append(r"\end{center}")
        output_file.append(r"\end{small}")
        output_file.append(r"\vspace{\MtnlpCompTableVSpace}")
        output_file.append(r"\end{table*}")
        output_file.save()
        return

    @classmethod
    def make_numbers_adv_comparison(cls,
                                    result_dir,
                                    tables_dir):
        tasks = ['sa']
        output_file = latex.File(tables_dir / f"adv-comp-numbers.tex")
        for task in tasks:
            if task == 'sa':
                search_dataset_name = 'sst'
                selection_method = 'random'
            else:
                search_dataset_name = 'hatexplain'
                selection_method = 'random'
            # end if
            adv_res_dir = result_dir / 'textattack' / f"{task}_{search_dataset_name}_{selection_method}"
            adv_res_file = adv_res_dir / 'diversity_scores.json'
            
            # Self-Bleu scores
            adv_res = Utils.read_json(adv_res_file)
            output_file.append_macro(latex.Macro(f"adv-comp-{task}-model-under-test", adv_res['model_under_test']))
            for adv_name in adv_res.keys():
                if adv_name=='ours_exp':
                    _adv_name = 'ours'
                else:
                    _adv_name = adv_name
                # end if
                if adv_name!='model_under_test':
                    output_file.append_macro(latex.Macro(f"adv-comp-{task}-{_adv_name}-num-adv",
                                                         cls.FMT_INT.format(adv_res[adv_name]['num_data'])))
                    output_file.append_macro(latex.Macro(f"adv-comp-{task}-{_adv_name}-num-samples",
                                                         cls.FMT_INT.format(adv_res[adv_name]['num_samples'][0])))
                    
                    output_file.append_macro(latex.Macro(f"adv-comp-{task}-{_adv_name}-selfbleu-med",
                                                         Utils.median(adv_res[adv_name]['selfbleu_scores'])))
                    output_file.append_macro(latex.Macro(f"adv-comp-{task}-{_adv_name}-selfbleu-avg",
                                                         Utils.avg(adv_res[adv_name]['selfbleu_scores'])))
                    output_file.append_macro(latex.Macro(f"adv-comp-{task}-{_adv_name}-selfbleu-std",
                                                        Utils.stdev(adv_res[adv_name]['selfbleu_scores'])))
                    
                    output_file.append_macro(latex.Macro(f"adv-comp-{task}-{_adv_name}-pdrcov-med",
                                                         Utils.median(adv_res[adv_name]['pdrcov_scores'])))
                    output_file.append_macro(latex.Macro(f"adv-comp-{task}-{_adv_name}-pdrcov-avg",
                                                         Utils.avg(adv_res[adv_name]['pdrcov_scores'])))
                    output_file.append_macro(latex.Macro(f"adv-comp-{task}-{_adv_name}-pdrcov-std",
                                                         Utils.stdev(adv_res[adv_name]['pdrcov_scores'])))
                # end if
            # end for
        # end for
        output_file.save()
        return
    
    @classmethod
    def make_table_adv_comparison(cls,
                                  result_dir,
                                  tables_dir):
        # mtnlp_res_dir = result_dir / 'mtnlp' / f"{task}_{search_dataset_name}_{selection_method}_sample"
        # prd_res_file = mtnlp_res_dir / f"mtnlp_sample_{task}_{search_dataset_name}_{selection_method}_pdrcov.json"
        # sb_res_file = mtnlp_res_dir / f"mtnlp_sample_{task}_{search_dataset_name}_{selection_method}_selfbleu.json"
        tasks = ['sa']
        output_file = latex.File(tables_dir / f"adv-comp-table.tex")

        
        
        # Header
        output_file.append(r"\begin{table}[htbp]")
        output_file.append(r"\begin{small}")
        output_file.append(r"\begin{center}")
        output_file.append(r"\caption{\AdvCompTableCaption}")
        output_file.append(r"\resizebox{0.47\textwidth}{!}{")
        output_file.append(r"\begin{tabular}{l|ccc}")
        output_file.append(r"\toprule")
        
        # Content
        output_file.append(r"\tApproach & \tNummut & \tSelfbleu & \tPdrcov \\")
        output_file.append(r"\midrule")
        
        for t_i, task in enumerate(tasks):
            task_latex = 'SA' if task=='sa' else 'HSD'
            if task == 'sa':
                search_dataset_name = 'sst'
                selection_method = 'random'
            else:
                search_dataset_name = 'hatexplain'
                selection_method = 'random'
            # end if
            adv_res_dir = result_dir / 'textattack' / f"{task}_{search_dataset_name}_{selection_method}"
            adv_res_file = adv_res_dir / 'diversity_scores.json'
            adv_names = [
                a for a in Utils.read_json(adv_res_file).keys()
                if a!='model_under_test' and a!='ours_exp'
            ]
            output_file.append(r" \tool & " + latex.Macro(f"adv-comp-{task}-ours-num-adv").use())
            output_file.append(f" & " + latex.Macro(f"adv-comp-{task}-ours-selfbleu-avg").use() + '\pm' +
                               latex.Macro(f"adv-comp-{task}-ours-selfbleu-std").use())
            output_file.append(f" & " + latex.Macro(f"adv-comp-{task}-ours-pdrcov-avg").use() + '\pm' +
                               latex.Macro(f"adv-comp-{task}-ours-pdrcov-std").use() + r"\\")
            for adv_name in adv_names:
                if adv_name=='pso':
                    _adv_name = 'PSO'
                elif adv_name=='bert-attack':
                    _adv_name = 'BERT-Attack'
                elif adv_name=='alzantot':
                    _adv_name = 'Alzantot'
                # end if
                output_file.append(f"{_adv_name} & " +
                                   latex.Macro(f"adv-comp-{task}-{adv_name}-num-adv").use())
                output_file.append(f" & " + latex.Macro(f"adv-comp-{task}-{adv_name}-selfbleu-avg").use() + '\pm' +
                                   latex.Macro(f"adv-comp-{task}-{adv_name}-selfbleu-std").use())
                output_file.append(f" & " + latex.Macro(f"adv-comp-{task}-{adv_name}-pdrcov-avg").use() + '\pm' +
                                   latex.Macro(f"adv-comp-{task}-{adv_name}-pdrcov-std").use() + r"\\")
            # end for
            if t_i+1<len(tasks):
                output_file.append(r"\hline")
            # end if
        # end for

        # Footer
        output_file.append(r"\bottomrule")
        output_file.append(r"\end{tabular}}")
        output_file.append(r"\end{center}")
        output_file.append(r"\end{small}")
        output_file.append(r"\vspace{\AdvCompTableVSpace}")
        output_file.append(r"\end{table*}")
        output_file.save()
        return
