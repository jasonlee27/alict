
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

        for item in which:
            if item == 'lc-req':
                cls.make_numbers_lc_requirement(Macros.result_dir, tables_dir)
                cls.make_table_lc_requirement(Macros.result_dir, tables_dir)
            elif item == 'test-results':
                task = options.pop('task', 'hs')
                search_dataset = options.pop('search_dataset_name', 'hatexplain')
                selection_method = options.pop('selection_method', 'random')
                cls.make_numbers_test_results(Macros.result_dir, tables_dir, task, search_dataset, selection_method)
                cls.make_table_test_results(Macros.result_dir, tables_dir, task, search_dataset, selection_method)
            elif item == 'test-results-baseline':
                task = options.pop('task', 'hs')
                search_dataset = options.pop('search_dataset_name', 'hatexplain')
                selection_method = options.pop('selection_method', 'random')
                num_seeds = options.pop('num_seeds', -1)
                num_trials = options.pop('num_trials', -1)
                cls.make_numbers_test_results_baseline(Macros.result_dir, tables_dir, task, search_dataset, selection_method, num_seeds, num_trials)
                cls.make_table_test_results_baseline(Macros.result_dir, tables_dir, task, search_dataset, selection_method, num_seeds, num_trials)
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
        req_file = req_dir / 'requirements_desc_hs.txt'
        output_file = latex.File(tables_dir / 'lc-hsd-requirement-numbers.tex')
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            l_split = l.strip().split('::')
            desc, search, transform = l_split[0], l_split[1], l_split[2]
            desc = cls.replace_latex_symbol(desc)
            search = r"\textbf{Search-based}~"+cls.replace_latex_symbol(search.split('<SEARCH>')[-1])
            transform = r"\textbf{Transform-based}~"+cls.replace_latex_symbol(transform.split('<TRANSFORM>')[-1])
            output_file.append_macro(latex.Macro(f"hsd_lc_{l_i+1}_desc", desc))
            output_file.append_macro(latex.Macro(f"hsd_lc_{l_i+1}_search", search))
            output_file.append_macro(latex.Macro(f"hsd_lc_{l_i+1}_transform", transform))
        # end for
        output_file.save()
        return

    @classmethod
    def make_table_lc_requirement(cls, results_dir: Path, tables_dir: Path):
        output_file = latex.File(tables_dir / "lc-hsd-requirement-table.tex")
        
        # Header
        output_file.append(r"\begin{table*}[t]")
        output_file.append(r"\begin{small}")
        output_file.append(r"\begin{center}")
        output_file.append(r"\caption{\HsdReqTableCaption}")
        output_file.append(r"\resizebox{0.8\textwidth}{!}{")
        output_file.append(r"\begin{tabular}{p{5cm}||p{12.5cm}}")
        output_file.append(r"\toprule")
        
        # Content
        output_file.append(r"\tLc & \tRules \\")
        output_file.append(r"\midrule")
        
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc_hs.txt'
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            l_split = l.split('::')
            len_l = len(l_split) # 3 or 4
            output_file.append("\multirow{2}{*}{\parbox{5cm}{" + \
                               f"LC{l_i+1}: " + latex.Macro(f"hsd_lc_{l_i+1}_desc").use() + "}}")
            output_file.append(" & " + latex.Macro(f"hsd_lc_{l_i+1}_search").use() + r"\\")
            output_file.append(" & " + latex.Macro(f"hsd_lc_{l_i+1}_transform").use() + r"\\")
            output_file.append(r"\hline")
        # end for

        # Footer
        output_file.append(r"\bottomrule")
        output_file.append(r"\end{tabular}}")
        output_file.append(r"\end{center}")
        output_file.append(r"\end{small}")
        output_file.append(r"\vspace{\HsdReqTableVSpace}")
        output_file.append(r"\end{table*}")
        output_file.save()
        return
    
    @classmethod
    def make_numbers_test_results(cls,
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

        output_file = latex.File(tables_dir / f"test-results-{task}-all-numbers.tex")
        
        lc_ids = dict()
        for l_i, l in enumerate(Macros.OUR_LC_LIST[6:]):
            desc = l.split('::')[1].strip()
            lc_ids[desc.lower()] = (desc,l_i)
            # desc = cls.replace_latex_symbol(desc)
            lc_descs[l_i] = desc
        # end for

        res_dir = result_dir / f"test_results_{task}_{search_dataset}_{selection_method}"
        result_file = res_dir / 'test_result_analysis.json'
        bl_result_file = res_dir / 'test_result_hatecheck_analysis.json'
        
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
                for bl_r in bl_result[model_name]:
                    bl_desc = bl_r['req'].split('::')[1]
                    if bl_desc.lower()==desc.lower():
                        bl_res = bl_r
                        break
                    # endfif
                # end for
                print(desc, _res_lc_i)
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
                    output_file.append_macro(latex.Macro(f"test-results-hs-lc{lc_i}", lc_descs[lc_i]))
                    output_file.append_macro(latex.Macro(f"test-results-hs-lc{lc_i}-num-seeds",
                                                         cls.FMT_INT.format(num_seeds_tot[m_name][lc_i][0])))

                    output_file.append_macro(latex.Macro(f"test-results-hs-lc{lc_i}-num-exps",
                                                         cls.FMT_INT.format(num_exps_tot[m_name][lc_i][0])))
                    output_file.append_macro(latex.Macro(f"test-results-hs-bl-lc{lc_i}-num-tcs",
                                                         cls.FMT_INT.format(bl_num_tc_tot[m_name][lc_i][0])))
                # end if

                output_file.append_macro(latex.Macro(f"test-results-hs-model{m_i}-lc{lc_i}-num-all-fail",
                                                     cls.FMT_INT.format(num_all_fail[m_name][lc_i][0])))
                output_file.append_macro(latex.Macro(f"test-results-hs-model{m_i}-lc{lc_i}-num-seed-failrate",
                                                     cls.FMT_FLOAT.format(num_seed_fail_rate[m_name][lc_i][0])))
                output_file.append_macro(latex.Macro(f"test-results-hs-model{m_i}-lc{lc_i}-num-exp-failrate",
                                                     cls.FMT_FLOAT.format(num_exp_fail_rate[m_name][lc_i][0])))
                output_file.append_macro(latex.Macro(f"test-results-hs-model{m_i}-lc{lc_i}-num-all-failrate",
                                                     cls.FMT_FLOAT.format(num_all_fail_rate[m_name][lc_i][0])))
                output_file.append_macro(latex.Macro(f"test-results-hs-model{m_i}-lc{lc_i}-num-pass-to-fail",
                                                     cls.FMT_INT.format(num_pass2fail[m_name][lc_i][0])))
                output_file.append_macro(latex.Macro(f"test-results-hs-bl-model{m_i}-lc{lc_i}-num-fail",
                                                     cls.FMT_INT.format(bl_num_tc_fail[m_name][lc_i][0])))
                output_file.append_macro(latex.Macro(f"test-results-hs-bl-model{m_i}-lc{lc_i}-num-failrate",
                                                     cls.FMT_FLOAT.format(bl_num_tc_fail_rate[m_name][lc_i][0])))
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
                                selection_method):
        output_file = latex.File(tables_dir / f"test-results-{task}-table.tex")
        res_dir = result_dir / f"test_results_{task}_{search_dataset}_{selection_method}"
        result_file = res_dir / 'test_result_analysis.json'
        bl_result_file = res_dir / 'test_result_hatecheck_analysis.json'
        
        result = Utils.read_json(result_file)
        bl_result = Utils.read_json(bl_result_file)
        model_names = list(result.keys())
        lcs_len = len(result[model_names[0]])
        
        # Header
        output_file.append(r"\begin{table*}[htbp]")
        output_file.append(r"\begin{small}")
        output_file.append(r"\begin{center}")
        output_file.append(r"\caption{\TestResultsHsTableCaption}")
        output_file.append(r"\resizebox{0.7\textwidth}{!}{")
        # output_file.append(r"\begin{tabular}{p{4cm}||p{1cm}p{2cm}p{1cm}p{2cm}p{1cm}p{2cm}p{2cm}}")
        output_file.append(r"\begin{tabular}{p{8cm}||ccclll}")
        output_file.append(r"\toprule")
        
        # output_file.append(r"\tLc & \parbox{1cm}{\tNumBlSents} & \parbox{1cm}{\tNumBlFail} & \parbox{1cm}{\tNumSeeds} & \parbox{1.5cm}{\tNumSeedFail} & \parbox{1cm}{\tNumExps} & \parbox{1.5cm}{\tNumExpFail} & \parbox{1.5cm}{\tNumPasstoFail}\\")
        output_file.append(r"\tLc & \parbox{1cm}{\tNumHsBlTcs} & \parbox{1cm}{\tNumSeeds} & \parbox{1cm}{\tNumExps} & \parbox{1.7cm}{\centering\tNumHsFail} & \parbox{1.7cm}{\centering\tHsFailRate} & \parbox{1.7cm}{\centering\tNumPasstoFail}\\")
        output_file.append(r"\midrule")
        
        # Content
        for lc_i in range(lcs_len):
            lc_prefix_str = f"LC{lc_i+1}: "
            # end if
            output_file.append("\multirow{"+str(len(model_names))+"}{*}{\parbox{8cm}{" + \
                               lc_prefix_str + latex.Macro(f"test-results-hs-lc{lc_i}").use() + "}}")
            output_file.append(" & \multirow{"+str(len(model_names))+"}{*}{\centering" + \
                               latex.Macro(f"test-results-hs-bl-lc{lc_i}-num-tcs").use() + "}")
            output_file.append(" & \multirow{"+str(len(model_names))+"}{*}{\centering" + \
                               latex.Macro(f"test-results-hs-lc{lc_i}-num-seeds").use() + "}")            
            output_file.append(" & \multirow{"+str(len(model_names))+"}{*}{\centering" + \
                               latex.Macro(f"test-results-hs-lc{lc_i}-num-exps").use() + "}")
            output_file.append(r" & alex-BERT$\colon$" + latex.Macro(f"test-results-hs-model0-lc{lc_i}-num-all-fail").use() + '/' + \
                               latex.Macro(f"test-results-hs-bl-model0-lc{lc_i}-num-fail").use())
            output_file.append(r" & alex-BERT$\colon$" + latex.Macro(f"test-results-hs-model0-lc{lc_i}-num-all-failrate").use() + '/' + \
                               latex.Macro(f"test-results-hs-bl-model0-lc{lc_i}-num-failrate").use())
            output_file.append(r" & alex-BERT$\colon$" + latex.Macro(f"test-results-hs-model0-lc{lc_i}-num-pass-to-fail").use() + r"\\")
            
            for m_i in range(1,len(model_names)):
                # if m_i==1:
                #     m_name = 'CNERG-BERT'
                # else:
                #     m_name = 'CNERG-BERT'
                # # end if
                m_name = 'CNERG-BERT'
                # output_file.append(f" & & {m_name}$\colon$" + latex.Macro(f"test-results-hs-bl-model{m_i}-lc{lc_i}-num-fail").use())
                output_file.append(f" & & & & {m_name}$\colon$" + latex.Macro(f"test-results-hs-model{m_i}-lc{lc_i}-num-all-fail").use() + '/' + \
                                   latex.Macro(f"test-results-hs-bl-model{m_i}-lc{lc_i}-num-fail").use())
                output_file.append(f" & {m_name}$\colon$" + latex.Macro(f"test-results-hs-model{m_i}-lc{lc_i}-num-all-failrate").use() + '/' + \
                                   latex.Macro(f"test-results-hs-bl-model{m_i}-lc{lc_i}-num-failrate").use())
                output_file.append(f" & {m_name}$\colon$" + latex.Macro(f"test-results-hs-model{m_i}-lc{lc_i}-num-pass-to-fail").use() + r"\\")
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
                                           num_seeds,
                                           num_trials):
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

        output_file = latex.File(tables_dir / f"test-results-{task}-bl-all-numbers.tex")
        
        lc_ids = dict()
        for l_i, l in enumerate(Macros.OUR_LC_LIST):
            desc = l.strip()
            lc_ids[desc.lower()] = (desc,l_i)
            lc_descs[l_i] = desc
        # end for

        if num_trials<=0:
            _num_trials = [-1]
        else:
            _num_trials = list(range(num_trials))
        # end if
        res_dir = result_dir / f"test_results_{task}_{search_dataset}_{selection_method}"
        result_file = res_dir / 'test_result_hatecheck_analysis.json'
        result = Utils.read_json(result_file)            
        for m_i, model_name in enumerate(result.keys()):
            if f"model{m_i}" not in num_seeds_tot.keys():
                num_seeds_tot[f"model{m_i}"] = dict()
                num_seed_fail[f"model{m_i}"] = dict()
                num_seed_fail_rate[f"model{m_i}"] = dict()
            # end if
                
            # model_name = model_name.replace('/', '-')
            temp_num_seeds, temp_num_exps = 0,0
            temp_num_seed_fail, temp_num_exp_fail = 0,0
            temp_num_pass2fail = 0
            for res_i, res in enumerate(result[model_name]):
                desc, _res_lc_i = lc_ids[res['req'].lower()]
                desc = desc.split('::')[1]
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
                    output_file.append_macro(latex.Macro(f"test-results-hs-bl-lc{lc_i}", lc_descs[lc_i]))
                    output_file.append_macro(latex.Macro(f"test-results-hs-bl-lc{lc_i}-num-tcs",
                                                         cls.FMT_INT.format(num_seeds_tot[m_name][lc_i][0])))
                # end if
                output_file.append_macro(latex.Macro(f"test-results-hs-bl-model{m_i}-lc{lc_i}-num-fail",
                                                     cls.FMT_INT.format(num_seed_fail[m_name][lc_i][0])))
                output_file.append_macro(latex.Macro(f"test-results-hs-bl-model{m_i}-lc{lc_i}-num-failrate",
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
                                         num_seeds,
                                         num_trials):
        output_file = latex.File(tables_dir / f"test-results-{task}-bl-table.tex")
        res_dir = result_dir / f"test_results_{task}_{search_dataset}_{selection_method}"
        result_file = res_dir / 'test_result_hatecheck_analysis.json'
        
        # result_file = result_dir / f"test_results_{task}_{search_dataset}_{selection_method}" / 'test_result_analysis.json'
        
        # baseline_result_file = result_dir / f"test_results_{task}_{search_dataset}_{selection_method}" / 'test_result_checklist_analysis.json'
        
        result = Utils.read_json(result_file)
        # baseline_result = Utils.read_json(baseline_result_file)
        model_names = list(result.keys())
        lcs_len = len(result[model_names[0]])
        
        # Header
        output_file.append(r"\begin{table*}[htbp]")
        output_file.append(r"\begin{small}")
        output_file.append(r"\begin{center}")
        output_file.append(r"\caption{\TestResultsHsBlTableCaption}")
        output_file.append(r"\resizebox{0.9\textwidth}{!}{")
        # output_file.append(r"\begin{tabular}{p{4cm}||p{1cm}p{2cm}p{1cm}p{2cm}p{1cm}p{2cm}p{2cm}}")
        output_file.append(r"\begin{tabular}{p{8cm}||cclll}")
        output_file.append(r"\toprule")
        
        # output_file.append(r"\tLc & \parbox{1cm}{\tNumBlSents} & \parbox{1cm}{\tNumBlFail} & \parbox{1cm}{\tNumSeeds} & \parbox{1.5cm}{\tNumSeedFail} & \parbox{1cm}{\tNumExps} & \parbox{1.5cm}{\tNumExpFail} & \parbox{1.5cm}{\tNumPasstoFail}\\")
        output_file.append(r"\tLc & \parbox{1cm}{\tNumSeeds} & \parbox{1.5cm}{\centering\tNumFail} & \parbox{1.5cm}{\centering\tFailRate}\\")
        output_file.append(r"\midrule")
        
        # Content
        for lc_i in range(lcs_len):
            lc_prefix_str = f"LC{lc_i+1}: "
            # end if
            output_file.append("\multirow{"+str(len(model_names))+"}{*}{\parbox{8cm}{" + \
                               lc_prefix_str + latex.Macro(f"test-results-hs-bl-lc{lc_i}").use() + "}}")
            output_file.append(" & \multirow{"+str(len(model_names))+"}{*}{\centering" + \
                               latex.Macro(f"test-results-hs-bl-lc{lc_i}-num-tcs").use() + "}")
            output_file.append(r" & alex-BERT$\colon$" + latex.Macro(f"test-results-hs-bl-model0-lc{lc_i}-num-fail").use())
            output_file.append(r" & alex-BERT$\colon$" + latex.Macro(f"test-results-hs-bl-model0-lc{lc_i}-num-failrate").use() + r"\\")
            
            for m_i in range(1,len(model_names)):
                if m_i==1:
                    m_name = 'CNERG-BERT'
                else:
                    m_name = 'CNERG-BERT'
                # end if
                output_file.append(f" & & {m_name}$\colon$" + latex.Macro(f"test-results-hs-bl-model{m_i}-lc{lc_i}-num-fail").use())
                output_file.append(f" & {m_name}$\colon$" + latex.Macro(f"test-results-hs-bl-model{m_i}-lc{lc_i}-num-failrate").use() + r"\\")
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
