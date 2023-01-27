
# This script is for generating all plots used in paper

import math
import pandas as pd
import seaborn as sns
import matplotlib as mpl

from typing import *
from pathlib import Path
from seutil import IOUtils
from matplotlib import pyplot as plt

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger

# from hdlp.Macros import Macros
# from hdlp.Utils import Utils


class Plots:

    @classmethod
    def make_plots(cls, **options):
        sns.set()
        # sns.set_palette(sns.cubehelix_palette(6, start=2.1, rot=1, gamma=0.9, hue=1, light=0.7, dark=0.2, reverse=True))
        sns.set_palette("Dark2")
        sns.set_context("paper")

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.titlesize'] = 'x-large'
        mpl.rcParams['axes.labelsize'] = 'large'
        mpl.rcParams['xtick.labelsize'] = 'large'
        mpl.rcParams['ytick.labelsize'] = 'large'

        task = options.pop('task', 'sa')
        search_dataset = options.pop('search_dataset_name', 'sst')
        selection_method = options.pop('selection_method', 'random')
        # num_seeds = options.pop('num_seeds', 50)
        # num_trials = options.pop('num_trials', 3)
        
        which: Union[str, List[str]] = options.pop("which", [])
        paper_dir: Path = Path(options.pop("paper_dir", Macros.paper_dir))
        figs_dir: Path = paper_dir / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(which, list):
            which = [which]
        # end if

        for item in which:
            if item == 'pdr-selfbleu-agg':
                # Results of Self-BLEU (left) and Production Rule Coverage (right) of S2LCT and CHECKLIST test cases.
                cls.pdr_selfbleu_agg_plot(Macros.result_dir, figs_dir)
            elif item == 'selfbleu':
                cls.selfbleu_sample_plot(Macros.result_dir, figs_dir)
            elif item == 'pdr':
                # Results of production rule coverage between S2LCT with 50 seeds and CHECKLIST test cases
                cls.pdr_bar_plot(Macros.result_dir, figs_dir)
            elif item == 'test-results':
                # cls.failrate_combined_over_seeds_plot(Macros.result_dir, figs_dir)
                # cls.failrate_seed_over_seeds_plot(Macros.result_dir, figs_dir)
                # cls.failrate_exp_over_seeds_plot(Macros.result_dir, figs_dir)
                # cls.pass2fail_over_seeds_plot(Macros.result_dir, figs_dir)
                # cls.numfail_over_seeds_plot(Macros.result_dir, figs_dir)
                cls.pass2fail_agg_over_seeds_plot(Macros.result_dir, figs_dir)
                cls.numfail_agg_over_seeds_plot(Macros.result_dir, figs_dir)
            elif item == 'numfail-pass2fail-agg':
                cls.numfail_pass2fail_agg_over_seeds_plot(Macros.result_dir, figs_dir)
            else:
                raise(f"Unknown plot {item}")
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
    def selfbleu_sample_plot(cls,
                             results_dir: Path,
                             figs_dir: Path,
                             task=Macros.sa_task,
                             search_dataset_name=Macros.datasets[Macros.sa_task][0],
                             selection_method='random'):
        num_trials = 10
        x_ticks = [100, 200, 300, 400, 500]
        result_file = results_dir / 'selfbleu' / f"seed_exp_bl_sample_{task}_{search_dataset_name}_{selection_method}_selfbleu.json"
        result = Utils.read_json(result_file)
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, lc in enumerate(Utils.read_txt(req_file)):
            data_lod: List[dict] = list()
            lc_desc = lc.strip().split('::')[0]
            lc_desc = lc_desc if lc_desc in result.keys() else lc_desc.lower()
            for s_i, num_sample in enumerate(result[lc_desc]['ours_seed'].keys()):
                _num_sample = int(num_sample.split('sample')[0])
                for t in range(num_trials):
                    data_lod.append({
                        'sample': s_i,
                        'lc': f"LC{l_i+1}",
                        'type': 'S$^2$LCT (SEED)',
                        'trial': t,
                        'num_seed': _num_sample,
                        'scores': result[lc_desc]['ours_seed'][num_sample]['selfbleu_scores'][t]
                    })
                    data_lod.append({
                        'sample': s_i,
                        'lc': f"LC{l_i+1}",
                        'type': 'S$^2$LCT (SEED+EXP)',
                        'trial': t,
                        'num_seed': _num_sample,
                        'scores': result[lc_desc]['ours_seed_exp'][num_sample]['selfbleu_scores'][t]
                    })
                    data_lod.append({
                        'sample': s_i,
                        'lc': f"LC{l_i+1}",
                        'type': 'CHECKLIST',
                        'trial': t,
                        'num_seed': _num_sample,
                        'scores': result[lc_desc]['bl'][num_sample]['selfbleu_scores'][t]
                    })
                # end for
            # end for
            df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))
            
            # Plotting part
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.subplots()
            
            hue_order = ['CHECKLIST', 'S$^2$LCT (SEED)', 'S$^2$LCT (SEED+EXP)']
            markers = ['$1$', '$2$', '$3$']
            from numpy import median
            ax = sns.lineplot(data=df, x='num_seed', y='scores',
                              hue='type',
                              hue_order=hue_order,
                              style='type',
                              estimator=median,
                              err_style="bars",
                              markers=True,
                              markersize=9,
                              markeredgewidth=0,
                              dashes=True,
                              palette="Set1",
                              err_kws={'capsize': 3},
                              ax=ax)
            plt.xticks(x_ticks)
            ax.set_ylim(0.0, 1.2)
            ax.set_xlabel("Number of seed samples")
            ax.set_ylabel("Self-BLEU")
            
            # Shrink current axis by 20%
            box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(0.68, 0.89))
            fig.tight_layout()
            fig.savefig(figs_dir / f"selfbleu-sample-lc{l_i+1}-lineplot.eps")
        # end for
        return

    @classmethod
    def pdr_bar_plot(cls,
                     results_dir: Path,
                     figs_dir: Path,
                     task=Macros.sa_task,
                     search_dataset_name=Macros.datasets[Macros.sa_task][0],
                     selection_method='random'):
        data_lod: List[dict] = list()
        x_ticks = [0] # , 1:100, 2:200}
        result_file = results_dir / 'pdr_cov' / f"seed_exp_bl_all_{task}_{search_dataset_name}_{selection_method}_pdrcov.json"
        result = Utils.read_json(result_file)
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, lc in enumerate(Utils.read_txt(req_file)):
            lc_desc = lc.strip().split('::')[0]
            lc_desc = lc_desc if lc_desc in result.keys() else lc_desc.lower()
            for ns in x_ticks:
                data_lod.append({
                    'lc': f"LC{l_i+1}",
                    'type': 'CHECKLIST',
                    'num_seed': ns,
                    'scores': math.log10(result[lc_desc]['bl']['coverage_scores'])
                })
                # print(l_i+1, bl_score, result[lc_desc]['ours_seed_exp']['med_score'], float(result[lc_desc]['ours']['med_score'])/bl_score)
                # data_lod.append({
                #     'lc': f"LC{l_i+1}",
                #     'type': 'S$^2$LCT (SEED)',
                #     'num_seed': ns,
                #     'scores': math.log10(result[lc_desc]['ours_seed']['coverage_scores'])
                # })
                data_lod.append({
                    'lc': f"LC{l_i+1}",
                    'type': 'S$^2$LCT (SEED+EXP)',
                    'num_seed': ns,
                    'scores': math.log10(result[lc_desc]['ours_seed_exp']['coverage_scores'])
                })
            # end for
        # end for
        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        # hue_order = ['CHECKLIST', 'S$^2$LCT (SEED)', 'S$^2$LCT (SEED+EXP)']
        hue_order = ['CHECKLIST', 'S$^2$LCT (SEED+EXP)']
        from numpy import median
        ax = sns.barplot(data=df, x='lc', y='scores',
                         hue='type',
                         hue_order=hue_order,
                         palette="Set1",
                         estimator=median)
        # plt.xticks([f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))])
        # ax.set_ylim(bottom=0, top=max(data_lod, key=lambda x: x['scores'])['scores']+10)
        ax.set_ylim(bottom=0, top=6)
        ax.set_xlabel("Linguistic Capabilities")
        ax.set_ylabel("Log of Number of Production Rules Covered")
        
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Put a legend to the right of the current axis
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
        fig.tight_layout()
        fig.savefig(figs_dir / "pdr-barplot.eps")
        return
    
    # @classmethod
    # def pdr_agg_plot(cls,
    #                  results_dir: Path,
    #                  figs_dir: Path,
    #                  task=Macros.sa_task,
    #                  search_dataset_name=Macros.datasets[Macros.sa_task][0],
    #                  selection_method='random'):
    #     data_lod: List[dict] = list()
    #     # num_seeds = [0,50,100,200] # x-axis
    #     x_ticks = {0:50, 1:100, 2:200}
    #     num_seeds = list(x_ticks.keys())
    #     req_dir = results_dir / 'reqs'
    #     req_file = req_dir / 'requirements_desc.txt'
        
    #     for ns in x_ticks.values():
    #         for l_i, l in enumerate(Utils.read_txt(req_file)):
    #             lc_desc = l.strip().split('::')[0].lower()
    #             result_file = results_dir / 'pdr_cov' / f"seeds_exps_over3_{task}_{search_dataset_name}_{selection_method}_{ns}seeds_pdrcov.json"
    #             result = Utils.read_json(result_file)
    #             data_lod.append({
    #                 'lc': f"LC{l_i+1}",
    #                 'type': 'S$^2$LCT(seed)',
    #                 'num_seed': ns,
    #                 'scores': float(result[lc_desc]['ours_seed']['med_score'])
    #             })
    #             data_lod.append({
    #                 'lc': f"LC{l_i+1}",
    #                 'type': 'S$^2$LCT(seed+exp)',
    #                 'num_seed': ns,
    #                 'scores': float(result[lc_desc]['ours_seed_exp']['med_score'])
    #             })

    #             result_bl_file = results_dir / 'pdr_cov' / f"seeds_sample_over3_{task}_{search_dataset_name}_{selection_method}_{ns}seeds_pdrcov.json"
    #             result_bl = Utils.read_json(result_bl_file)
    #             data_lod.append({
    #                 'lc': f"LC{l_i+1}",
    #                 'type': 'CHECKLIST', 
    #                 'num_seed': ns,
    #                 'scores': float(result_bl[lc_desc]['bl']['med_score'])
    #             })
    #         # end for
    #     # end for
        
    #     df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))
        
    #     # Plotting part
    #     fig: plt.Figure = plt.figure()
    #     ax: plt.Axes = fig.subplots()
        
    #     hue_order = ['S$^2$LCT(seed)', 'S$^2$LCT(seed+exp)', 'CHECKLIST']
        
    #     ax = sns.lineplot(data=df, x='num_seed', y='scores',
    #                       hue='type',
    #                       hue_order=hue_order,
    #                       estimator='median',
    #                       style='type',
    #                       err_style=None, # or "bars"
    #                       markers=True,
    #                       markersize=9,
    #                       markeredgewidth=0,
    #                       dashes=True,
    #                       errorbar='sd',
    #                       ax=ax)
    #     plt.xticks(list(x_ticks.values()))
    #     ax.set_ylim(0, 850)
    #     ax.set_xlabel("Number of seeds")
    #     ax.set_ylabel("Number of Production Rules Covered")
        
    #     # Shrink current axis by 20%
    #     box = ax.get_position()
    #     ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        
    #     # Put a legend to the right of the current axis
    #     # ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
    #     fig.tight_layout()
    #     fig.savefig(figs_dir / "pdr-agg-lineplot.eps")
    #     return

    @classmethod
    def pdr_selfbleu_agg_plot(cls,
                              results_dir: Path,
                              figs_dir: Path,
                              task=Macros.sa_task,
                              search_dataset_name=Macros.datasets[Macros.sa_task][0],
                              selection_method='random'):
        # num_seeds = [0,50,100,200] # x-axis
        x_ticks = [200, 400, 600, 800, 1000]
        num_trials = 10
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        selfbleu_result_file = results_dir / 'selfbleu' / f"seed_exp_bl_sample_{task}_{search_dataset_name}_{selection_method}_selfbleu.json"
        pdr_cov_result_file = results_dir / 'pdr_cov' / f"seed_exp_bl_sample_{task}_{search_dataset_name}_{selection_method}_pdrcov.json"
        
        pdr_cov_result = Utils.read_json(pdr_cov_result_file)
        selfbleu_result = Utils.read_json(selfbleu_result_file)
        for l_i, lc in enumerate(Utils.read_txt(req_file)):
            data_pdr_lod: List[dict] = list()
            data_sb_lod: List[dict] = list()
            lc_desc = lc.strip().split('::')[0]
            lc_desc = lc_desc if lc_desc in pdr_cov_result.keys() else lc_desc.lower()
            _pdr_x_ticks = [int(s.split('sample')[0]) for s in pdr_cov_result[lc_desc]['ours_seed'].keys()]
            temp = len(_pdr_x_ticks)//4
            pdr_x_ticks = [x for x_i, x in enumerate(_pdr_x_ticks) if x_i%temp==0][:-1] +[_pdr_x_ticks[-1]]
            pdr_y_limit = -1
            sb_y_limit = -1
            for t in range(num_trials):
                for ns in pdr_x_ticks:
                    data_pdr_lod.append({
                        'lc': f"LC{l_i+1}",
                        'type': 'S$^2$LCT (SEED)',
                        'num_seed': ns,
                        'scores': pdr_cov_result[lc_desc]['ours_seed'][f"{ns}sample"]['coverage_scores'][t]
                    })
                    data_pdr_lod.append({
                        'lc': f"LC{l_i+1}",
                        'type': 'S$^2$LCT (SEED+EXP)',
                        'num_seed': ns,
                        'scores': pdr_cov_result[lc_desc]['ours_seed_exp'][f"{ns}sample"]['coverage_scores'][t]
                    })
                    data_pdr_lod.append({
                        'lc': f"LC{l_i+1}",
                        'type': 'CHECKLIST',
                        'num_seed': ns,
                        'scores': pdr_cov_result[lc_desc]['bl'][f"{ns}sample"]['coverage_scores'][t]
                    })
                    pdr_y_limit = max(
                        pdr_y_limit,
                        max(
                            pdr_cov_result[lc_desc]['ours_seed_exp'][f"{ns}sample"]['coverage_scores'][t],
                            pdr_cov_result[lc_desc]['ours_seed'][f"{ns}sample"]['coverage_scores'][t],
                            pdr_cov_result[lc_desc]['bl'][f"{ns}sample"]['coverage_scores'][t]
                        )
                    )
                # end for
                
                for ns in x_ticks:
                    data_sb_lod.append({
                        'lc': f"LC{l_i+1}",
                        'type': 'S$^2$LCT (SEED)',
                        'num_seed': ns,
                        'scores': float(selfbleu_result[lc_desc]['ours_seed'][f"{ns}sample"]['selfbleu_scores'][t])
                    })
                    data_sb_lod.append({
                        'lc': f"LC{l_i+1}",
                        'type': 'S$^2$LCT (SEED+EXP)',
                        'num_seed': ns,
                        'scores': float(selfbleu_result[lc_desc]['ours_seed_exp'][f"{ns}sample"]['selfbleu_scores'][t])
                    })
                    data_sb_lod.append({
                        'lc': f"LC{l_i+1}",
                        'type': 'CHECKLIST',
                        'num_seed': ns,
                        'scores': float(selfbleu_result[lc_desc]['bl'][f"{ns}sample"]['selfbleu_scores'][t])
                    })
                    sb_y_limit = max(
                        sb_y_limit,
                        max(
                            selfbleu_result[lc_desc]['ours_seed_exp'][f"{ns}sample"]['selfbleu_scores'][t],
                            selfbleu_result[lc_desc]['ours_seed'][f"{ns}sample"]['selfbleu_scores'][t],
                            selfbleu_result[lc_desc]['bl'][f"{ns}sample"]['selfbleu_scores'][t]
                        )
                    )
                # end for
            # end for

            df_pdr: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_pdr_lod))
            df_sb: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_sb_lod))
        
            # Plotting part
            fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False)
            # fig: plt.Figure = plt.figure()
            # ax: plt.Axes = fig.subplots()
            
            hue_order = ['CHECKLIST', 'S$^2$LCT (SEED)', 'S$^2$LCT (SEED+EXP)']
            
            from numpy import median
            ax_sb = sns.lineplot(data=df_sb, x='num_seed', y='scores',
                                 hue='type',
                                 hue_order=hue_order,
                                 estimator=median,
                                 style='type',
                                 err_style='bars',
                                 markers=['X', 's', 'o'],
                                 markersize=5,
                                 markeredgewidth=0,
                                 dashes=True,
                                 palette="Set1",
                                 err_kws={'capsize': 3},
                                 ax=ax1)
            
            ax_pdr = sns.lineplot(data=df_pdr, x='num_seed', y='scores',
                                  hue='type',
                                  hue_order=hue_order,
                                  estimator=median,
                                  style='type',
                                  err_style='bars',
                                  markers=['X', 's', 'o'],
                                  markersize=5,
                                  markeredgewidth=0,
                                  dashes=True,
                                  palette="Set1",
                                  err_kws={'capsize': 3},
                                  ax=ax2)
            # plt.xticks(x_ticks)
            ax_sb.set_xticks(x_ticks)
            sb_y_limit = sb_y_limit+0.5
            ax_sb.set_ylim(0.0, sb_y_limit)
            ax_sb.set_xlabel("Number of seeds")
            ax_sb.set_ylabel("Self-BLEU score")
            
            # Shrink current axis by 20%
            # box = ax_sb.get_position()
            # ax_sb.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            ax_pdr.set_xticks(pdr_x_ticks)
            pdr_y_limit = pdr_y_limit+200 if pdr_y_limit<1000 else pdr_y_limit+1000
            ax_pdr.set_ylim(-100, pdr_y_limit)
            ax_pdr.set_xlabel("Number of seeds")
            ax_pdr.set_ylabel("Number of Production Rules Covered")
            
            # Shrink current axis by 20%
            # box = ax_sb.get_position()
            # ax_pdr.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            
            # Put a legend to the right of the current axis
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
            fig.tight_layout()
            fig.savefig(figs_dir / f"pdr-selfbleu-agg-lc{l_i+1}-lineplot.eps")
        # end for
        return
        

    @classmethod
    def failrate_combined_over_seeds_plot(cls,
                                          results_dir: Path,
                                          figs_dir: Path,
                                          task=Macros.sa_task,
                                          search_dataset_name=Macros.datasets[Macros.sa_task][0],
                                          selection_method='random'):
        # num_seeds = [0,50,100,200] # x-axis
        num_trials = 3
        x_ticks = [50, 100, 150, 200]
        data_lod = dict()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        lc_index_dict = dict()
        hue_order, markers = list(), list()
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            lc_desc = l.strip().split('::')[0].lower()
            lc_index_dict[lc_desc] = f"LC{l_i+1}"
            marker = f"${l_i+1}$"
            hue_order.append(lc_index_dict[lc_desc])
            markers.append(marker)
        # end for

        for ns_i, ns in enumerate(x_ticks):
            for num_trial in range(num_trials):
                _num_trial = '' if num_trial==0 else str(num_trial+1)
                seed_file = Macros.result_dir / f"test_results{_num_trial}_{task}_{search_dataset_name}_{selection_method}_{ns}seeds" / "test_result_analysis.json"
                seed_dict = Utils.read_json(seed_file)
                for model_name, results_per_model in seed_dict.items():
                    if model_name not in data_lod.keys():
                        data_lod[model_name] = list()
                    # end if
                    for lc_i, lc_result in enumerate(results_per_model):
                        lc = lc_result['req']
                        lc_key = lc.lower()
                        lc_index = lc_index_dict[lc_key]
                        num_seeds = lc_result['num_seeds']
                        num_seed_fail = lc_result['num_seed_fail']
                        num_exps = lc_result['num_exps']
                        num_exp_fail = lc_result['num_exp_fail']
                        fr = (num_exp_fail+num_seed_fail)*1./(num_exps+num_seeds)
                        num_pass2fail = lc_result['num_pass2fail']
                        data_lod[model_name].append({
                            'lc': lc_index,
                            'num_seed': ns,
                            'failrate_all': fr,
                        })
                    # end for
                # end for
            # end for
        # end for

        for model_name, _data_lod in data_lod.items():
            # data_lod: List[dict] = list()
            _model_name = model_name.split('/')[-1]
            _data_lod = sorted(_data_lod, key=lambda x: int(x['lc'].split('LC')[-1]))
            df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(_data_lod))
            
            # Plotting part
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.subplots()
            ax = sns.lineplot(data=df, x='num_seed', y='failrate_all',
                              hue='lc',
                              hue_order=hue_order,
                              style='lc',
                              estimator='median',
                              err_style=None, # or "bars"
                              markers=markers,
                              markersize=9,
                              markeredgewidth=0,
                              dashes=True,
                              errorbar='sd',
                              ax=ax)
            plt.xticks(x_ticks)
            ax.set_ylim(-0.1, 1.2)
            ax.set_xlabel("Number of seeds")
            ax.set_ylabel("Failure rate on S$^2$LCT test cases")
            
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
            fig.savefig(figs_dir / f"failrate-combined-{_model_name}-lineplot.eps")
        # end for
        return
    
    @classmethod
    def failrate_seed_over_seeds_plot(cls,
                                      results_dir: Path,
                                      figs_dir: Path,
                                      task=Macros.sa_task,
                                      search_dataset_name=Macros.datasets[Macros.sa_task][0],
                                      selection_method='random'):
        # num_seeds = [0,50,100,200] # x-axis
        num_trials = 3
        x_ticks = [50, 100, 150, 200]
        data_lod = dict()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        lc_index_dict = {
            l.strip().split('::')[0].lower(): f"LC{l_i+1}"
            for l_i, l in enumerate(Utils.read_txt(req_file))
        }
        hue_order = [f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]

        for ns_i, ns in enumerate(x_ticks):
            for num_trial in range(num_trials):
                _num_trial = '' if num_trial==0 else str(num_trial+1)
                seed_file = Macros.result_dir / f"test_results{_num_trial}_{task}_{search_dataset_name}_{selection_method}_{ns}seeds" / "test_result_analysis.json"
                seed_dict = Utils.read_json(seed_file)
                for model_name, results_per_model in seed_dict.items():
                    if model_name not in data_lod.keys():
                        data_lod[model_name] = list()
                    # end if
                    for lc_i, lc_result in enumerate(results_per_model):
                        lc = lc_result['req']
                        lc_key = lc.lower()
                        lc_index = lc_index_dict[lc_key]
                        num_seeds = lc_result['num_seeds']
                        num_seed_fail = lc_result['num_seed_fail']
                        seed_fr = num_seed_fail*1./num_seeds
                        num_exps = lc_result['num_exps']
                        num_exp_fail = lc_result['num_exp_fail']
                        exp_fr = num_exp_fail*1./num_exps
                        num_pass2fail = lc_result['num_pass2fail']
                        data_lod[model_name].append({
                            'lc': lc_index,
                            'num_seed': ns,
                            'failrate_seed': seed_fr
                        })
                    # end for
                # end for
            # end for
        # end for

        for model_name, _data_lod in data_lod.items():
            # data_lod: List[dict] = list()
            _model_name = model_name.split('/')[-1]
            _data_lod = sorted(_data_lod, key=lambda x: int(x['lc'].split('LC')[-1]))
            df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(_data_lod))
            
            # Plotting part
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.subplots()
            
            ax = sns.lineplot(data=df, x='num_seed', y='failrate_seed',
                              hue='lc',
                              hue_order=hue_order,
                              style='lc',
                              estimator='median',
                              err_style=None, # or "bars"
                              markers=markers,
                              markersize=9,
                              markeredgewidth=0,
                              dashes=True,
                              errorbar='sd',
                              ax=ax)
            plt.xticks(x_ticks)
            ax.set_ylim(-0.1, 1.2)
            ax.set_xlabel("Number of seeds")
            ax.set_ylabel("Failure rate on S$^2$LCT seed test cases")
            
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
            fig.savefig(figs_dir / f"failrate-seed-{_model_name}-lineplot.eps")
        # end for
        return

    @classmethod
    def failrate_exp_over_seeds_plot(cls,
                                     results_dir: Path,
                                     figs_dir: Path,
                                     task=Macros.sa_task,
                                     search_dataset_name=Macros.datasets[Macros.sa_task][0],
                                     selection_method='random'):
        # num_seeds = [0,50,100,200] # x-axis
        num_trials = 3
        x_ticks = [50, 100, 150, 200]
        data_lod = dict()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        lc_index_dict = {
            l.strip().split('::')[0].lower(): f"LC{l_i+1}"
            for l_i, l in enumerate(Utils.read_txt(req_file))
        }
        hue_order = [f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        
        for ns_i, ns in enumerate(x_ticks):
            for num_trial in range(num_trials):
                _num_trial = '' if num_trial==0 else str(num_trial+1)
                seed_file = Macros.result_dir / f"test_results{_num_trial}_{task}_{search_dataset_name}_{selection_method}_{ns}seeds" / "test_result_analysis.json"
                seed_dict = Utils.read_json(seed_file)
                for model_name, results_per_model in seed_dict.items():
                    if model_name not in data_lod.keys():
                        data_lod[model_name] = list()
                    # end if
                    for lc_i, lc_result in enumerate(results_per_model):
                        lc = lc_result['req']
                        lc_key = lc.lower()
                        lc_index = lc_index_dict[lc_key]
                        num_seeds = lc_result['num_seeds']
                        num_seed_fail = lc_result['num_seed_fail']
                        seed_fr = num_seed_fail*1./num_seeds
                        num_exps = lc_result['num_exps']
                        num_exp_fail = lc_result['num_exp_fail']
                        exp_fr = num_exp_fail*1./num_exps
                        num_pass2fail = lc_result['num_pass2fail']
                        data_lod[model_name].append({
                            'lc': lc_index,
                            'num_seed': ns,
                            'failrate_exp': exp_fr
                        })
                    # end for
                # end for
            # end for
        # end for

        for model_name, _data_lod in data_lod.items():
            # data_lod: List[dict] = list()
            _model_name = model_name.split('/')[-1]
            _data_lod = sorted(_data_lod, key=lambda x: int(x['lc'].split('LC')[-1]))
            df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(_data_lod))
            
            # Plotting part
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.subplots()
            
            ax = sns.lineplot(data=df, x='num_seed', y='failrate_exp',
                              hue='lc',
                              hue_order=hue_order,
                              style='lc',
                              estimator='median',
                              err_style=None, # or "bars"
                              markers=markers,
                              markersize=9,
                              markeredgewidth=0,
                              dashes=True,
                              errorbar='sd',
                              ax=ax)
            plt.xticks(x_ticks)
            ax.set_ylim(-0.1, 1.2)
            ax.set_xlabel("Number of seeds")
            ax.set_ylabel("Failure rate on S$^2$LCT expanded test cases")
            
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
            fig.savefig(figs_dir / f"failrate-exp-{_model_name}-lineplot.eps")
        # end for
        return
    
    @classmethod
    def pass2fail_over_seeds_plot(cls,
                                  results_dir: Path,
                                  figs_dir: Path,
                                  task=Macros.sa_task,
                                  search_dataset_name=Macros.datasets[Macros.sa_task][0],
                                  selection_method='random'):
        # num_seeds = [0,50,100,200] # x-axis
        num_trials = 3
        x_ticks = [50, 100, 150, 200]
        data_lod = dict()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        lc_index_dict = {
            l.strip().split('::')[0].lower(): f"LC{l_i+1}"
            for l_i, l in enumerate(Utils.read_txt(req_file))
        }
        hue_order = [f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]

        for ns_i, ns in enumerate(x_ticks):
            for num_trial in range(num_trials):
                _num_trial = '' if num_trial==0 else str(num_trial+1)
                seed_file = Macros.result_dir / f"test_results{_num_trial}_{task}_{search_dataset_name}_{selection_method}_{ns}seeds" / "test_result_analysis.json"
                seed_dict = Utils.read_json(seed_file)
                for model_name, results_per_model in seed_dict.items():
                    if model_name not in data_lod.keys():
                        data_lod[model_name] = list()
                    # end if
                    for lc_i, lc_result in enumerate(results_per_model):
                        lc = lc_result['req']
                        lc_key = lc.lower()
                        lc_index = lc_index_dict[lc_key]
                        num_seeds = lc_result['num_seeds']
                        num_seed_fail = lc_result['num_seed_fail']
                        seed_fr = num_seed_fail*1./num_seeds
                        num_exps = lc_result['num_exps']
                        num_exp_fail = lc_result['num_exp_fail']
                        exp_fr = num_exp_fail*1./num_exps
                        num_pass2fail = lc_result['num_pass2fail']
                        data_lod[model_name].append({
                            'lc': lc_index,
                            'num_seed': ns,
                            'num_pass2fail': num_pass2fail
                        })
                    # end for
                # end for
            # end for
        # end for

        for model_name, _data_lod in data_lod.items():
            # data_lod: List[dict] = list()
            _model_name = model_name.split('/')[-1]
            _data_lod = sorted(_data_lod, key=lambda x: int(x['lc'].split('LC')[-1]))
            df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(_data_lod))
            
            # Plotting part
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.subplots()

            ax = sns.lineplot(data=df, x='num_seed', y='num_pass2fail',
                              hue='lc',
                              hue_order=hue_order,
                              style='lc',
                              estimator='median',
                              err_style=None, # or "bars"
                              markers=markers,
                              markersize=9,
                              markeredgewidth=0,
                              dashes=True,
                              errorbar='sd',
                              ax=ax)
            plt.xticks(x_ticks)
            ax.set_ylim(-20, 350)
            ax.set_xlabel("Number of seeds")
            ax.set_ylabel("Number of Pass2fail")
            
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
            fig.savefig(figs_dir / f"pass2fail-{_model_name}-lineplot.eps")
        # end for
        return

    @classmethod
    def numfail_over_seeds_plot(cls,
                                results_dir: Path,
                                figs_dir: Path,
                                task=Macros.sa_task,
                                search_dataset_name=Macros.datasets[Macros.sa_task][0],
                                selection_method='random'):
        # num_seeds = [0,50,100,200] # x-axis
        num_trials = 3
        x_ticks = [50, 100, 150, 200]
        data_lod = dict()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        lc_index_dict = {
            l.strip().split('::')[0].lower(): f"LC{l_i+1}"
            for l_i, l in enumerate(Utils.read_txt(req_file))
        }
        hue_order = [f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]

        for ns_i, ns in enumerate(x_ticks):
            for num_trial in range(num_trials):
                _num_trial = '' if num_trial==0 else str(num_trial+1)
                seed_file = Macros.result_dir / f"test_results{_num_trial}_{task}_{search_dataset_name}_{selection_method}_{ns}seeds" / "test_result_analysis.json"
                seed_dict = Utils.read_json(seed_file)
                for model_name, results_per_model in seed_dict.items():
                    if model_name not in data_lod.keys():
                        data_lod[model_name] = list()
                    # end if
                    for lc_i, lc_result in enumerate(results_per_model):
                        lc = lc_result['req']
                        lc_key = lc.lower()
                        lc_index = lc_index_dict[lc_key]
                        num_seeds = lc_result['num_seeds']
                        num_seed_fail = lc_result['num_seed_fail']
                        seed_fr = num_seed_fail*1./num_seeds
                        num_exps = lc_result['num_exps']
                        num_exp_fail = lc_result['num_exp_fail']
                        exp_fr = num_exp_fail*1./num_exps
                        num_pass2fail = lc_result['num_pass2fail']
                        data_lod[model_name].append({
                            'lc': lc_index,
                            'num_seed': ns,
                            'num_fail': num_seed_fail+num_exp_fail
                        })
                    # end for
                # end for
            # end for
        # end for

        for model_name, _data_lod in data_lod.items():
            # data_lod: List[dict] = list()
            _model_name = model_name.split('/')[-1]
            _data_lod = sorted(_data_lod, key=lambda x: int(x['lc'].split('LC')[-1]))
            df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(_data_lod))
            
            # Plotting part
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.subplots()

            ax = sns.lineplot(data=df, x='num_seed', y='num_fail',
                              hue='lc',
                              hue_order=hue_order,
                              style='lc',
                              estimator='median',
                              err_style=None, # or "bars"
                              markers=markers,
                              markersize=9,
                              markeredgewidth=0,
                              dashes=True,
                              errorbar='sd',
                              ax=ax)
            plt.xticks(x_ticks)
            ax.set_ylim(-500, 6000)
            ax.set_xlabel("Number of seeds")
            ax.set_ylabel("Number of fail cases")
            
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
            fig.savefig(figs_dir / f"numfail-{_model_name}-lineplot.eps")
        # end for
        return

    @classmethod
    def numfail_agg_over_seeds_plot(cls,
                                    results_dir: Path,
                                    figs_dir: Path,
                                    task=Macros.sa_task,
                                    search_dataset_name=Macros.datasets[Macros.sa_task][0],
                                    selection_method='random'):
        # num_seeds = [0,50,100,200] # x-axis
        num_trials = 3
        x_ticks = [50, 100, 150, 200]
        data_lod = list()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        lc_index_dict = {
            l.strip().split('::')[0].lower(): f"LC{l_i+1}"
            for l_i, l in enumerate(Utils.read_txt(req_file))
        }
        model_names = list()

        for ns_i, ns in enumerate(x_ticks):
            for num_trial in range(num_trials):
                _num_trial = '' if num_trial==0 else str(num_trial+1)
                seed_file = Macros.result_dir / f"test_results{_num_trial}_{task}_{search_dataset_name}_{selection_method}_{ns}seeds" / "test_result_analysis.json"
                seed_dict = Utils.read_json(seed_file)
                for model_name, results_per_model in seed_dict.items():
                    _model_name = model_name.split('/')[-1]
                    if _model_name.startswith('bert-base'):
                        model_names.append('BERT')
                        _model_name = 'BERT'
                    elif _model_name.startswith('roberta-base'):
                        model_names.append('RoBERTa')
                        _model_name = 'RoBERTa'
                    else:
                        model_names.append('DistilBERT')
                        _model_name = 'DistilBERT'
                    # end if
                    for lc_i, lc_result in enumerate(results_per_model):
                        lc = lc_result['req']
                        lc_key = lc.lower()
                        lc_index = lc_index_dict[lc_key]
                        num_seeds = lc_result['num_seeds']
                        num_seed_fail = lc_result['num_seed_fail']
                        seed_fr = num_seed_fail*1./num_seeds
                        num_exps = lc_result['num_exps']
                        num_exp_fail = lc_result['num_exp_fail']
                        exp_fr = num_exp_fail*1./num_exps
                        num_pass2fail = lc_result['num_pass2fail']
                        data_lod.append({
                            'lc': lc_index,
                            'model': _model_name,
                            'num_seed': ns,
                            'num_fail': num_seed_fail+num_exp_fail
                        })
                    # end for
                # end for
            # end for
        # end for

        # data_lod: List[dict] = list()
        # data_lod = sorted(_data_lod, key=lambda x: int(x['lc'].split('LC')[-1]))
        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        hue_order = model_names
        # markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        
        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        ax = sns.lineplot(data=df, x='num_seed', y='num_fail',
                          hue='model',
                          hue_order=hue_order,
                          style='model',
                          estimator='median',
                          err_style=None, # or "bars"
                          markers=True,
                          markersize=9,
                          markeredgewidth=0,
                          dashes=True,
                          palette='Set1',
                          errorbar='sd',
                          ax=ax)
        plt.xticks(x_ticks)
        ax.set_ylim(-500, 4000)
        ax.set_xlabel("Number of seeds")
        ax.set_ylabel("Number of fail cases")
            
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            
        # Put a legend to the right of the current axis
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
        fig.tight_layout()
        fig.savefig(figs_dir / f"numfail-agg-lineplot.eps")
        return

    @classmethod
    def pass2fail_agg_over_seeds_plot(cls,
                                      results_dir: Path,
                                      figs_dir: Path,
                                      task=Macros.sa_task,
                                      search_dataset_name=Macros.datasets[Macros.sa_task][0],
                                      selection_method='random'):
        # num_seeds = [0,50,100,200] # x-axis
        num_trials = 3
        x_ticks = [50, 100, 150, 200]
        data_lod = list()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        lc_index_dict = {
            l.strip().split('::')[0].lower(): f"LC{l_i+1}"
            for l_i, l in enumerate(Utils.read_txt(req_file))
        }
        model_names = list()

        for ns_i, ns in enumerate(x_ticks):
            for num_trial in range(num_trials):
                _num_trial = '' if num_trial==0 else str(num_trial+1)
                seed_file = Macros.result_dir / f"test_results{_num_trial}_{task}_{search_dataset_name}_{selection_method}_{ns}seeds" / "test_result_analysis.json"
                seed_dict = Utils.read_json(seed_file)
                for model_name, results_per_model in seed_dict.items():
                    _model_name = model_name.split('/')[-1]
                    if _model_name.startswith('bert-base'):
                        model_names.append('BERT')
                        _model_name = 'BERT'
                    elif _model_name.startswith('roberta-base'):
                        model_names.append('RoBERTa')
                        _model_name = 'RoBERTa'
                    else:
                        model_names.append('DistilBERT')
                        _model_name = 'DistilBERT'
                    # end if
                    for lc_i, lc_result in enumerate(results_per_model):
                        lc = lc_result['req']
                        lc_key = lc.lower()
                        lc_index = lc_index_dict[lc_key]
                        num_seeds = lc_result['num_seeds']
                        num_seed_fail = lc_result['num_seed_fail']
                        seed_fr = num_seed_fail*1./num_seeds
                        num_exps = lc_result['num_exps']
                        num_exp_fail = lc_result['num_exp_fail']
                        exp_fr = num_exp_fail*1./num_exps
                        num_pass2fail = lc_result['num_pass2fail']
                        data_lod.append({
                            'lc': lc_index,
                            'model': _model_name,
                            'num_seed': ns,
                            'num_pass2fail': num_pass2fail
                        })
                    # end for
                # end for
            # end for
        # end for

        # data_lod: List[dict] = list()
        # data_lod = sorted(_data_lod, key=lambda x: int(x['lc'].split('LC')[-1]))
        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        hue_order = model_names
        # markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        
        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        ax = sns.lineplot(data=df, x='num_seed', y='num_pass2fail',
                          hue='model',
                          hue_order=hue_order,
                          style='model',
                          estimator='median',
                          err_style=None, # or "bars"
                          markers=True,
                          markersize=9,
                          markeredgewidth=0,
                          palette="Set1",
                          dashes=True,
                          errorbar='sd',
                          ax=ax)
        plt.xticks(x_ticks)
        ax.set_ylim(0, 120)
        ax.set_xlabel("Number of seeds")
        ax.set_ylabel("Number of pass-to-fail cases")
            
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            
        # Put a legend to the right of the current axis
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
        fig.tight_layout()
        fig.savefig(figs_dir / f"pass2fail-agg-lineplot.eps")
        return

    @classmethod
    def numfail_pass2fail_agg_over_seeds_plot(cls,
                                              results_dir: Path,
                                              figs_dir: Path,
                                              task=Macros.sa_task,
                                              search_dataset_name=Macros.datasets[Macros.sa_task][0],
                                              selection_method='random'):
        # num_seeds = [0,50,100,150,200] # x-axis
        num_trials = 10
        x_ticks = [200, 400, 600, 800]
        data_lod_numfail = list()
        data_lod_pass2fail = list()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        p2f_result_file = Macros.result_dir / f"p2f_f2p_{task}_{search_dataset_name}_{selection_method}.json"
        fail_result_file = Macros.result_dir / f"failcases_{task}_{search_dataset_name}_{selection_method}.json"
        fail_bl_result_file = Macros.result_dir / f"failcases_bl_{task}_{search_dataset_name}_{selection_method}.json"
        p2f_result = Utils.read_json(p2f_result_file)
        fail_result = Utils.read_json(fail_result_file)
        fail_bl_result = Utils.read_json(fail_bl_result_file)
        model_names = list()
        for l_i, lc in enumerate(Utils.read_txt(req_file)):
            lc_desc = lc.strip().split('::')[0]
            data_lod_pass2fail: List[dict] = list()
            data_lod_numfail: List[dict] = list()
            p2f_y_limit = -1
            fail_y_limit = -1
            for model_name in p2f_result.keys():
                p2f_result_model = p2f_result[model_name]
                fail_result_model = fail_result[model_name]
                fail_bl_result_model = fail_bl_result[model_name]
                _model_name = model_name.split('/')[-1]
                lc_desc = lc_desc if lc_desc in p2f_result_model.keys() else lc_desc.lower()
                if _model_name.startswith('bert-base'):
                    model_names.append('BERT')
                    _model_name = 'BERT'
                elif _model_name.startswith('roberta-base'):
                    model_names.append('RoBERTa')
                    _model_name = 'RoBERTa'
                else:
                    model_names.append('DistilBERT')
                    _model_name = 'DistilBERT'
                # end if
                for ns in x_ticks:
                    sample_key = f"{ns}sample"
                    num_p2fs = list()
                    num_fails = list()
                    num_bl_fails = list()
                    for t in range(num_trials):
                        data_lod_pass2fail.append({
                            'model': _model_name,
                            'num_seed': ns,
                            'num_trial': t,
                            'num_pass2fail': p2f_result_model[lc_desc][sample_key]['p2f'][t]
                        })
                        data_lod_numfail.append({
                            'model': f"{_model_name} (S^2LCT)",
                            'num_seed': ns,
                            'num_trial': t,
                            'num_fail': fail_result_model[lc_desc][sample_key][t]
                        })
                        data_lod_numfail.append({
                            'model': f"{_model_name} (CHECKLIST)",
                            'num_seed': ns,
                            'num_trial': t,
                            'num_fail': fail_bl_result_model[lc_desc][sample_key][t]
                        })
                        p2f_y_limit = max(
                            p2f_y_limit,
                            p2f_result_model[lc_desc][sample_key]['p2f'][t]
                        )
                        fail_y_limit = max(
                            fail_y_limit,
                            max(
                                fail_result_model[lc_desc][sample_key][t],
                                fail_bl_result_model[lc_desc][sample_key][t]
                            )
                        )
                    # end for
                # end for
            # end for
            df_numfail: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod_numfail))
            df_pass2fail: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod_pass2fail))
        
            # Plotting part
            fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False)
            # fig: plt.Figure = plt.figure()
            # ax: plt.Axes = fig.subplots()
            
            from numpy import median
            hue_order = [f"{m} (S^2LCT)" for m in model_names]+[f"{m} (CHECKLIST)" for m in model_names]
            # markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]
            ax_numfail = sns.lineplot(data=df_numfail, x='num_seed', y='num_fail',
                                      hue='model',
                                      hue_order=hue_order,
                                      estimator=median,
                                      style='model',
                                      err_style="bars",
                                      markers=True,
                                      markersize=9,
                                      markeredgewidth=0,
                                      dashes=True,
                                      palette="Set1",
                                      err_kws={'capsize': 3},
                                      ax=ax1)
            hue_order = model_names
            ax_pass2fail = sns.lineplot(data=df_pass2fail, x='num_seed', y='num_pass2fail',
                                        hue='model',
                                        hue_order=hue_order,
                                        estimator=median,
                                        style='model',
                                        err_style="bars",
                                        markers=True,
                                        markersize=9,
                                        markeredgewidth=0,
                                        dashes=True,
                                        palette="Set1",
                                        err_kws={'capsize': 3},
                                        ax=ax2)
            plt.xticks(x_ticks)
            fail_y_limit = fail_y_limit+200 if fail_y_limit<1000 else fail_y_limit+1000
            ax_numfail.set_ylim(-10, fail_y_limit)
            ax_numfail.set_xlabel("Number of seeds")
            ax_numfail.set_ylabel("Number of fail cases")

            p2f_y_limit = p2f_y_limit+10 if p2f_y_limit<1000 else p2f_y_limit+1000
            ax_pass2fail.set_ylim(-1, p2f_y_limit)
            ax_pass2fail.set_xlabel("Number of seeds")
            ax_pass2fail.set_ylabel("Number of pass-to-fail cases")
            
            # Shrink current axis by 20%
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            
            # Put a legend to the right of the current axis
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
            fig.tight_layout()
            fig.savefig(figs_dir / f"numfail-pass2fail-agg-lc{l_i+1}-lineplot.eps")
        # end for
        return
