
# This script is for generating all plots used in paper

from typing import *

import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

from seutil import IOUtils

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
        num_seeds = options.pop('num_seeds', 50)
        num_trials = options.pop('num_trials', 3)
        
        which: Union[str, List[str]] = options.pop("which", [])
        paper_dir: Path = Path(options.pop("paper_dir", Macros.paper_dir))
        figs_dir: Path = paper_dir / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(which, list):
            which = [which]
        # end if

        for item in which:
            if item == 'pdr-selfbleu-agg':
                cls.pdr_selfbleu_agg_plot(Macros.result_dir, figs_dir)
            elif item == 'selfbleu':
                cls.selfbleu_ours_sample_plot(Macros.result_dir, figs_dir)
                cls.selfbleu_bl_sample_plot(Macros.result_dir, figs_dir)
                cls.selfbleu_agg_sample_plot(Macros.result_dir, figs_dir)
            elif item == 'pdr':
                cls.pdr_ours_plot(Macros.result_dir, figs_dir)
                cls.pdr_ours_sample_plot(Macros.result_dir, figs_dir)
                cls.pdr_bl_sample_plot(Macros.result_dir, figs_dir)
                cls.pdr_seed_exp_plot(Macros.result_dir, figs_dir)
                cls.pdr_agg_plot(Macros.result_dir, figs_dir)
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
    def selfbleu_ours_sample_plot(cls,
                                  results_dir: Path,
                                  figs_dir: Path,
                                  task=Macros.sa_task,
                                  search_dataset_name=Macros.datasets[Macros.sa_task][0],
                                  selection_method='random'):
        data_lod: List[dict] = list()
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())

        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            lc_desc = l.strip().split('::')[0].lower()
            for ns in x_ticks.values():
                result_file = results_dir / 'selfbleu' / f"seeds_over3_{task}_{search_dataset_name}_{selection_method}_{ns}seeds_selfbleu.json"
                result = Utils.read_json(result_file)
                for s_i, score in enumerate(result[lc_desc]['ours']['selfbleu_scores']):
                    result_lc_ours = {
                        'sample': s_i,
                        'lc': f"LC{l_i+1}",
                        'num_seed': ns,
                        'scores': score
                        # 'avg': result[lc_desc]['ours']['avg_score'],
                        # 'med': result[lc_desc]['ours']['med_score'],
                        # 'std': result[lc_desc]['ours']['std_score']
                    }
                    data_lod.append(result_lc_ours)
                # end for
            # end for
        # end for

        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        hue_order = [f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]

        ax = sns.lineplot(data=df, x='num_seed', y='scores',
                          hue='lc',
                          hue_order=hue_order,
                          style='lc',
                          estimator='median',
                          err_style=None, # or "band"
                          markers=markers,
                          markersize=9,
                          markeredgewidth=0,
                          dashes=True,
                          ci='sd',
                          ax=ax)
        plt.xticks(list(x_ticks.values()))
        ax.set_ylim(0.0, 1.2)
        ax.set_xlabel("Number of seeds")
        ax.set_ylabel("Self-BLEU")
        
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
        # fig.tight_layout()
        fig.savefig(figs_dir / "selfbleu-ours-sample-lineplot.eps")
        return

    @classmethod
    def selfbleu_bl_sample_plot(cls,
                                results_dir: Path,
                                figs_dir: Path,
                                task=Macros.sa_task,
                                search_dataset_name=Macros.datasets[Macros.sa_task][0],
                                selection_method='random'):
        data_lod: List[dict] = list()
        # num_seeds = [0,50,100,200] # x-axis
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())

        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            lc_desc = l.strip().split('::')[0].lower()
            for ns in x_ticks.values():
                result_file = results_dir / 'selfbleu' / f"seeds_over3_{task}_{search_dataset_name}_{selection_method}_{ns}seeds_selfbleu.json"
                result = Utils.read_json(result_file)
                for score in result[lc_desc]['bl']['selfbleu_scores']:
                    result_lc_bl = {
                        'lc': f"LC{l_i+1}",
                        'num_seed': ns,
                        'scores': score
                        # 'avg': result[lc_desc]['ours']['avg_score'],
                        # 'med': result[lc_desc]['ours']['med_score'],
                        # 'std': result[lc_desc]['ours']['std_score']
                    }
                    data_lod.append(result_lc_bl)
                # end for
            # end for
        # end for

        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        hue_order = [f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]

        ax = sns.lineplot(data=df, x='num_seed', y='scores',
                          hue='lc',
                          hue_order=hue_order,
                          style='lc',
                          estimator='median',
                          err_style=None, # or "bars"
                          markers=markers,
                          markersize=9,
                          markeredgewidth=0,
                          dashes=True,
                          ci='sd',
                          ax=ax)
        plt.xticks(list(x_ticks.values()))
        ax.set_ylim(0.0, 1.2)
        ax.set_xlabel("Number of seeds")
        ax.set_ylabel("Self-BLEU")

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
        fig.savefig(figs_dir / "selfbleu-bl-sample-lineplot.eps")
        return

    @classmethod
    def selfbleu_agg_sample_plot(cls,
                                 results_dir: Path,
                                 figs_dir: Path,
                                 task=Macros.sa_task,
                                 search_dataset_name=Macros.datasets[Macros.sa_task][0],
                                 selection_method='random'):
        data_lod: List[dict] = list()
        # num_seeds = [0,50,100,200] # x-axis
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())
    
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for ns in x_ticks.values():
            for l_i, l in enumerate(Utils.read_txt(req_file)):
                lc_desc = l.strip().split('::')[0].lower()
                result_file = results_dir / 'selfbleu' / f"seeds_over3_{task}_{search_dataset_name}_{selection_method}_{ns}seeds_selfbleu.json"
                result = Utils.read_json(result_file)
                data_lod.append({
                    'lc': f"LC{l_i+1}",
                    'type': 'S$^2$LCT',
                    'num_seed': ns,
                    'scores': float(result[lc_desc]['ours']['avg_score'])
                })
                data_lod.append({
                    'lc': f"LC{l_i+1}",
                    'type': 'CHECKLIST',
                    'num_seed': ns,
                    'scores': float(result[lc_desc]['bl']['avg_score'])
                })
            # end for
        # end for
    

        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))
        
        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        # hue_order = [f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        # markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        hue_order = ['S$^2$LCT', 'CHECKLIST']
        markers = ['$1$', '$2$']

        ax = sns.lineplot(data=df, x='num_seed', y='scores',
                          hue='type',
                          hue_order=hue_order,
                          style='type',
                          err_style=None, # or "bars"
                          estimator='median',
                          markers=True,
                          markersize=9,
                          markeredgewidth=0,
                          dashes=True,
                          ci='sd',
                          ax=ax)
        plt.xticks(list(x_ticks.values()))
        ax.set_ylim(0.0, 1.2)
        ax.set_xlabel("Number of seeds")
        ax.set_ylabel("Self-BLEU")

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        # Put a legend to the right of the current axis
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))

        fig.tight_layout()
        fig.savefig(figs_dir / "selfbleu-agg-sample-lineplot.eps")
        return

    
    
    @classmethod
    def pdr_ours_sample_plot(cls,
                             results_dir: Path,
                             figs_dir: Path,
                             task=Macros.sa_task,
                             search_dataset_name=Macros.datasets[Macros.sa_task][0],
                             selection_method='random'):
        data_lod: List[dict] = list()
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())

        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            lc_desc = l.strip().split('::')[0].lower()
            for ns in x_ticks.values():
                result_file = results_dir / 'pdr_cov' / f"seeds_sample_over3_{task}_{search_dataset_name}_{selection_method}_{ns}seeds_pdrcov.json"
                result = Utils.read_json(result_file)
                for s_i, score in enumerate(result[lc_desc]['ours']['coverage_scores']):
                    result_lc_ours = {
                        'sample': s_i,
                        'lc': f"LC{l_i+1}",
                        'num_seed': ns,
                        'scores': score
                        # 'avg': result[lc_desc]['ours']['avg_score'],
                        # 'med': result[lc_desc]['ours']['med_score'],
                        # 'std': result[lc_desc]['ours']['std_score']
                    }
                    data_lod.append(result_lc_ours)
                # end for
            # end for
        # end for

        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        hue_order = [f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]

        ax = sns.lineplot(data=df, x='num_seed', y='scores',
                          hue='lc',
                          hue_order=hue_order,
                          style='lc',
                          estimator='median',
                          err_style=None, # or "band"
                          markers=markers,
                          markersize=9,
                          markeredgewidth=0,
                          dashes=True,
                          ci='sd',
                          ax=ax)
        plt.xticks(list(x_ticks.values()))
        ax.set_ylim(-50, 700)
        ax.set_xlabel("Number of seeds")
        ax.set_ylabel("Number of Production Rules Covered")
        
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
        # fig.tight_layout()
        fig.savefig(figs_dir / "pdr-ours-sample-lineplot.eps")
        return

    @classmethod
    def pdr_bl_sample_plot(cls,
                           results_dir: Path,
                           figs_dir: Path,
                           task=Macros.sa_task,
                           search_dataset_name=Macros.datasets[Macros.sa_task][0],
                           selection_method='random'):
        data_lod: List[dict] = list()
        # num_seeds = [0,50,100,200] # x-axis
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())

        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            lc_desc = l.strip().split('::')[0].lower()
            for ns in x_ticks.values():
                result_file = results_dir / 'pdr_cov' / f"seeds_sample_over3_{task}_{search_dataset_name}_{selection_method}_{ns}seeds_pdrcov.json"
                result = Utils.read_json(result_file)
                for score in result[lc_desc]['bl']['coverage_scores']:
                    result_lc_bl = {
                        'lc': f"LC{l_i+1}",
                        'num_seed': ns,
                        'scores': score
                        # 'avg': result[lc_desc]['ours']['avg_score'],
                        # 'med': result[lc_desc]['ours']['med_score'],
                        # 'std': result[lc_desc]['ours']['std_score']
                    }
                    data_lod.append(result_lc_bl)
                # end for
            # end for
        # end for

        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        hue_order = [f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]

        ax = sns.lineplot(data=df, x='num_seed', y='scores',
                          hue='lc',
                          hue_order=hue_order,
                          style='lc',
                          estimator='median',
                          err_style=None, # or "bars"
                          markers=markers,
                          markersize=9,
                          markeredgewidth=0,
                          dashes=True,
                          ci='sd',
                          ax=ax)
        plt.xticks(list(x_ticks.values()))
        ax.set_ylim(-50, 700)
        ax.set_xlabel("Number of seeds")
        ax.set_ylabel("Number of Production Rules Covered")

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
        fig.savefig(figs_dir / "pdr-bl-sample-lineplot.eps")
        return

    @classmethod
    def pdr_seed_exp_plot(cls,
                          results_dir: Path,
                          figs_dir: Path,
                          task=Macros.sa_task,
                          search_dataset_name=Macros.datasets[Macros.sa_task][0],
                          selection_method='random'):
        # data_lod: List[dict] = list()
        data_lod = {
            'ours-seed': list(),
            'ours-seed-exp': list(),
        }
        # num_seeds = [0,50,100,200] # x-axis
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            lc_desc = l.strip().split('::')[0].lower()
            for ns in x_ticks.values():
                result_file = results_dir / 'pdr_cov' / f"seeds_exps_over3_{task}_{search_dataset_name}_{selection_method}_{ns}seeds_pdrcov.json"
                result = Utils.read_json(result_file)
                for score in result[lc_desc]['ours_seed']['coverage_scores']:
                    data_lod['ours-seed'].append({
                        'lc': f"LC{l_i+1}",
                        'num_seed': ns,
                        'scores': score
                        # 'avg': result[lc_desc]['ours']['avg_score'],
                        # 'med': result[lc_desc]['ours']['med_score'],
                        # 'std': result[lc_desc]['ours']['std_score']
                    })
                # end for
                for score in result[lc_desc]['ours_seed_exp']['coverage_scores']:
                    data_lod['ours-seed-exp'].append({
                        'lc': f"LC{l_i+1}",
                        'num_seed': ns,
                        'scores': score
                        # 'avg': result[lc_desc]['ours']['avg_score'],
                        # 'med': result[lc_desc]['ours']['med_score'],
                        # 'std': result[lc_desc]['ours']['std_score']
                    })
                # end for
            # end for
        # end for

        for data_type, _data_lod in data_lod.items():

            df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(_data_lod))

            # Plotting part
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.subplots()
            
            hue_order = [f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))]
            markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]
            
            ax = sns.lineplot(data=df, x='num_seed', y='scores',
                              hue='lc',
                              hue_order=hue_order,
                              style='lc',
                              estimator='median',
                              err_style=None, # or "bars"
                              markers=markers,
                              markersize=9,
                              markeredgewidth=0,
                              dashes=True,
                              ci='sd',
                              ax=ax)
            plt.xticks(list(x_ticks.values()))
            ax.set_ylim(0, 850)
            ax.set_xlabel("Number of seeds")
            ax.set_ylabel("Number of Production Rules Covered")
            
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
            fig.savefig(figs_dir / f"pdr-{data_type}-lineplot.eps")
        # end if
        return

    @classmethod
    def pdr_ours_plot(cls,
                      results_dir: Path,
                      figs_dir: Path,
                      task=Macros.sa_task,
                      search_dataset_name=Macros.datasets[Macros.sa_task][0],
                      selection_method='random'):
        data_lod: List[dict] = list()
        x_ticks = {0:50} # , 1:100, 2:200}
        num_seeds = list(x_ticks.keys())

        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            lc_desc = l.strip().split('::')[0].lower()
            for ns in x_ticks.values():
                result_file = results_dir / 'pdr_cov' / f"seeds_over3_{task}_{search_dataset_name}_{selection_method}_{ns}seeds_pdrcov.json"
                result = Utils.read_json(result_file)
                bl_score = result[lc_desc]['bl']['coverage_scores'][0]
                data_lod.append({
                    'lc': f"LC{l_i+1}",
                    'type': 'CHECKLIST',
                    'num_seed': ns,
                    'scores': bl_score
                    # 'avg': result[lc_desc]['ours']['avg_score'],
                    # 'med': result[lc_desc]['ours']['med_score'],
                    # 'std': result[lc_desc]['ours']['std_score']
                })
                print(l_i+1, bl_score, result[lc_desc]['ours']['med_score'], float(result[lc_desc]['ours']['med_score'])/bl_score)
                for score in result[lc_desc]['ours']['coverage_scores']:
                    result_lc_ours = {
                        'lc': f"LC{l_i+1}",
                        'type': 'S$^2$LCT',
                        'num_seed': ns,
                        'scores': score
                        # 'avg': result[lc_desc]['ours']['avg_score'],
                        # 'med': result[lc_desc]['ours']['med_score'],
                        # 'std': result[lc_desc]['ours']['std_score']
                    }
                    data_lod.append(result_lc_ours)
                # end for
            # end for
        # end for

        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        hue_order = ['CHECKLIST', 'S$^2$LCT']
        
        from numpy import median
        ax = sns.barplot(data=df, x='lc', y='scores',
                         hue='type',
                         hue_order=hue_order,
                         estimator=median)
        # plt.xticks([f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))])
        ax.set_ylim(bottom=0, top=600)
        ax.set_xlabel("Linguistic Capabilities")
        ax.set_ylabel("Number of Production Rules Covered")
        
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Put a legend to the right of the current axis
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
        fig.tight_layout()
        fig.savefig(figs_dir / "pdr-ours-50seeds-barplot.eps")
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
    #                       ci='sd',
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
        data_pdr_lod: List[dict] = list()
        # num_seeds = [0,50,100,200] # x-axis
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        
        for ns in x_ticks.values():
            for l_i, l in enumerate(Utils.read_txt(req_file)):
                lc_desc = l.strip().split('::')[0].lower()
                result_file = results_dir / 'pdr_cov' / f"seeds_exps_over3_{task}_{search_dataset_name}_{selection_method}_{ns}seeds_pdrcov.json"
                result = Utils.read_json(result_file)
                data_pdr_lod.append({
                    'lc': f"LC{l_i+1}",
                    'type': 'S$^2$LCT(seed)',
                    'num_seed': ns,
                    'scores': float(result[lc_desc]['ours_seed']['med_score'])
                })
                data_pdr_lod.append({
                    'lc': f"LC{l_i+1}",
                    'type': 'S$^2$LCT(seed+exp)',
                    'num_seed': ns,
                    'scores': float(result[lc_desc]['ours_seed_exp']['med_score'])
                })

                result_bl_file = results_dir / 'pdr_cov' / f"seeds_sample_over3_{task}_{search_dataset_name}_{selection_method}_{ns}seeds_pdrcov.json"
                result_bl = Utils.read_json(result_bl_file)
                data_pdr_lod.append({
                    'lc': f"LC{l_i+1}",
                    'type': 'CHECKLIST', 
                    'num_seed': ns,
                    'scores': float(result_bl[lc_desc]['bl']['med_score'])
                })
            # end for
        # end for

        data_sb_lod: List[dict] = list()
        for ns in x_ticks.values():
            for l_i, l in enumerate(Utils.read_txt(req_file)):
                lc_desc = l.strip().split('::')[0].lower()
                result_file = results_dir / 'selfbleu' / f"seeds_over3_{task}_{search_dataset_name}_{selection_method}_{ns}seeds_selfbleu.json"
                result = Utils.read_json(result_file)
                data_sb_lod.append({
                    'lc': f"LC{l_i+1}",
                    'type': 'S$^2$LCT',
                    'num_seed': ns,
                    'scores': float(result[lc_desc]['ours']['avg_score'])
                })
                data_sb_lod.append({
                    'lc': f"LC{l_i+1}",
                    'type': 'CHECKLIST',
                    'num_seed': ns,
                    'scores': float(result[lc_desc]['bl']['avg_score'])
                })
            # end for
        # end for

        df_pdr: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_pdr_lod))
        df_sb: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_sb_lod))
        
        # Plotting part
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False)
        # fig: plt.Figure = plt.figure()
        # ax: plt.Axes = fig.subplots()

        hue_order_sb = ['S$^2$LCT', 'CHECKLIST']
        hue_order_pdr = ['S$^2$LCT(seed)', 'S$^2$LCT(seed+exp)', 'CHECKLIST']
        
        ax_sb = sns.lineplot(data=df_sb, x='num_seed', y='scores',
                             hue='type',
                             hue_order=hue_order_sb,
                             estimator='median',
                             style='type',
                             err_style=None, # or "bars"
                             markers=True,
                             markersize=9,
                             markeredgewidth=0,
                             dashes=True,
                             ci='sd',
                             ax=ax1)
        ax_pdr = sns.lineplot(data=df_pdr, x='num_seed', y='scores',
                              hue='type',
                              hue_order=hue_order_pdr,
                              estimator='median',
                              style='type',
                              err_style=None, # or "bars"
                              markers=True,
                              markersize=9,
                              markeredgewidth=0,
                              dashes=True,
                              ci='sd',
                              ax=ax2)
        plt.xticks(list(x_ticks.values()))
        ax_sb.set_ylim(0.0, 1.2)
        ax_sb.set_xlabel("Number of seeds")
        ax_sb.set_ylabel("Self-BLEU score")

        # Shrink current axis by 20%
        # box = ax_sb.get_position()
        # ax_sb.set_position([box.x0, box.y0, box.width * 0.9, box.height])
                
        ax_pdr.set_ylim(0, 850)
        ax_pdr.set_xlabel("Number of seeds")
        ax_pdr.set_ylabel("Number of Production Rules Covered")

        # Shrink current axis by 20%
        # box = ax_sb.get_position()
        # ax_pdr.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        
        # Put a legend to the right of the current axis
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
        fig.tight_layout()
        fig.savefig(figs_dir / "pdr-selfbleu-agg-lineplot.eps")
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
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())
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

        for ns_i, ns in x_ticks.items():
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
                              ci='sd',
                              ax=ax)
            plt.xticks(list(x_ticks.values()))
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
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())
        data_lod = dict()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        lc_index_dict = {
            l.strip().split('::')[0].lower(): f"LC{l_i+1}"
            for l_i, l in enumerate(Utils.read_txt(req_file))
        }
        hue_order = [f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]

        for ns_i, ns in x_ticks.items():
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
                              ci='sd',
                              ax=ax)
            plt.xticks(list(x_ticks.values()))
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
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())
        data_lod = dict()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        lc_index_dict = {
            l.strip().split('::')[0].lower(): f"LC{l_i+1}"
            for l_i, l in enumerate(Utils.read_txt(req_file))
        }
        hue_order = [f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        
        for ns_i, ns in x_ticks.items():
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
                              ci='sd',
                              ax=ax)
            plt.xticks(list(x_ticks.values()))
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
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())
        data_lod = dict()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        lc_index_dict = {
            l.strip().split('::')[0].lower(): f"LC{l_i+1}"
            for l_i, l in enumerate(Utils.read_txt(req_file))
        }
        hue_order = [f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]

        for ns_i, ns in x_ticks.items():
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
                              ci='sd',
                              ax=ax)
            plt.xticks(list(x_ticks.values()))
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
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())
        data_lod = dict()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        lc_index_dict = {
            l.strip().split('::')[0].lower(): f"LC{l_i+1}"
            for l_i, l in enumerate(Utils.read_txt(req_file))
        }
        hue_order = [f"LC{l_i+1}" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]

        for ns_i, ns in x_ticks.items():
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
                              ci='sd',
                              ax=ax)
            plt.xticks(list(x_ticks.values()))
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
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())
        data_lod = list()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        lc_index_dict = {
            l.strip().split('::')[0].lower(): f"LC{l_i+1}"
            for l_i, l in enumerate(Utils.read_txt(req_file))
        }
        model_names = list()

        for ns_i, ns in x_ticks.items():
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
                          ci='sd',
                          ax=ax)
        plt.xticks(list(x_ticks.values()))
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
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())
        data_lod = list()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        lc_index_dict = {
            l.strip().split('::')[0].lower(): f"LC{l_i+1}"
            for l_i, l in enumerate(Utils.read_txt(req_file))
        }
        model_names = list()

        for ns_i, ns in x_ticks.items():
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
                          ci='sd',
                          ax=ax)
        plt.xticks(list(x_ticks.values()))
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
        # num_seeds = [0,50,100,200] # x-axis
        num_trials = 3
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())
        data_lod_numfail = list()
        data_lod_pass2fail = list()
        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        lc_index_dict = {
            l.strip().split('::')[0].lower(): f"LC{l_i+1}"
            for l_i, l in enumerate(Utils.read_txt(req_file))
        }
        model_names = list()

        for ns_i, ns in x_ticks.items():
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
                        data_lod_numfail.append({
                            'lc': lc_index,
                            'model': _model_name,
                            'num_seed': ns,
                            'num_fail': num_seed_fail+num_exp_fail
                        })
                        data_lod_pass2fail.append({
                            'lc': lc_index,
                            'model': _model_name,
                            'num_seed': ns,
                            'num_pass2fail': num_pass2fail
                        })
                    # end for
                # end for
            # end for
        # end for

        df_numfail: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod_numfail))
        df_pass2fail: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod_pass2fail))
        
        # Plotting part
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False)
        # fig: plt.Figure = plt.figure()
        # ax: plt.Axes = fig.subplots()

        hue_order = model_names
        # markers = [f"${l_i+1}$" for l_i, _ in enumerate(Utils.read_txt(req_file))]
        
        ax_numfail = sns.lineplot(data=df_numfail, x='num_seed', y='num_fail',
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
                                  ci='sd',
                                  ax=ax1)
        ax_pass2fail = sns.lineplot(data=df_pass2fail, x='num_seed', y='num_pass2fail',
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
                                    ci='sd',
                                    ax=ax2)
        plt.xticks(list(x_ticks.values()))
        ax_numfail.set_ylim(0, 3500)
        ax_numfail.set_xlabel("Number of seeds")
        ax_numfail.set_ylabel("Number of fail cases")

        ax_pass2fail.set_ylim(0, 100)
        ax_pass2fail.set_xlabel("Number of seeds")
        ax_pass2fail.set_ylabel("Number of pass-to-fail cases")
            
        # Shrink current axis by 20%
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            
        # Put a legend to the right of the current axis
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
        fig.tight_layout()
        fig.savefig(figs_dir / f"numfail-pass2fail-agg-lineplot.eps")
        return