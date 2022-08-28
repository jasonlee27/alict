
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
            if item == "selfbleu":
                cls.selfbleu_ours_plot(Macros.result_dir, figs_dir)
                cls.selfbleu_bl_plot(Macros.result_dir, figs_dir)
            # elif item == "pdr":
            #     pass
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
    def selfbleu_ours_plot(cls, results_dir: Path, figs_dir: Path):
        data_lod: List[dict] = list()
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())

        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            lc_desc = l.strip().split('::')[0].lower()
            for ns in x_ticks.values():
                result_file = results_dir / 'selfbleu' / f"seeds_over3_sa_sst_random_{ns}seeds_selfbleu.json"
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

        ax = sns.lineplot(data=df, x="num_seed", y="scores",
                          hue="lc",
                          hue_order=hue_order,
                          style="lc",
                          err_style="bars", # or "band"
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
        fig.savefig(figs_dir / "selfbleu-ours-lineplot.eps")
        return

    @classmethod
    def selfbleu_bl_plot(cls, results_dir: Path, figs_dir: Path):
        data_lod: List[dict] = list()
        # num_seeds = [0,50,100,200] # x-axis
        x_ticks = {0:50, 1:100, 2:200}
        num_seeds = list(x_ticks.keys())

        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            lc_desc = l.strip().split('::')[0].lower()
            for ns in x_ticks.values():
                result_file = results_dir / 'selfbleu' / f"seeds_over3_sa_sst_random_{ns}seeds_selfbleu.json"
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

        ax = sns.lineplot(data=df, x="num_seed", y="scores",
                          hue="lc",
                          hue_order=hue_order,
                          style="lc",
                          err_style="bars", # or "bars"
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
        fig.savefig(figs_dir / "selfbleu-bl-lineplot.eps")
        return
