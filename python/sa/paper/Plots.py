
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

    logger = LoggingUtils.get_logger(__name__)

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

        
        which: Union[str, List[str]] = options.pop("which", [])
        paper_dir: Path = Path(options.pop("paper_dir", Macros.paper_dir))
        figs_dir: Path = paper_dir / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(which, list):
            which = [which]
        # end if

        for item in which:
            if item == "selfbleu":
                cls.selfbleu_ours_plot(Macros.results_dir, figs_dir)
                cls.selfbleu_bl_plot(Macros.results_dir, figs_dir)
            # elif item == "pdr":
            #     pass
            else:
                cls.logger.warning(f"Unknown plot {item}")
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
        num_seeds = [0,50,100,200] # x-axis

        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            lc_desc = l.strip().split('::')[0]
            for ns in num_seeds[1:]:
                result_file = results_dir / 'selfbleu' / f"seeds_over3_sa_sst_random_{ns}seeds_selfbleu.json"
                result = Utils.read_json(result_file)
                result_lc_ours = {
                    'lc': f"LC{li+1}",
                    'num_seed': ns,
                    'scores': result[lc_desc]['ours']['selfbleu_scores']
                    # 'avg': result[lc_desc]['ours']['avg_score'],
                    # 'med': result[lc_desc]['ours']['med_score'],
                    # 'std': result[lc_desc]['ours']['std_score']
                }
                data_lod.append(result_lc_ours)
                # result_lc_bl = {
                #     'lc': f"LC{li+1}",
                #     'num_seed': ns,
                #     'scores': result[lc_desc]['bl']['selfbleu_scores']
                #     # 'avg': result[lc_desc]['bl']['avg_score'],
                #     # 'med': result[lc_desc]['bl']['med_score'],
                #     # 'std': result[lc_desc]['bl']['std_score']
                # }
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
            err_style="band" # or "bars"
            markers=markers,
            markersize=6,
            markeredgewidth=0,
            dashes=True,
            ci=None,
            ax=ax,
        )

        ax.set_ylim(-0.1, 1.2)
        ax.set_xlabel("Number of seeds")
        ax.set_ylabel("Self-BLEU scores")
        fig.savefig(fig_dir / "selfbleu-ours-lineplot.eps")
        return

    @classmethod
    def selfbleu_bl_plot(cls, results_dir: Path, figs_dir: Path):
        data_lod: List[dict] = list()
        num_seeds = [0,50,100,200] # x-axis

        req_dir = results_dir / 'reqs'
        req_file = req_dir / 'requirements_desc.txt'
        for l_i, l in enumerate(Utils.read_txt(req_file)):
            lc_desc = l.strip().split('::')[0]
            for ns in num_seeds[1:]:
                result_file = results_dir / 'selfbleu' / f"seeds_over3_sa_sst_random_{ns}seeds_selfbleu.json"
                result = Utils.read_json(result_file)
                # result_lc_ours = {
                #     'lc': f"LC{li+1}",
                #     'num_seed': ns,
                #     'scores': result[lc_desc]['ours']['selfbleu_scores']
                #     # 'avg': result[lc_desc]['ours']['avg_score'],
                #     # 'med': result[lc_desc]['ours']['med_score'],
                #     # 'std': result[lc_desc]['ours']['std_score']
                # }
                # data_lod.append(result_lc_ours)
                result_lc_bl = {
                    'lc': f"LC{li+1}",
                    'num_seed': ns,
                    'scores': result[lc_desc]['bl']['selfbleu_scores']
                    # 'avg': result[lc_desc]['bl']['avg_score'],
                    # 'med': result[lc_desc]['bl']['med_score'],
                    # 'std': result[lc_desc]['bl']['std_score']
                }
                data_lod.append(result_lc_bl)
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
            err_style="band" # or "bars"
            markers=markers,
            markersize=6,
            markeredgewidth=0,
            dashes=True,
            ci=None,
            ax=ax,
        )

        ax.set_ylim(-0.1, 1.2)
        ax.set_xlabel("Number of seeds")
        ax.set_ylabel("Self-BLEU scores")
        fig.savefig(fig_dir / "selfbleu-bl-lineplot.eps")
        return
