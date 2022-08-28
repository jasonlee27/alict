
# This script is for generating all plots used in paper

from typing import *

import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

from seutil import IOUtils, LoggingUtils

from hdlp.Macros import Macros
from hdlp.Utils import Utils


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
            if item == "ce-all-projects":
                cls.cross_entropy_all_projects(Macros.results_dir, figs_dir)
            elif item == "ce-in-project":
                cls.cross_entropy_in_project(Macros.results_dir, Macros.data_dir, figs_dir)
            elif item == "perplexity-all-projects":
                cls.perplexity_all_projects(Macros.results_dir, figs_dir)
            elif item == "perplexity-in-project":
                cls.perplexity_in_project(Macros.results_dir, Macros.data_dir, figs_dir)
            elif item == "num-tokens":
                cls.num_tokens(Macros.results_dir, figs_dir)
                cls.num_unique_tokens(Macros.results_dir, figs_dir)
            elif item == "loc":
                cls.loc(Macros.results_dir, figs_dir)
            elif item == "lhs-var-type-dist":
                cls.lhs_var_type_dist(Macros.results_dir, figs_dir)
            elif item == "agn-var-type-dist":
                cls.agn_var_type_dist(Macros.results_dir, figs_dir)
            elif item == "ngram-rhs-completion-bleu":
                cls.ngram_bleu_plot(Macros.results_dir, figs_dir)
            elif item == "pcsnaming-models":
                #DELETE
                cls.pcsmodels_bleu_plot(Macros.results_dir, figs_dir)
                cls.pcsmodels_accuracy_plot(Macros.results_dir, figs_dir)

            else:
                cls.logger.warning(f"Unknown plot {item}")
            # end if
        # end for

        return

    LANG_NAMES = {
        "vhdl": "VHDL",
        "verilog": "Verilog",
        "systemverilog": "SystemVerilog",
        "java": "Java (Popular)",
        "javasmall": "Java (Naturalness)"
    }

    LANG_NAMES_SHORT = {
        "vhdl": "VHDL",
        "verilog": "Verilog",
        "systemverilog": "SystemVerilog",
        "java": "Java\n(Popular)",
        "javasmall": "Java\n(Naturalness)"
    }

    LANG_NAMES_LETTER = {
        "vhdl": "H",
        "verilog": "V",
        "systemverilog": "S",
        "java": "J",
        "javasmall": "A"
    }
    pcsnaming_models = ["bl_s2s", "bl_s2s_attn"] #DELETE

    @classmethod
    def cross_entropy_all_projects(cls, results_dir: Path, figs_dir: Path):
        data_lod: List[dict] = list()
        orders = list(range(1, 11, 1))
        folds = list(range(10))
        for lang in Macros.all_lang:
            ce_results_dir = results_dir / lang / "ALL" / "ce"

            for order_i, order in enumerate(orders):
                entropy_list = IOUtils.load(ce_results_dir / f"order-{order}.json", IOUtils.Format.json)
                for fold_i, fold in enumerate(folds):
                    entropy = entropy_list[fold_i]
                    data_lod.append({
                        "lang": lang,
                        "order": order,
                        "fold": fold,
                        "entropy": entropy,
                    })
                # end for
            # end for
        # end for

        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        hue_order = list(cls.LANG_NAMES.keys())
        markers = ["$H$", "$V$", "$S$", "$J$", "$A$"]

        ax = sns.lineplot(data=df, x="order", y="entropy",
            hue="lang",
            hue_order=hue_order,
            style="lang",
            markers=markers,
            markersize=6,
            markeredgewidth=0,
            dashes=True,
            ci=None,
            ax=ax,
        )

        ax.set_ylim(1.9, 9.2)
        ax.set_xlabel("n")
        ax.set_ylabel("Cross Entropy")
        for t in ax.get_legend().get_texts():
            if t.get_text() == "lang":
                t.set_text("Language")
            else:
                t.set_text(cls.LANG_NAMES[t.get_text()])
            # end if
        # end for

        with IOUtils.cd(figs_dir):
            fig.savefig("ce-all-projects-lineplot.eps")
        # end with

        return

    @classmethod
    def cross_entropy_in_project(cls, results_dir: Path, data_dir: Path, figs_dir: Path):
        data_lod: List[dict] = list()
        orders = list(range(1, 11, 1))
        folds = list(range(10))
        for lang in Macros.all_lang:
            # Get repos
            repositories_file = data_dir / lang / "repositories.txt"
            repositories = IOUtils.load(repositories_file, IOUtils.Format.txt).strip().splitlines()
            urls = [l.split()[0] for l in repositories]

            repo_full_names = []
            for url in urls:
                m = Utils.RE_GITHUB_URL.fullmatch(url)
                if m is None:  raise Exception(f"Malformed URL: {url}")
                repo_full_names.append(m.group("user")+"_"+m.group("repo"))
            # end for

            # Get ce results for each repo
            for repo_full_name in repo_full_names:
                ce_results_dir = results_dir / lang / repo_full_name / "ce"
                if not ce_results_dir.is_dir():
                    # Bad repo -- no data
                    cls.logger.warning(f"lang {lang}, repo {repo_full_name} do not have results, skipping")
                    continue
                # end if

                for order_i, order in enumerate(orders):
                    entropy_list = IOUtils.load(ce_results_dir / f"order-{order}.json", IOUtils.Format.json)
                    for fold_i, fold in enumerate(folds):
                        entropy = entropy_list[fold_i]
                        data_lod.append({
                            "lang": lang,
                            "repo": repo_full_name,
                            "order": order,
                            "fold": fold,
                            "entropy": entropy,
                        })
                    # end for
                # end for
            # end for

        # end for

        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        hue_order = list(cls.LANG_NAMES.keys())
        markers = ["$H$", "$V$", "$S$", "$J$", "$A$"]

        ax = sns.lineplot(data=df, x="order", y="entropy",
            hue="lang",
            hue_order=hue_order,
            style="lang",
            markers=markers,
            markersize=6,
            markeredgewidth=0,
            dashes=True,
            ci=None,
            ax=ax,
        )

        ax.set_ylim(1.9, 9.2)
        ax.set_xlabel("n")
        ax.set_ylabel("Cross Entropy")
        for t in ax.get_legend().get_texts():
            if t.get_text() == "lang":
                t.set_text("Language")
            else:
                t.set_text(cls.LANG_NAMES[t.get_text()])
            # end if
        # end for

        with IOUtils.cd(figs_dir):
            fig.savefig("ce-in-project-lineplot.eps")
        # end with

        return

    @classmethod
    def perplexity_all_projects(cls, results_dir: Path, figs_dir: Path):
        data_lod: List[dict] = list()
        orders = list(range(1, 11, 1))
        folds = list(range(10))
        for lang in Macros.all_lang:
            ce_results_dir = results_dir / lang / "ALL" / "ce"

            for order_i, order in enumerate(orders):
                entropy_list = IOUtils.load(ce_results_dir / f"order-{order}.json", IOUtils.Format.json)
                for fold_i, fold in enumerate(folds):
                    perplexity = 2**entropy_list[fold_i]
                    data_lod.append({
                        "lang": lang,
                        "order": order,
                        "fold": fold,
                        "perplexity": perplexity,
                    })
                # end for
            # end for
        # end for

        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        hue_order = list(cls.LANG_NAMES.keys())
        markers = ["$H$", "$V$", "$S$", "$J$", "$A$"]

        ax = sns.lineplot(data=df, x="order", y="perplexity",
            hue="lang",
            hue_order=hue_order,
            style="lang",
            markers=markers,
            markersize=6,
            markeredgewidth=0,
            dashes=True,
            ci=None,
            ax=ax,
        )

        ax.set_ylim(0, 550)
        ax.set_xlabel("n")
        ax.set_ylabel("Perplexity")
        for t in ax.get_legend().get_texts():
            if t.get_text() == "lang":
                t.set_text("Language")
            else:
                t.set_text(cls.LANG_NAMES[t.get_text()])
            # end if
        # end for

        with IOUtils.cd(figs_dir):
            fig.savefig("perplexity-all-projects-lineplot.eps")
        # end with

        return

    @classmethod
    def perplexity_in_project(cls, results_dir: Path, data_dir: Path, figs_dir: Path):
        data_lod: List[dict] = list()
        orders = list(range(1, 11, 1))
        folds = list(range(10))
        for lang in Macros.all_lang:
            # Get repos
            repositories_file = data_dir / lang / "repositories.txt"
            repositories = IOUtils.load(repositories_file, IOUtils.Format.txt).strip().splitlines()
            urls = [l.split()[0] for l in repositories]

            repo_full_names = []
            for url in urls:
                m = Utils.RE_GITHUB_URL.fullmatch(url)
                if m is None:  raise Exception(f"Malformed URL: {url}")
                repo_full_names.append(m.group("user")+"_"+m.group("repo"))
            # end for

            # Get ce results for each repo
            for repo_full_name in repo_full_names:
                ce_results_dir = results_dir / lang / repo_full_name / "ce"
                if not ce_results_dir.is_dir():
                    # Bad repo -- no data
                    cls.logger.warning(f"lang {lang}, repo {repo_full_name} do not have results, skipping")
                    continue
                # end if

                for order_i, order in enumerate(orders):
                    entropy_list = IOUtils.load(ce_results_dir / f"order-{order}.json", IOUtils.Format.json)
                    for fold_i, fold in enumerate(folds):
                        perplexity = 2**entropy_list[fold_i]
                        data_lod.append({
                            "lang": lang,
                            "repo": repo_full_name,
                            "order": order,
                            "fold": fold,
                            "perplexity": perplexity,
                        })
                    # end for
                # end for
            # end for

        # end for

        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        hue_order = list(cls.LANG_NAMES.keys())
        markers = ["$H$", "$V$", "$S$", "$J$", "$A$"]

        ax = sns.lineplot(data=df, x="order", y="perplexity",
            hue="lang",
            hue_order=hue_order,
            style="lang",
            markers=markers,
            markersize=6,
            markeredgewidth=0,
            dashes=True,
            ci=None,
            ax=ax,
        )

        ax.set_ylim(0, 250)
        ax.set_xlabel("n")
        ax.set_ylabel("Perplexity")
        for t in ax.get_legend().get_texts():
            if t.get_text() == "lang":
                t.set_text("Language")
            else:
                t.set_text(cls.LANG_NAMES[t.get_text()])
            # end if
        # end for

        with IOUtils.cd(figs_dir):
            fig.savefig("perplexity-in-project-lineplot.eps")
        # end with

        return

    #DELETE
    @classmethod
    def pcsmodels_bleu_plot(cls, results_dir: Path, figs_dir: Path):
        pcsnaming_results_dir = results_dir / "vhdl" / "ALL" / "pcsnaming"
        combined_dict = dict()
        for model in cls.pcsnaming_models:
            model_stats_file = pcsnaming_results_dir / f"{model}_stats.json"
            model_data = IOUtils.load(model_stats_file, IOUtils.Format.json)
            combined_dict[model] = model_data["bleu-data"]
            # end for
        # end for

        # Plotting part
        # Box plot
        fig: plt.Figure = plt.figure(figsize=(5, 5))
        ax: plt.Axes = fig.subplots()

        ax.boxplot(combined_dict.values())
        ax.set_ylabel("BLEU Score")
        ax.set_xticklabels(combined_dict.keys())
        fig.tight_layout()

        with IOUtils.cd(figs_dir):
            fig.savefig("pcsnaming-bleu-boxplot.eps")
        # end with
        return

    #DELETE
    @classmethod
    def pcsmodels_accuracy_plot(cls, results_dir: Path, figs_dir: Path):
        pcsnaming_results_dir = results_dir / "vhdl" / "ALL" / "pcsnaming"
        combined_dict = dict()
        for model in cls.pcsnaming_models:
            model_stats_file = pcsnaming_results_dir / f"{model}_stats.json"
            model_data = IOUtils.load(model_stats_file, IOUtils.Format.json)
            combined_dict[model] = model_data["acc-data"]
            # end for
        # end for

        # Plotting part
        # Box plot
        fig: plt.Figure = plt.figure(figsize=(5, 5))
        ax: plt.Axes = fig.subplots()

        ax.boxplot(combined_dict.values())
        ax.set_ylabel("Accuracy")
        ax.set_xticklabels(combined_dict.keys())
        fig.tight_layout()

        with IOUtils.cd(figs_dir):
            fig.savefig("pcsnaming-acc-boxplot.eps")
        # end with
        return

    
    @classmethod
    def num_tokens(cls, results_dir: Path, figs_dir: Path):
        data_lod: List[dict] = list()

        for lang in Macros.all_lang:
            num_tokens_file = results_dir / lang / "ALL" / "metrics" / "num-tokens-per-file.txt"
            num_tokens_list = [int(l) for l in IOUtils.load(num_tokens_file, IOUtils.Format.txt).splitlines() if l != ""]
            for num_token in num_tokens_list:
                data_lod.append({
                    "lang": lang,
                    "num_token": num_token,
                })
            # end for
        # end for

        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        # Plotting part
        # Box plot
        fig: plt.Figure = plt.figure(figsize=(5, 5))
        ax: plt.Axes = fig.subplots()

        ax = sns.boxplot(data=df, x="lang", y="num_token",
            width=0.6,
            linewidth=0.4,
            ax=ax,
        )

        ax.set_ylim(-2100/29, 2100)

        # ax.set_xlabel("Language Corpus")
        ax.set_xlabel(None)
        ax.set_ylabel("#Tokens")
        ax.set_xticklabels([cls.LANG_NAMES[t.get_text()] for t in ax.get_xticklabels()], rotation=45, ha="right", rotation_mode="anchor")
        fig.tight_layout()

        with IOUtils.cd(figs_dir):
            fig.savefig("num-tokens-boxplot.eps")
        # end with
        return

    @classmethod
    def num_unique_tokens(cls, results_dir: Path, figs_dir: Path):
        # num-unique-tokens
        data_lod: List[dict] = list()

        for lang in Macros.all_lang:
            num_unique_tokens_file = results_dir / lang / "ALL" / "metrics" / "num-unique-tokens-per-file.txt"
            num_unique_tokens_list = [int(l) for l in IOUtils.load(num_unique_tokens_file, IOUtils.Format.txt).splitlines() if l != ""]
            for num_unique_token in num_unique_tokens_list:
                data_lod.append({
                    "lang": lang,
                    "num_unique_token": num_unique_token,
                })
            # end for
        # end for

        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        # Plotting part
        # Box plot
        fig: plt.Figure = plt.figure(figsize=(5, 5))
        ax: plt.Axes = fig.subplots()

        ax = sns.boxplot(data=df, x="lang", y="num_unique_token",
            width=0.6,
            linewidth=0.4,
            ax=ax,
        )

        ax.set_ylim(-260/29, 260)

        # ax.set_xlabel("Language Corpus")
        ax.set_xlabel(None)
        ax.set_ylabel("#Unique Tokens")
        ax.set_xticklabels([cls.LANG_NAMES[t.get_text()] for t in ax.get_xticklabels()], rotation=45, ha="right", rotation_mode="anchor")
        fig.tight_layout()

        with IOUtils.cd(figs_dir):
            fig.savefig("num-unique-tokens-boxplot.eps")
        # end with
        return

    @classmethod
    def loc(cls, results_dir: Path, figs_dir: Path):
        data_lod: List[dict] = list()

        for lang in Macros.all_lang:
            loc_file = results_dir / lang / "ALL" / "metrics" / "loc-per-file.json"
            loc_list = IOUtils.load(loc_file, IOUtils.Format.json)["loc-list"]
            for loc in loc_list:
                data_lod.append({
                    "lang": lang,
                    "loc": loc,
                })
            # end for
        # end for

        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        # Plotting part
        fig: plt.Figure = plt.figure(figsize=(5, 5))
        ax: plt.Axes = fig.subplots()

        ax = sns.boxplot(data=df, x="lang", y="loc",
            width=0.6,
            linewidth=0.4,
            ax=ax,
        )

        ax.set_ylim(-370/29, 370)

        # ax.set_xlabel("Language Corpus")
        ax.set_xlabel(None)
        ax.set_ylabel("LOC")
        ax.set_xticklabels([cls.LANG_NAMES[t.get_text()] for t in ax.get_xticklabels()], rotation=45, ha="right", rotation_mode="anchor")
        fig.tight_layout()

        with IOUtils.cd(figs_dir):
            fig.savefig("loc-boxplot.eps")
        # end with

        return

    @classmethod
    def lhs_var_type_dist(cls, results_dir: Path, figs_dir: Path):
        data_lod: List[dict] = list()

        var_type_dist_file = results_dir / "vhdl" / "ALL" / "lhs-var-type-dist.txt"
        var_type_dist_list = [l for l in IOUtils.load(var_type_dist_file, IOUtils.Format.txt).splitlines()]

        for var_type in var_type_dist_list:
            key, val = var_type.split(',')
            data_lod.append({key: int(val)})

        df: pd.DataFrame = pd.DataFrame.from_dict(data_lod)

        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        ax = sns.barplot(data=df)
        ax.set_xlabel("Types")
        ax.set_ylabel("Frequencies")

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        groupedvalues = df.sum().reset_index()
        for index, row in groupedvalues.iterrows():
            dt_list = row.tolist()
            ax.text(index, dt_list[1], int(dt_list[1]), color='black', ha="center")
        
        fig.tight_layout()

        with IOUtils.cd(figs_dir):
            fig.savefig("lhs-var-type-dist-barplot.eps")
        # end with
        return

    @classmethod
    def agn_var_type_dist(cls, results_dir: Path, figs_dir: Path):
        types = list()
        str_frequencies = list()
        frequencies = list()
        
        var_type_dist_file = results_dir / "vhdl" / "ALL" / "agn-var-type-dist.txt"
        var_type_dist_list = [l for l in IOUtils.load(var_type_dist_file, IOUtils.Format.txt).splitlines()]
        
        for var_type in var_type_dist_list:
            key, val = var_type.split(',')
            if key == "<unk>":  key = "<T>"
            types.append(key)
            frequencies.append(int(val))
        # end for
        
        df: pd.DataFrame = pd.DataFrame.from_dict({"type": types, "frequency": frequencies})
        
        # Plotting part
        fig: plt.Figure = plt.figure(figsize=(20,10))
        ax: plt.Axes = fig.subplots()

        ax = sns.barplot(data=df, x="frequency", y="type")
        ax.yaxis.set_tick_params(labelsize=35)
        ax.set_xlim(0, 70_000)
        ax.set_xlabel("Frequency", fontsize=35)
        ax.set_ylabel("Type", fontsize=35)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i, frequency in enumerate(frequencies):
            y = frequency
            ax.text(y+5, i+0.2, str(frequency), color='black', ha="left", fontsize=28)
        # end for
        
        # fig.tight_layout()
        plt.tight_layout()
        
        with IOUtils.cd(figs_dir):
            fig.savefig("pa-var-type-dist-barplot.eps")
        # end with
        return

    @classmethod
    def ngram_bleu_plot(cls, results_dir: Path, figs_dir: Path):
        data_lod: List[dict] = list()
        bleu_results_file = results_dir / "ngram-pred" / "ngram_results_over_orders.txt"
        score_list: List[List[str]] = [x.split() for x in IOUtils.load(bleu_results_file, IOUtils.Format.txt).splitlines()]

        for score in score_list:
            order = int(score[0])
            bleu = float(score[-1])
            accuracy = float(score[-2])
            data_lod.append({
                        "order": order,
                        "bleu": bleu,
                        "accuracy": accuracy,
                    })
            
        df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

        # Plotting part
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        ax = sns.lineplot(data=df, x="order", y="bleu",
            markers=True,
            ci=None,
            ax=ax,
        )

        ax.set_ylim(0, 0.5)
        ax.set_xlabel("Order")
        ax.set_ylabel("BLEU")
        # for t in ax.get_legend().get_texts():
        #     if t.get_text() == "lang":
        #         t.set_text("Language")
        #     else:
        #         t.set_text(cls.LANG_NAMES[t.get_text()])
        #     # end if
        # # end for

        with IOUtils.cd(figs_dir):
            fig.savefig("ngram-rhs-completion-bleu-lineplot.eps")
        # end with

        return
