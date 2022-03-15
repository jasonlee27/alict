
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

    @classmethod
    def make_tables(cls, **options):
        which: Union[str, List[str]] = options.pop("which", [])
        paper_dir: Path = Path(options.pop("paper_dir", Macros.paper_dir))
        tables_dir: Path = paper_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(which, list):
            which = [which]
        # end if

        for item in which:
            if item == "lc-req":
                cls.make_numbers_lc_requirement(Macros.results_dir, Macros.data_dir, tables_dir)
                cls.make_table_lc_requirement(tables_dir)
            else:
                cls.logger.warning(f"Unknown table {item}")
            # end if
        # end for
        return

    @classmethod
    def make_numbers_lc_requirement(cls, results_dir: Path, data_dir: Path, tables_dir: Path):
        output_file = latex.File(tables_dir / 'lc-requirement-numbers.tex')
        
        for lang in Macros.all_lang:
            # number of repos
            url_file = data_dir / lang / "repositories.txt"
            num_repos = len(IOUtils.load(url_file, IOUtils.Format.txt).splitlines())
            output_file.append_macro(latex.Macro(f"dataset-{lang}-num-repos", cls.FMT_INT.format(num_repos)))

            # number of files, parsable, unparsable, parsable-unique
            fc_output_file = results_dir / lang / "ALL" / "fc.txt"
            fc_output = IOUtils.load(fc_output_file, IOUtils.Format.txt)
            files_parsable = [l.split(":") for l in fc_output.splitlines()]
            num_files = len(files_parsable)
            output_file.append_macro(latex.Macro(f"dataset-{lang}-num-all-files", cls.FMT_INT.format(num_files)))

            num_files_parsable = len([f for f, p in files_parsable if p == "true"])
            output_file.append_macro(latex.Macro(f"dataset-{lang}-num-files-parsable", cls.FMT_INT.format(num_files_parsable)))
            percent_parsable = num_files_parsable / num_files
            output_file.append_macro(latex.Macro(f"dataset-{lang}-percent-files-parsable", cls.FMT_PER.format(percent_parsable).replace('%', r'\%')))

            num_files_unparsable = len([f for f, p in files_parsable if p == "false"])
            output_file.append_macro(latex.Macro(f"dataset-{lang}-num-files-unparsable", cls.FMT_INT.format(num_files_unparsable)))
            percent_unparsable = num_files_unparsable / num_files
            output_file.append_macro(latex.Macro(f"dataset-{lang}-percent-files-unparsable", cls.FMT_PER.format(percent_unparsable).replace('%', r'\%')))

            num_files_metrics_file = results_dir / lang / "ALL" / "metrics" / "num-files.json"
            num_files_metrics = IOUtils.load(num_files_metrics_file, IOUtils.Format.json)
            for k, v in num_files_metrics.items():
                output_file.append_macro(latex.Macro(f"dataset-{lang}-{k}", (cls.FMT_INT if isinstance(v, int) else cls.FMT_FLOAT).format(v)))
                if k == "frac-duplicate-files":
                    output_file.append_macro(latex.Macro(f"dataset-{lang}-percent-duplicate-files", cls.FMT_PER.format(v).replace('%', r'\%')))
                # end if
            # end for

            # LOC
            loc_metrics_file = results_dir / lang / "ALL" / "metrics" / "loc-per-file.json"
            loc_metrics = IOUtils.load(loc_metrics_file, IOUtils.Format.json)
            for s, func in Utils.SUMMARIES_FUNCS.items():
                output_file.append_macro(latex.Macro(f"dataset-{lang}-loc-{s}", (cls.FMT_INT if Utils.SUMMARIES_PRESERVE_INT[s] else cls.FMT_FLOAT).format(loc_metrics[f"loc-per-file-{s}"])))
            # end for

            # vocab size
            vocab_size_file = results_dir / lang / "ALL" / "metrics" / "vocab-size.txt"
            vocab_size = int(IOUtils.load(vocab_size_file, IOUtils.Format.txt))
            output_file.append_macro(latex.Macro(f"dataset-{lang}-vocab-size", cls.FMT_INT.format(vocab_size)))

            # num tokens
            num_tokens_file = results_dir / lang / "ALL" / "metrics" / "num-tokens-per-file.txt"
            num_tokens_list = [int(l) for l in IOUtils.load(num_tokens_file, IOUtils.Format.txt).splitlines() if l != ""]
            for s, func in Utils.SUMMARIES_FUNCS.items():
                output_file.append_macro(latex.Macro(f"dataset-{lang}-num-tokens-{s}", (cls.FMT_INT if Utils.SUMMARIES_PRESERVE_INT[s] else cls.FMT_FLOAT).format(func(num_tokens_list))))
            # end for
        # end for

        output_file.save()
        return
