from typing import *

import os
from pathlib import Path


class Macros:

    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    root_dir: Path = this_dir.parent.parent
    result_dir: Path = root_dir / "_results"
    download_dir: Path = root_dir / "_downloads"

    hatecheck_data_dir: Path = download_dir / "hatecheck" / "hatecheck-data"
    
    FMT_INT = "{:,d}"
    FMT_PER = "{:.1%}"
    FMT_FLOAT = "{:,.2f}"