from typing import *
from pathlib import Path

import os
import csv

from Macros import Macros

class Data:

    @classmethod
    def read_csv(cls, file_path: Path, have_column_name=True):
        column_names = None
        data_list = list()
        with open(str(file_path), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row_i, row in enumerate(reader):
                if have_column_name:
                    if row_i==0:
                        column_names = row
                    else:
                        data_list.append(row)
                    # end if
                else:
                    data_list.append(row)
                # end if
            # end for
        # end with
        return data_list, column_names

    @classmethod
    def get_hatechecker_templates(cls, file_path: Path):
        result = dict()
        templ_column_names = ["templ_id", "case_templ"]
        csv_data, column_names = cls.read_csv(file_path, have_column_name=True)
        templ_id_index = column_names.index(templ_column_names[0])
        case_templ_index = column_names.index(templ_column_names[1])
        for d in csv_data:
            templ_id, case_templ = d[templ_id_index], d[case_templ_index]
            if templ_id not in result.keys():
                result[str(templ_id)] = case_templ
            # end if
        # end for
        return result

    @classmethod
    def write_hatecheck_templates(cls):
        csv_path = Macros.hatecheck_data_dir / "test_suite_cases.csv"
        result_dir = Macros.result_dir / "hatecheck"
        result_dir.mkdir(parents=True, exist_ok=True)
        templates = cls.get_hatechecker_templates(csv_path)
        with open(result_dir / "templates.txt", 'w') as f:
            for _id, templ in templates.items():
                f.write(f"{_id}\t{templ}\n")
            # end for
        # end with

if __name__=="__main__":
    Data.write_hatecheck_templates()
    
