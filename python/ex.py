
from typing import *
from pathlib import Path

from checklist.test_suite import TestSuite as suite

import os


def example_to_dict_fn(data):
    return {
        "test_sent": data,
    }

this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__))) # nlptest/python/utils
root_dir: Path = this_dir.parent # nlptest/
download_dir: Path = root_dir / "_downloads" # nlptest/_downloads

checklist_dir: Path = download_dir / "checklist"
checklist_data_dir: Path = checklist_dir / "release_suites"
checklist_sst_dataset_file: Path = checklist_data_dir / "ex_sentiment_suite.pkl"

tsuite = suite().from_file(checklist_sst_dataset_file)
# tsuite.to_raw_file(checklist_data_dir / "ex_sentiment_suite.txt")
tsuite_dict = tsuite.to_dict(example_to_dict_fn=example_to_dict_fn)

test_names = list(set(tsuite_dict['test_name']))
test_data = dict()
for test_name in test_names:
    test_data[test_name] = {
        "sents": tsuite.tests[test_name].data,
        "labels": tsuite.tests[test_name].labels
    }
    if type(test_data[test_name]['labels'])!=list:
        test_data[test_name]['labels'] = [test_data[test_name]['labels']]*len(test_data[test_name]['sents'])
    # end if
    # print(test_name, len(test_data[test_name]['sents']), len(test_data[test_name]['labels']))
# end for

num_examples = 100
results = list()
for idx in range(num_examples):
    test_sent = tsuite_dict['test_sent'][idx]
    test_name = tsuite_dict['test_name'][idx]
    test_case = tsuite_dict['test_case'][idx]
    label = test_data[test_name]['labels'][
        test_data[test_name]['sents'].index(test_sent)
    ]
    results.append({
        "sent": test_sent,
        "label": label,
        "test_case": test_case,
        "test_name": test_name
    })
    print(results[-1])
# end if
    
# for key in tsuite_dict.keys():
#     print(f"{key}:: {tsuite_dict[key][-10:]}")
# # end for
