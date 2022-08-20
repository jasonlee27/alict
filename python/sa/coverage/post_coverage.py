import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from typing import *
from pathlib import Path
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nlp_task', type=str, default='sa',
                    choices=['sa'],
                    help='nlp task of focus')
parser.add_argument('--search_dataset', type=str, default='sst',
                    help='name of dataset for searching testcases that meets the requirement')
parser.add_argument('--syntax_selection', type=str, default='random',
                    choices=['prob', 'random', 'bertscore', 'noselect'],
                    help='method for selection of syntax suggestions')
parser.add_argument('--lc_cksum', type=str, default='',
                    help='cksum value for lc')
args = parser.parse_args()

nlp_task = args.nlp_task
search_dataset_name = args.search_dataset
selection_method = args.syntax_selection
lc_cksum = args.lc_cksum

storage_dir: Path = Path('/glusterfs/data/jxl115330/s2lct')
result_dir: Path = storage_dir / "_results"
cov_dir: Path = result_dir / f"seeds_{nlp_task}_{search_dataset_name}" / 'coverage'
new_log_dir: Path = cov_dir / 'new_log' / lc_cksum
res_dir: Path = cov_dir / 'res' / lc_cksum

res_dir.mkdir(parents=True, exist_ok=True)

# if not os.path.isdir('res'):
#     os.mkdir('res')

def measure_coverage(upper_covered, lower_covered):
    sum_num, upper_num, lower_num = 0, 0, 0
    for k in upper_covered:
        sum_num += upper_covered[k].numel()
        upper_num += (upper_covered[k] != 0).sum()
        lower_num += (lower_covered[k] != 0).sum()
    return (upper_num + lower_num) / (2 * sum_num), upper_num / sum_num


for task_id in range(3):
    task_name = str(task_id)
    our_file = new_log_dir / f"ours_feature_{task_name}_0.pkl"
    base_file = new_log_dir / f"baseline_feature_{task_name}_0.pkl"

    our_cov1, our_cov2, _, _ = torch.load(str(our_file))
    base_cov1, base_cov2, _, _ = torch.load(str(base_file))
    min_val, max_val = torch.load(str(new_log_dir / f"train_feature_{task_name}.pkl"))

    for k in our_cov1:
        if our_cov1[k].shape != min_val[k].shape:
            our_cov1[k] = our_cov1[k].mean(1)
            our_cov2[k] = our_cov2[k].mean(1)
            base_cov1[k] = base_cov1[k].mean(1)
            base_cov2[k] = base_cov2[k].mean(1)
    res = []
    for i in tqdm(range(1, 100)):
        our_file = new_log_dir / f"ours_feature_{task_name}_{str(i)}.pkl"
        base_file = new_log_dir / f"baseline_feature_{task_name}_{str(i)}.pkl"
        # our_file = 'new_log/ours_feature_' + task_name + '_' + str(i) + '.pkl'
        # base_file = 'new_log/baseline_feature_' + task_name + '_' + str(i) + '.pkl'
        if os.path.exists(str(our_file)) and os.path.exists(str(base_file)):
            our_new_cov1, our_new_cov2, _, _ = torch.load(str(our_file))
            base_new_cov1, base_new_cov2, _, _ = torch.load(str(base_file))

            for k in our_cov1:
                if our_cov1[k].size() != our_new_cov1[k].size():
                    our_cov1[k] += our_new_cov1[k].mean(1)
                    our_cov2[k] += our_new_cov2[k].mean(1)
                    base_cov1[k] += base_new_cov1[k].mean(1)
                    base_cov2[k] += base_new_cov2[k].mean(1)
                else:
                    our_cov1[k] += our_new_cov1[k]
                    our_cov2[k] += our_new_cov2[k]
                    base_cov1[k] += base_new_cov1[k]
                    base_cov2[k] += base_new_cov2[k]
                # end if
            # end for
            our_c1, our_c2 = measure_coverage(our_cov1, our_cov2)
            base_c1, base_c2 = measure_coverage(base_cov1, base_cov2)
            res.append([our_c1, our_c2, base_c1, base_c2 ])
        # end if
    # end for
    res = np.array(res)
    fig, ax = plt.subplots()
    ax.plot(list(range(len(res[:, 0]))), res[:, 0], label='our_c1', color='red')
    ax.plot(list(range(len(res[:, 2]))), res[:, 2], label='base_c1', color = 'blue')
    # plt.plot(res[:, 0], 'r')
    # plt.plot(res[:, 2], 'b')
    plt.show()
    ax.plot(list(range(len(res[:, 1]))), res[:, 1], label='our_c2', color='red')
    ax.plot(list(range(len(res[:, 3]))), res[:, 3], label='base_c2', color = 'blue')
    # plt.plot(res[:, 1], 'r')
    # plt.plot(res[:, 3], 'b')
    plt.show()
    plt.savefig(str(res_dir / f"{str(task_id)}.png"))
    np.savetxt(str(res_dir / f"{str(task_id)}.csv"), res, delimiter=',')
    # np.savetxt('res/' + str(task_id) + '.csv', res, delimiter=',')
