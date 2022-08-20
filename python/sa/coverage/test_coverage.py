import os
import torch
import numpy as np
import argparse

from typing import *
from pathlib import Path

from utils import *


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--our_sents', type=str, default=None,
                    help='filename of target our sents')
parser.add_argument('--bl_sents', type=str, default=None,
                    help='filename of target baseline(checklist) sents')
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

new_log_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)

for task_id in range(3):
    model, tokenizer, key_list = load_model(task_id=task_id)
    train_data, our_suite, baseline_suite = load_test_suite(task_id,
                                                            args.our_sents,
                                                            args.bl_sents)
    device = torch.device(0)
    task_name = str(task_id)
    coverage_module = NeuronBoundaryCoverage(model, tokenizer, key_list, task_name, device)
    feature_list, min_value, max_value = \
        coverage_module.initialization(train_data)
    save_file = 'train_feature_' + task_name + '.pkl'
    torch.save([min_value, max_value], str(new_log_dir / save_file))
    # torch.save([min_value, max_value], 'new_log/train_feature_' + task_name + '.pkl')

    iter_num = min(len(baseline_suite) // 100, len(our_suite) // 100)

    coverage_res = []
    for i in range(iter_num):
        st, ed = i * 100, i * 100 + 100
        upper_covered, lower_covered, base_cov_1, base_cov_2 = \
            coverage_module.compute_coverage(baseline_suite[st:ed])
        save_file = 'baseline_feature_' + task_name + '_' + str(i) + '.pkl'
        # torch.save([upper_covered, lower_covered, base_cov_1, base_cov_2], 'new_log/baseline_feature_' + task_name + '_' + str(i) + '.pkl')
        torch.save([upper_covered, lower_covered, base_cov_1, base_cov_2], str(new_log_dir / save_file))
        print(task_name, 'baseline', base_cov_1, base_cov_2)

        upper_covered, lower_covered, cov_1, cov_2 = \
            coverage_module.compute_coverage(our_suite[st:ed])
        save_file = 'ours_feature_' + task_name + '_' + str(i) + '.pkl'
        torch.save([upper_covered, lower_covered, cov_1, cov_2], str(new_log_dir / save_file))
        # torch.save([upper_covered, lower_covered, cov_1, cov_2], 'new_log/ours_feature_' + task_name + '_' + str(i) + '.pkl')
        print(task_name, 'ours', cov_1, cov_2)

        coverage_res.append([base_cov_1, base_cov_2, cov_1, cov_2])
    coverage_res = np.array(coverage_res)
    save_file = 'coverage_' + task_name + '.csv'
    np.savetxt(str(res_dir / save_file), coverage_res, delimiter=',')
    # np.savetxt('res/coverage_' + task_name + 'csv', coverage_res, delimiter=',')



    
# for task_id in range(3):
#     model, tokenizer, key_list = load_model(task_id=task_id)
#     train_data, our_suite, baseline_suite = load_test_suite(task_id,
#                                                             args.our_sents,
#                                                             args.bl_sents)
#     device = torch.device(0)
#     task_name = str(task_id)
#     coverage_module = NeuronBoundaryCoverage(model, tokenizer, key_list, task_name, device)
#     feature_list, min_value, max_value = \
#         coverage_module.initialization(train_data)
#     save_file = 'train_feature_' + task_name + '.pkl'
#     torch.save([min_value, max_value], str(new_log_dir / save_file))
#     # torch.save([min_value, max_value], 'new_log/train_feature_' + task_name + '.pkl')

#     iter_num = min(len(baseline_suite) // 100, len(our_suite) // 100)

#     coverage_res = []
#     for i in range(iter_num):
#         st, ed = i * 100, i * 100 + 100
#         upper_covered, lower_covered, base_cov_1, base_cov_2 = \
#             coverage_module.compute_coverage(baseline_suite[st:ed])
#         save_file = 'baseline_feature_' + task_name + '_' + str(i) + '.pkl'
#         # torch.save([upper_covered, lower_covered, base_cov_1, base_cov_2], 'new_log/baseline_feature_' + task_name + '_' + str(i) + '.pkl')
#         torch.save([upper_covered, lower_covered, base_cov_1, base_cov_2], str(new_log_dir / save_file))
#         print(task_name, 'baseline', base_cov_1, base_cov_2)

#         upper_covered, lower_covered, cov_1, cov_2 = \
#             coverage_module.compute_coverage(our_suite[st:ed])
#         save_file = 'ours_feature_' + task_name + '_' + str(i) + '.pkl'
#         torch.save([upper_covered, lower_covered, cov_1, cov_2], str(new_log_dir / save_file))
#         # torch.save([upper_covered, lower_covered, cov_1, cov_2], 'new_log/ours_feature_' + task_name + '_' + str(i) + '.pkl')
#         print(task_name, 'ours', cov_1, cov_2)

#         coverage_res.append([base_cov_1, base_cov_2, cov_1, cov_2])
#     coverage_res = np.array(coverage_res)
#     save_file = 'coverage_' + task_name + '.csv'
#     np.savetxt(str(res_dir / save_file), coverage_res, delimiter=',')
#     # np.savetxt('res/coverage_' + task_name + 'csv', coverage_res, delimiter=',')
