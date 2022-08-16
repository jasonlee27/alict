import os
import torch
import numpy as np

from utils import *

if not os.path.isdir('new_log'):
    os.mkdir('new_log')
if not os.path.isdir('res'):
    os.mkdir('res')

for task_id in range(3):
    model, tokenizer, key_list = load_model(task_id=task_id)
    train_data, our_suite, baseline_suite = load_test_suite(task_id=task_id)
    device = torch.device(0)
    task_name = str(task_id)
    coverage_module = NeuronBoundaryCoverage(model, tokenizer, key_list, task_name, device)
    feature_list, min_value, max_value = \
        coverage_module.initialization(train_data)
    torch.save([min_value, max_value], 'new_log/train_feature_' + task_name + '.pkl')

    iter_num = min(len(baseline_suite) // 100, len(our_suite) // 100)

    coverage_res = []
    for i in range(iter_num):
        st, ed = i * 100, i * 100 + 100
        upper_covered, lower_covered, base_cov_1, base_cov_2 = \
            coverage_module.compute_coverage(baseline_suite[st:ed])
        torch.save([upper_covered, lower_covered, base_cov_1, base_cov_2], 'new_log/baseline_feature_' + task_name + '_' + str(i) + '.pkl')
        print(task_name, 'baseline', base_cov_1, base_cov_2)

        upper_covered, lower_covered, cov_1, cov_2 = \
            coverage_module.compute_coverage(our_suite[st:ed])
        torch.save([upper_covered, lower_covered, cov_1, cov_2], 'new_log/ours_feature_' + task_name + '_' + str(i) + '.pkl')
        print(task_name, 'ours', cov_1, cov_2)

        coverage_res.append([base_cov_1, base_cov_2, cov_1, cov_2])
    coverage_res = np.array(coverage_res)
    np.savetxt('res/coverage_' + task_name + 'csv', coverage_res, delimiter=',')