import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

if not os.path.isdir('res'):
    os.mkdir('res')

def measure_coverage(upper_covered, lower_covered):
    sum_num, upper_num, lower_num = 0, 0, 0
    for k in upper_covered:
        sum_num += upper_covered[k].numel()
        upper_num += (upper_covered[k] != 0).sum()
        lower_num += (lower_covered[k] != 0).sum()
    return (upper_num + lower_num) / (2 * sum_num), upper_num / sum_num


for task_id in range(3):
    task_name = str(task_id)
    our_file = 'new_log/ours_feature_' + task_name + '_0.pkl'
    base_file = 'new_log/baseline_feature_' + task_name + '_0.pkl'

    our_cov1, our_cov2, _, _ = torch.load(our_file)
    base_cov1, base_cov2, _, _ = torch.load(base_file)
    min_val, max_val = torch.load('new_log/train_feature_' + task_name + '.pkl')

    for k in our_cov1:
        if our_cov1[k].shape != min_val[k].shape:
            our_cov1[k] = our_cov1[k].mean(1)
            our_cov2[k] = our_cov2[k].mean(1)
            base_cov1[k] = base_cov1[k].mean(1)
            base_cov2[k] = base_cov2[k].mean(1)
    res = []
    for i in tqdm(range(1, 100)):
        our_file = 'new_log/ours_feature_' + task_name + '_' + str(i) + '.pkl'
        base_file = 'new_log/baseline_feature_' + task_name + '_' + str(i) + '.pkl'

        our_new_cov1, our_new_cov2, _, _ = torch.load(our_file)
        base_new_cov1, base_new_cov2, _, _ = torch.load(base_file)

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

        our_c1, our_c2 = measure_coverage(our_cov1, our_cov2)
        base_c1, base_c2 = measure_coverage(base_cov1, base_cov2)
        res.append([our_c1, our_c2, base_c1, base_c2 ])
    res = np.array(res)
    plt.plot(res[:, 0], 'r')
    plt.plot(res[:, 2], 'b')
    plt.show()
    plt.plot(res[:, 1], 'r')
    plt.plot(res[:, 3], 'b')
    plt.show()
    np.savetxt('res/' + str(task_id) + '.csv', res, delimiter=',')
