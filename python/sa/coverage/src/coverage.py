import torch
import copy
from tqdm import tqdm

from torch.nn.modules.module import register_module_forward_hook


class BaseCoverage:
    def __init__(self, model, tokenizer, key_list, task_name, device):
        self.tokenizer = tokenizer
        self.device = device
        self.model = model.eval().to(device)
        self.task_name = task_name
        self.key_list = key_list

    def compute_coverage(self, test_suite):
        pass


class NeuronBoundaryCoverage(BaseCoverage):
    def __init__(self, model, tokenizer, key_list, task_name, device):
        super(NeuronBoundaryCoverage, self).__init__(model, tokenizer, key_list, task_name, device)
        layers = [n for (n, m) in self.model.named_modules()]
        tmp = {layer: torch.empty(0) for layer in layers}
        self._features = {}

        for key in key_list:
            self._features[key] = tmp[key]

        for layer_id in self._features:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

        self.min_value, self.max_value = {}, {}

    def initialization(self, training_data):
        feature_list = self.collect_feature_map(training_data)
        for k in feature_list[0]:
            self.min_value[k] = torch.clone(feature_list[0][k])
            self.max_value[k] = torch.clone(feature_list[0][k])

        for feature in feature_list:
            for k in feature:
                if self.min_value[k].shape != feature[k].shape:
                    self.min_value[k] = torch.clone(torch.min(self.min_value[k].mean(1), feature[k].mean(1)))
                    self.max_value[k] = torch.clone(torch.max(self.max_value[k].mean(1), feature[k].mean(1)))
                else:
                    self.min_value[k] = torch.clone(torch.min(self.min_value[k], feature[k]))
                    self.max_value[k] = torch.clone(torch.max(self.max_value[k], feature[k]))
        return feature_list, self.min_value, self.max_value

    def save_outputs_hook(self, layer_id: str):
        def fn(module_, input_, output_):
            if type(output_) is torch.Tensor:
                if len(output_.size()) == 3:
                    self._features[layer_id] = output_
        return fn

    def compute_coverage(self, test_suite):
        feature_list = self.collect_feature_map(test_suite)
        upper_covered, lower_covered = {}, {}

        for k in feature_list[0]:
            upper_covered[k] = torch.zeros_like(feature_list[0][k])
            lower_covered[k] = torch.zeros_like(feature_list[0][k])

        for feature in feature_list:
            for k in feature:
                if self.min_value[k].shape != feature[k].shape:
                    upper_covered[k] += (feature[k].mean(1) > self.max_value[k])
                    lower_covered[k] += (feature[k].mean(1) < self.min_value[k])
                else:
                    upper_covered[k] += (feature[k] > self.max_value[k])
                    lower_covered[k] += (feature[k] < self.min_value[k])
        sum_num, upper_num, lower_num = 0, 0, 0
        for k in upper_covered:
            sum_num += upper_covered[k].numel()
            upper_num += (upper_covered[k] != 0).sum()
            lower_num += (lower_covered[k] != 0).sum()
        return upper_covered, lower_covered, (upper_num + lower_num) / (2 * sum_num), upper_num / sum_num

    def collect_feature_map(self, dataset):
        feature_list = []
        for x in tqdm(dataset):
            input_token = self.tokenizer(x, return_tensors="pt", padding=True).input_ids
            input_token = input_token.to(self.device)

            self.model(input_token)
            for key in self._features:
                self._features[key] = torch.clone(self._features[key].detach().cpu())
            feature_list.append(copy.deepcopy(self._features))
        return feature_list



