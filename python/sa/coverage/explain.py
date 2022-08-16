import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from explainNLP.explain import CasualInferenceExplain

eval_model_name = 'textattack/roberta-base-SST-2'

with open('./test_result_analysis.json', 'r') as f:
    data = json.load(f)

original_pass_fail = [
    s['fail->pass']
    # s['pass->fail']
    for s in data[eval_model_name]
    if len(s['pass->fail']) != 0
]

original_type = [
    s['req']
    for s in data[eval_model_name]
    if len(s['pass->fail']) != 0
]

exp_sentence_list = []
for sentence_list, s_type in zip(original_pass_fail, original_type):
    for sentence in sentence_list:
        from_sent = sentence['from']['sent']
        to_sent_list = [s['sent'] for s in sentence['to']]
        exp_sentence_list.append([from_sent, to_sent_list, s_type])

tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
model = AutoModelForSequenceClassification.from_pretrained(eval_model_name)
device = torch.device(0)
model = model.eval().to(device)
config = {
    'mutate_rate': 0.8,
    'mutate_num': 500,
    'threshold': 0.9,
}

for sentence_pair in exp_sentence_list:
    ori_sent, mutated_sent_list, s_type = sentence_pair
    sentence_list = [ori_sent] + mutated_sent_list

    for i, sentence in enumerate(sentence_list):
        input_token = tokenizer(sentence, return_tensors="pt", padding=True).input_ids
        input_token = input_token.to(device)

        exp = CasualInferenceExplain(model, tokenizer, device, config)
        res, ori_predict = exp.explain(sentence)

        mask = exp.create_mask(res, sentence, ori_predict)
        mask_str = exp.tokenizer.decode([m for m in mask[0]])
        mask_str = mask_str.replace(exp.tokenizer.unk_token, '[MASK]')
        print(sentence)
        print(mask_str)
        print(ori_predict)
        res = sorted(res, key=lambda x: x[0])
        for s in res:
            print(s[-1])
        print(s_type, i, '----------------------------------------')
    print('###########################################')