import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from explainNLP.explain import CasualInferenceExplain


with open('./test_result_analysis.json', 'r') as f:
    data = json.load(f)

original_sent_list = [
    s['pass->fail'][0]['from']['sent']
    for s in data['textattack/roberta-base-SST-2']
    if len(s['pass->fail']) != 0
]

print()

tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")
model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2")
device = torch.device(7)
model = model.eval().to(device)
config = {
    'mutate_rate': 0.8,
    'mutate_num': 500
}

for sentence in original_sent_list:
    input_token = tokenizer(sentence, return_tensors="pt", padding=True).input_ids
    input_token = input_token.to(device)

    exp = CasualInferenceExplain(model, tokenizer, device, config)
    res = exp.explain(sentence)
    print(sentence)
    for s in res:
        print(s)
    print('----------------------------------------')