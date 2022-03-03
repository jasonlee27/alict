
from typing import *

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..utils.Macros import Macros
from ..utils.Utils import Utils

from .explain import CasualInferenceExplain


def explain_nlp_main(task, dataset_name, selection_method, model_name):
    test_results_dir = Macros.result_dir / f"test_results_{task}_{dataset_name}_{selection_method}"
    test_results_analysis_file = test_results_dir / 'test_results_analyisis.json'
    data = Utils.read_json(test_results_analysis_file)
    original_sent_list = [
        s['pass->fail'][0]['from']['sent']
        for s in data[model_name]
        if len(s['pass->fail']) != 0
    ]

    print()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
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
    # end for
    return
