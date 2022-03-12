
from typing import *

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..utils.Macros import Macros
from ..utils.Utils import Utils

from .explain import CasualInferenceExplain

def get_word_importance(sentence, tokenizer, model, device, config):
    input_token = tokenizer(sentence, return_tensors="pt", padding=True).input_ids
    input_token = input_token.to(device)
    
    exp = CasualInferenceExplain(model, tokenizer, device, config)
    res, importance_scores = exp.explain(sentence)
    return res, importance_scores

def explain_nlp_main(task, dataset_name, selection_method, model_name, _type='pass->fail'):
    test_results_dir = Macros.result_dir / f"test_results_{task}_{dataset_name}_{selection_method}"
    test_results_analysis_file = test_results_dir / 'test_result_analysis.json'
    data = Utils.read_json(test_results_analysis_file)
    _data = [s for s in data[model_name] if 'pass->fail' in s.keys()]
    
    original_sent_list = list()
    for d in _data:
        if len(d[_type]) != 0:
            for s in d[_type]:
                original_sent_list.append({
                    'req': d['req'],
                    'from_sent': s['from'],
                    'to_sent': s['to']
                })
            # end for
        # end if
    # end for
    
    # original_sent_list = [
    #     s['pass->fail'][0]['from']['sent']
    #     for s in _data
    #     if len(s['pass->fail']) != 0 
    # ]

    print()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    config = {
        'mutate_rate': 0.8,
        'mutate_num': 500
    }

    for sent_dict in original_sent_list:
        req = sent_dict['req']
        from_sent = sent_dict['from_sent']['sent']
        from_pred = sent_dict['from_sent']['pred']
        from_label = sent_dict['from_sent']['label']
        from_conf = sent_dict['from_sent']['conf']
        
        to_sent_list = sent_dict['to_sent']

        # from_sent
        res_from_sent, imp_score_from_sent = get_word_importance(from_sent, tokenizer, model, device, config)

        # to_sents
        for to_sent_dict in to_sent_list:
            print('>>>>>>>>>>')
            print(f"LC:{req}\nFROM_SENT:{from_sent}\nFROM_PRED:{from_pred}\nFROM_LABEL:{from_label}\nFROM_CONF:{from_conf}")
            print(res_from_sent)
            print(imp_score_from_sent)
            # for fs in res_from_sent:
            #     print(fs)
            # # end for
            print('----------')

            to_sent = to_sent_dict['sent']
            to_pred = to_sent_dict['pred']
            to_label = to_sent_dict['label']
            to_conf = to_sent_dict['conf']
            res_to_sent, imp_score_to_sent = get_word_importance(to_sent, tokenizer, model, device, config)
            print(f"TO_SENT:{to_sent}\nTO_PRED:{to_pred}\nTO_LABEL:{to_label}\nTO_CONF:{to_conf}")
            print(res_to_sent)
            print(imp_score_to_sent)
            # for ts in res_to_sent:
            #     print(ts)
            # # end for
            print('<<<<<<<<<<')
        # end for
    # end for
    return
