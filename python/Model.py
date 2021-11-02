# This script is to load models to be evaluated
# given generated test suite

from typing import *
from pathlib import Path
from transformers import pipeline, AutoTokenizer
# from nlp import load_dataset

import os
import numpy as np

from Utils import Utils
from Macros import Macros

class Model:
    
    model_map = {
        "sentiment_analysis": ("sentiment-analysis", Macros.sa_models_file)
    }

    @classmethod
    def read_model_list(cls, task: str):
        _task, model_file = cls.model_map[task]
        return _task, [l.strip() for l in Utils.read_txt(model_file)]

    @classmethod
    def load_models(cls, task: str):
        _task, model_names = cls.read_model_list(task)
        for m in model_names:
            tokenizer = AutoTokenizer.from_pretrained(m)
            yield m, pipeline(_task, model=m, tokenizer=tokenizer, framework="pt", device=0)
        # end for

    @classmethod
    def get_batch(cls, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
        # end for

    @classmethod
    def batch_predict(cls, data: List[str], batch_size: int = 32):
        preds = list()
        for d in cls.get_batch(data, batch_size):
            preds.extend(cls.model(d))
        # end for
        return preds
    
    @classmethod
    def sentiment_pred_and_conf(cls, data: List[str]):
        # change format to softmax, make everything in [0.33, 0.66] range be predicted as neutral
        preds = cls.batch_predict(data)
        pr = np.array([x['score'] if x['label'] == 'POSITIVE' else 1 - x['score'] for x in preds])
        pp = np.zeros((pr.shape[0], 3))
        margin_neutral = 1/3.
        mn = margin_neutral / 2.
        neg = pr < 0.5 - mn
        pp[neg, 0] = 1 - pr[neg]
        pp[neg, 2] = pr[neg]
        pos = pr > 0.5 + mn
        pp[pos, 0] = 1 - pr[pos]
        pp[pos, 2] = pr[pos]
        neutral_pos = (pr >= 0.5) * (pr < 0.5 + mn)
        pp[neutral_pos, 1] = 1 - (1 / margin_neutral) * np.abs(pr[neutral_pos] - 0.5)
        pp[neutral_pos, 2] = 1 - pp[neutral_pos, 1]
        neutral_neg = (pr < 0.5) * (pr > 0.5 - mn)
        pp[neutral_neg, 1] = 1 - (1 / margin_neutral) * np.abs(pr[neutral_neg] - 0.5)
        pp[neutral_neg, 0] = 1 - pp[neutral_neg, 1]
        preds = np.argmax(pp, axis=1)
        return preds, pp

    @classmethod
    def run(cls, testsuite, model, pred_and_conf_fn):
        cls.model = model
        testsuite.run(pred_and_conf_fn, n=500, overwrite=True)
        testsuite.summary(n=100)
        return