# This script is to load models to be evaluated
# given generated test suite

from typing import *
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModel
# from nlp import load_dataset

import os
import numpy as np

from ..utils.Macros import Macros
from ..utils.Utils import Utils

class Model:
    
    model_map = {
        "hs": ("text-classification", Macros.hs_models_file)
    }

    @classmethod
    def read_model_list(cls, task: str):
        _task, model_file = cls.model_map[task]
        return _task, [l.strip() for l in Utils.read_txt(model_file)]

    @classmethod
    def load_models(cls, task: str):
        _task, model_names = cls.read_model_list(task)
        for m in model_names:
            print('@@@', m)
            # model = AutoModel.from_pretrained(m)
            # print(type(model))
            tokenizer = AutoTokenizer.from_pretrained(m)
            yield m, pipeline(task=_task,
                              model=m,
                              tokenizer=tokenizer,
                              framework="pt",
                              device=0)
        # end for

    @classmethod
    def load_local_model(cls, task, model_name):
        # model_dir = Macros.retrain_output_dir / model_name.replace("/", "-")
        model_dir = Macros.retrain_model_dir / task / model_name
        _task, model_file = cls.model_map[task]
        checkpoints = sorted([d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir,d)) and d.startswith("checkpoint-")])
        checkpoint_dir = model_dir / checkpoints[-1]
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return pipeline(_task, model=str(checkpoint_dir), tokenizer=tokenizer, framework="pt", device=0)

    @classmethod
    def get_batch(cls, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
        # end for

    @classmethod
    def batch_predict(cls, data: List[str], batch_size: int = 16):
        preds = list()
        for d in cls.get_batch(data, batch_size):
            preds.extend(cls.model(d))
        # end for
        return preds
    
    @classmethod
    def sentiment_pred_and_conf(cls, data: List[str], batch_size=16):
        # cardiffnlp/twitter-roberta-base-hate: {'label': 'hate/non-hate', 'score': 0.5082786083221436}
        preds = cls.batch_predict(data, batch_size=batch_size)
        pr = np.array([x['score'] if x['label'].lower()=='hate' else 1 - x['score'] for x in preds])
        pp = np.zeros((pr.shape[0], 2))
        nht = pr <= 0.5
        pp[nht, 0] = 1 - pr[nht]
        pp[nht, 1] = pr[nht]
        ht = pr > 0.5
        pp[ht, 0] = 1 - pr[ht]
        pp[ht, 1] = pr[ht]
        preds = np.argmax(pp, axis=1)
        return preds, pp

    @classmethod
    def format_example(cls, x, pred, conf, *args, **kwargs):
        softmax = type(conf) in [np.array, np.ndarray]
        expect_result = args[0]
        label = expect_result
        pred_res = "FAIL" if pred!=label else "PASS"
        binary = False
        if softmax:
            if conf.shape[0] == 2:
                confs = ' '.join(['%.1f' % c for c in conf])
                # conf = conf[1]
                return f"DATA::{pred_res}::{str(confs)}::{str(pred)}::{str(label)}::{str(x)}"
            elif conf.shape[0] <= 4:
                confs = ' '.join(['%.1f' % c for c in conf])
                # return f"{pred_res}::{conf}::{str(x)}"
                return f"DATA::{pred_res}::{str(confs)}::{str(pred)}::{str(label)}::{str(x)}"
            else:
                confs = ' '.join(['%.1f' % c for c in conf])
                # return f"{pred_res}::{pred}:({conf:%.1f})::{str(x)}"
                return f"DATA::{pred_res}::{str(confs)}::{str(pred)}::{str(label)}::{str(x)}"
        else:
            return f"DATA::{pred_res}::[]::{str(pred)}::{str(label)}::{str(x)}"
        
    @classmethod
    def print_result(cls, x, pred, conf, expect_result, label=None, meta=None, format_example_fn=None, logger=None):
        isfailed = expect_result[0]!=True
        if logger is None:
            print(format_example_fn(x, pred, conf, expect_result, label, isfailed=isfailed))
        else:
            logger.print(format_example_fn(x, pred, conf, expect_result, label, isfailed=isfailed))
        # end if

    @classmethod
    def run(cls,
            testsuite,
            model,
            pred_and_conf_fn,
            print_fn=None,
            format_example_fn=None,
            n=Macros.nsamples,
            logger=None):
        cls.model = model
        testsuite.run(pred_and_conf_fn, n=None, overwrite=True, logger=logger)
        testsuite.summary(logger=logger,
                          print_fn=cls.print_result if print_fn is not None else None,
                          format_example_fn=cls.format_example if format_example_fn is None else None)
        return
