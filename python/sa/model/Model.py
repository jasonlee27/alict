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
        "sa": ("sentiment-analysis", Macros.sa_models_file)
    }

    @classmethod
    def read_model_list(cls, task: str):
        _task, model_file = cls.model_map[task]
        return _task, [l.strip() for l in Utils.read_txt(model_file)]

    @classmethod
    def load_models(cls, task: str):
        _task, model_names = cls.read_model_list(task)
        for m in model_names:
            # model = AutoModel.from_pretrained(m)
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
        # distilbert-base-uncased-finetuned-sst-2-english: label [NEGATIVE, POSITIVE]
        # textattack/bert-base-uncased-SST-2: label [LABEL_0, LABEL_1]
        # textattack/bert-base-SST-2: label [LABEL_0, LABEL_1]
        # change format to softmax, make everything in [0.33, 0.66] range be predicted as neutral
        preds = cls.batch_predict(data, batch_size=batch_size)
        pr = np.array([x['score'] if x['label']=='POSITIVE' or x['label']=='LABEL_1' else 1 - x['score'] for x in preds])
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
    def format_example(cls, x, pred, conf, *args, **kwargs):
        softmax = type(conf) in [np.array, np.ndarray]
        pred_res = "FAIL" if kwargs["isfailed"] else "PASS"
        expect_result, label = args[0], args[1]
        binary = False
        if softmax:
            # print(conf.shape, type(conf), type(conf[0]))
            if conf.shape[0] == 2 and type(conf[0])!=np.ndarray:
                conf = conf[1]
                return f"DATA::{pred_res}::{conf:%.1f}::{str(pred)}::{str(label)}::{str(x)}"
            elif conf.shape[0] <= 4 and type(conf[0])!=np.ndarray:
                confs = ' '.join(['%.1f' % c for c in conf])
                # return f"{pred_res}::{conf}::{str(x)}"
                return f"DATA::{pred_res}::{str(confs)}::{str(pred)}::{str(label)}::{str(x)}"
            elif type(conf[0])==np.ndarray:
                conf = ' '.join(['%.1f' % c for c in conf[0]])
                # return f"{pred_res}::{pred}:({conf:%.1f})::{str(x)}"
                return f"DATA::{pred_res}::{str(conf)}::{str(pred)}::{str(label)}::{str(x[0])}"
            else:
                conf = conf[pred]
                # return f"{pred_res}::{pred}:({conf:%.1f})::{str(x)}"
                return f"DATA::{pred_res}::{conf:%.1f}::{str(pred)}::{str(label)}::{str(x)}"
            # end if
        else:
            return f"DATA::{pred_res}::[]::{str(pred)}::{str(label)}::{str(x)}"
        
    @classmethod
    def print_result(cls, x, pred, conf, expect_result, passed=None, label=None, meta=None, format_example_fn=None, nsamples=None, logger=None):
        if type(expect_result[0])==bool:
            isfailed = expect_result[0]!=True
        else:
            isfailed = not passed
        # end if
        if logger is not None:
            logger.print(format_example_fn(x, pred, conf, expect_result, label, isfailed=isfailed))
        # end if
        print(format_example_fn(x, pred, conf, expect_result, label, isfailed=isfailed))

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
        testsuite.summary(
            print_fn=cls.print_result if print_fn is None else print_fn,
            format_example_fn=cls.format_example if format_example_fn is None else format_example_fn,
            logger=logger
        )
        return
