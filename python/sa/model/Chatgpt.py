
import os
import openai
import numpy as np

from typing import *
from pathlib import Path

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger


openai.api_key = os.environ["OPENAI_API_KEY"]

class Chatgpt:

    engine_name = Macros.openai_chatgpt_engine_name # model name

    @classmethod
    def set_model_name(
        cls, 
        engine_name
    ) -> None:
        cls.engine_name = engine_name
        return

    @classmethod
    def predict(
        cls, 
        query_text: str,
        prompt: str='',
        prompt_append: bool=False
    ) -> str:
        '''
        TODO: if i need to add period(.) between prompt and query text? does it make significant change on output?
        '''
        text = f"{prompt} {query_text}" 
        if prompt_append:
            text = f"{query_text} {prompt}"
        # end if
        response = openai.Completion.create(
            engine=cls.engine_name,
            prompt=text,
            temperature=Macros.resp_temp,
            max_tokens=Macros.llm_resp_max_len
        )
        # print(text)
        # print(response["choices"][0]["text"])
        # print()
        return response["choices"][0]["text"]

    @classmethod
    def get_batch(cls, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
        # end for

    @classmethod
    def batch_predict(cls, data: List[str], batch_size: int = 32):
        preds = list()
        for batch in cls.get_batch(data, batch_size):
            for d in batch:
                response = cls.predict(
                    query_text=d,
                    prompt=Macros.openai_chatgpt_sa_prompt,
                    prompt_append=False
                ) 
                label = response.split('the sentiment is ')[-1].strip().upper()
                preds.append({
                    'response': response,
                    'label': label
                })
            # end for
        # end for
        return preds

    @classmethod
    def sentiment_pred_and_conf(cls, data: List[str]):
        # score of the sentiment ranges between -1.0(negative) and 1.0(positive)
        # First, score in [-1.0, 1.0] is normalized into the range of [0,1] 
        preds = cls.batch_predict(data)
        preds_index = list()
        pp = list()
        for p in preds:
            if p['label']=='POSITIVE':
                preds_index.append(2)
                pp.append([0.,0.,1.])
            elif p['label']=='NEGATIVE':
                preds_index.append(0)
                pp.append([1.,0.,0.])
            else:
                preds_index.append(1)
                pp.append([0.,1.,0.])
            # end if
        # end for
        preds = np.array(preds_index)
        pp = np.array(pp)
        return preds, pp

    @classmethod
    def format_example(
        cls, 
        x, 
        pred, 
        conf, 
        *args, 
        **kwargs
    ):
        softmax = type(conf) in [np.array, np.ndarray]
        pred_res = "FAIL" if kwargs["isfailed"] else "PASS"
        expect_result, label = args[0], args[1]
        binary = False
        if softmax:
            if conf.shape[0] == 2:
                conf = conf[1]
                return f"DATA::{pred_res}::{conf:%.1f}::{str(pred)}::{str(label)}::{str(x)}"
            elif conf.shape[0] <= 4:
                confs = ' '.join(['%.1f' % c for c in conf])
                # return f"{pred_res}::{conf}::{str(x)}"
                return f"DATA::{pred_res}::{str(confs)}::{str(pred)}::{str(label)}::{str(x)}"
            else:
                conf = conf[pred]
                # return f"{pred_res}::{pred}:({conf:%.1f})::{str(x)}"
                return f"DATA::{pred_res}::{conf:%.1f}::{str(pred)}::{str(label)}::{str(x)}"
        else:
            return f"DATA::{pred_res}::[]::{str(pred)}::{str(label)}::{str(x)}"
        
    @classmethod
    def print_result(
        cls, 
        x, 
        pred, 
        conf, 
        expect_result, 
        label=None,
        meta=None, 
        format_example_fn=None,
        nsamples=None,
        logger=None
    ):
        isfailed = expect_result[0]!=True
        if logger is None:
            print(format_example_fn(x, pred, conf, expect_result, label, isfailed=isfailed))
        else:
            res_str = format_example_fn(x, pred, conf, expect_result, label, isfailed=isfailed)
            print(res_str)
            logger.print(res_str)
        # end if
    
    @classmethod
    def run(
        cls,
        testsuite,
        model_name,
        pred_and_conf_fn,
        print_fn=None,
        format_example_fn=None,
        n=Macros.nsamples,
        logger=None
    ):
        cls.set_model_name(
            engine_name=model_name
        )
        testsuite.run(
            pred_and_conf_fn, 
            n=n, 
            overwrite=True, 
            logger=logger
        )
        testsuite.summary(
            logger=logger,
            print_fn=cls.print_result if print_fn is None else print_fn,
            format_example_fn=cls.format_example if format_example_fn is None else format_example_fn
        )
        return
        
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
#     )
#     parser.add_argument(
#         "movie_review_filename",
#         help="The filename of the movie review you'd like to analyze.",
#     )
#     args = parser.parse_args()

#     analyze(args.movie_review_filename)
