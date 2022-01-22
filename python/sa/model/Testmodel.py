# This script is to test models
# using written testsuites

from typing import *
from pathlib import Path

from checklist.test_suite import TestSuite as suite

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..testsuite.Testsuite import Testsuite
from ..testsuite.Search import DynasentRoundOne

from .Model import Model
from .GoogleModel import GoogleModel

import os


class Testmodel:

    model_func_map = {
        "sa": Model.sentiment_pred_and_conf
    }

    @classmethod
    def load_testsuite(cls, testsuite_file: Path):
        tsuite = suite().from_file(testsuite_file)
        # print(tsuite.info)
        return tsuite

    @classmethod
    def _run_testsuite(cls, task: str, dataset_name: str, local_model_name=None):
        print(f"***** TASK: {task} *****")
        cksum_vals = [
            os.path.basename(test_file).split("_")[-1].split(".")[0]
            for test_file in os.listdir(Macros.result_dir / f"test_results_{task}_{dataset_name}")
            if test_file.startswith(f"{task}_testsuite_seeds_") and test_file.endswith(".pkl")
        ]
        for cksum_val in cksum_vals:
            testsuite_files = [
                Macros.result_dir / f"test_results_{task}_{dataset_name}" / f for f in [
                    f"{task}_testsuite_seeds_{cksum_val}.pkl",
                    f"{task}_testsuite_seed_templates_{cksum_val}.pkl",
                    f"{task}_testsuite_exp_templates_{cksum_val}.pkl"
                ] if os.path.exists(Macros.result_dir / f"test_results_{task}_{dataset_name}" / f)
            ]
            for testsuite_file in testsuite_files:
                testsuite = cls.load_testsuite(testsuite_file)
                if local_model_name is None:
                    # # Run Google nlp model
                    # print(f">>>>> MODEL: Google NLP model")
                    # GoogleModel.run(testsuite, GoogleModel.sentiment_pred_and_conf, n=Macros.nsamples)
                    # print(f"<<<<< MODEL: Google NLP model")
                    
                    for mname, model in Model.load_models(task):
                        print(f">>>>> MODEL: {mname}")
                        Model.run(testsuite, model, cls.model_func_map[task], n=Macros.nsamples)
                        print(f"<<<<< MODEL: {mname}")
                    # end for
                else:
                    print(f">>>>> RETRAINED MODEL: {local_model_name}")
                    model = Model.load_local_model(task, local_model_name)
                    Model.run(testsuite, model, cls.model_func_map[task])
                    print(f"<<<<< RETRAINED MODEL: {local_model_name}")
                # end if
            # end for
        # end for
        print("**********")
        return
    
    @classmethod
    def _run_bl_testsuite(cls, task, bl_name, local_model_name=None):
        print(f"***** TASK: {task} *****")
        print(f"***** Baseline: {bl_name} *****")
        testsuite = cls.load_testsuite(Macros.BASELINES[bl_name]["testsuite_file"])

        if local_model_name is None:
            # Run Google nlp model
            # print(f">>>>> MODEL: Google NLP model")
            # GoogleModel.run(testsuite, GoogleModel.sentiment_pred_and_conf, n=Macros.nsamples)
            # print(f"<<<<< MODEL: Google NLP model")
            
            for mname, model in Model.load_models(task):
                print(f">>>>> MODEL: {mname}")
                Model.run(testsuite, model, cls.model_func_map[task], n=Macros.nsamples)
                print(f"<<<<< MODEL: {mname}")
            # end for
            print("**********")
        else:
            print(f">>>>> RETRAINED MODEL: {local_model_name}")
            model = Model.load_local_model(task, local_model_name)
            Model.run(testsuite, model, cls.model_func_map[task], n=Macros.nsamples)
            print(f"<<<<< RETRAINED MODEL: {local_model_name}")
        # end if
        return

    @classmethod
    def run_testsuite(cls, task: str, dataset_name: str, test_baseline: bool, local_model_name:str = None):
        # run models on checklist introduced testsuite format
        bl_name = None
        if test_baseline:
            cls._run_bl_testsuite(task, "checklist", local_model_name=local_model_name)
        else:
            cls._run_testsuite(task, dataset_name, local_model_name=local_model_name)
        # end if
        return

    @classmethod
    def run_on_diff_dataset(cls, task: str, dataset_name: str, test_type: str = None):
        # run models on other type of dataset
        def run(model, data):
            preds_all, pp_all = list(), list()
            for batch in Model.get_batch(data, 32):
                preds = model(batch)
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
                preds_all.extend(preds)
                pp_all.extend(pp)
            # end for
            return preds_all, pp_all
        
        dataset_name = None
        if test_type is not None:
            dataset_name = test_type
            assert(dataset_name in Macros.datasets[task])
            print(f"***** TASK: {task} *****")
            print(f"***** DATASET: {dataset_name} *****")
            if dataset_name=="dynasent":
                srcs = DynasentRoundOne.get_data(Macros.dyna_r1_test_src_file)
                sents = [s[1] for s in srcs]
                labels = [s[-1] for s in srcs]
                for mname, model in Model.load_models(task):
                    print(f">>>>> MODEL: {mname}")
                    preds, pp = run(model, sents)
                    fail_cnt, fail_rate = Utils.compute_failure_rate(task, preds, labels)
                    print(f"Test cases run:\t{len(preds)}")
                    print(f"Fails (rate):\t{fail_cnt} \({fail_rate}\)")
                    print(f"<<<<< MODEL: {mname}")
                # end for
    
            # end if
        # end if
        return

    
def main(task, dataset_name, test_baseline, test_type, local_model_name=None):
    if local_model_name is None:
        if test_type=="testsuite":
            Testmodel.run_testsuite(task, dataset_name, test_baseline)
        else:
            Testmodel.run_on_diff_dataset(task, dataset_name, test_type=test_type)
        # end if
    else:
        if test_type=="testsuite":
            Testmodel.run_testsuite(task, dataset_name, test_baseline, local_model_name=local_model_name)
        else:
            Testmodel.run_on_diff_dataset(task, dataset_name, test_type=test_type)
        # end if
    return


# if __name__=="__main__":
#     main()
