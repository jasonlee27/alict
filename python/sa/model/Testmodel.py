# This script is to test models
# using written testsuites

from typing import *
from pathlib import Path

from checklist.test_suite import TestSuite as suite

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger
from ..testsuite.Testsuite import Testsuite

from .Model import Model
from .GoogleModel import GoogleModel

import os
import shutil

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
    def _run_testsuite(cls,
                       task: str,
                       dataset_name: str,
                       selection_method: str,
                       logger,
                       local_model_name=None):
        logger.print(f"***** TASK: {task} *****")
        cksum_vals = [
            os.path.basename(test_file).split("_")[-1].split(".")[0]
            for test_file in os.listdir(Macros.result_dir / f"test_results_{task}_{dataset_name}_{selection_method}")
            if test_file.startswith(f"{task}_testsuite_seeds_") and test_file.endswith(".pkl")
        ]
        # cksum_vals = [v for v in cksum_vals if v in ['d3af59d', 'a416a87', '22f987a']]
        for cksum_val in cksum_vals:
            # testsuite_files = [
            #     Macros.result_dir / f"test_results_{task}_{dataset_name}" / f for f in [
            #         f"{task}_testsuite_seeds_{cksum_val}.pkl",
            #         f"{task}_testsuite_exps_{cksum_val}.pkl",
            #         f"{task}_testsuite_seed_templates_{cksum_val}.pkl",
            #         f"{task}_testsuite_exp_templates_{cksum_val}.pkl"
            #     ] if os.path.exists(Macros.result_dir / f"test_results_{task}_{dataset_name}" / f)
            # ]
            testsuite_files = [
                Macros.result_dir / f"test_results_{task}_{dataset_name}_{selection_method}" / f for f in [
                    f"{task}_testsuite_seeds_{cksum_val}.pkl",
                    f"{task}_testsuite_exps_{cksum_val}.pkl",
                ] if os.path.exists(Macros.result_dir / f"test_results_{task}_{dataset_name}_{selection_method}" / f)
            ]
            for testsuite_file in testsuite_files:
                testsuite = cls.load_testsuite(testsuite_file)
                if local_model_name is None:
                    # # Run Google nlp model
                    # print(f">>>>> MODEL: Google NLP model")
                    # GoogleModel.run(testsuite, GoogleModel.sentiment_pred_and_conf, n=Macros.nsamples)
                    # print(f"<<<<< MODEL: Google NLP model")
                    
                    for mname, model in Model.load_models(task):
                        logger.print(f">>>>> MODEL: {mname}")
                        Model.run(testsuite, model, cls.model_func_map[task], n=Macros.nsamples, logger=logger)
                        logger.print(f"<<<<< MODEL: {mname}")
                    # end for
                else:
                    logger.print(f">>>>> RETRAINED MODEL: {local_model_name}")
                    model = Model.load_local_model(task, local_model_name)
                    Model.run(testsuite, model, cls.model_func_map[task], logger=logger)
                    logger.print(f"<<<<< RETRAINED MODEL: {local_model_name}")
                # end if
            # end for
        # end for
        logger.print('**********')
        return

    @classmethod
    def _run_seed_testsuite(cls,
                            task: str,
                            dataset_name: str,
                            logger,
                            local_model_name=None):
        logger.print(f"***** TASK: {task} *****")
        res_dir = Macros.result_dir/ f"seeds_{task}_{dataset_name}"
        cksum_vals = [
            os.path.basename(test_file).split("_")[-1].split(".")[0]
            for test_file in os.listdir(res_dir)
            if test_file.startswith(f"{task}_testsuite_seeds_") and test_file.endswith(".pkl")
        ]
        for cksum_val in cksum_vals:
            testsuite_file = res_dir / f"{task}_testsuite_seeds_{cksum_val}.pkl"
            testsuite = cls.load_testsuite(testsuite_file)
            if local_model_name is None:
                # # Run Google nlp model
                # print(f">>>>> MODEL: Google NLP model")
                # GoogleModel.run(testsuite, GoogleModel.sentiment_pred_and_conf, n=Macros.nsamples)
                # print(f"<<<<< MODEL: Google NLP model")
                
                for mname, model in Model.load_models(task):
                    logger.print(f">>>>> MODEL: {mname}")
                    Model.run(testsuite, model, cls.model_func_map[task], n=Macros.nsamples, logger=logger)
                    logger.print(f"<<<<< MODEL: {mname}")
                # end for
            else:
                logger.print(f">>>>> RETRAINED MODEL: {local_model_name}")
                model = Model.load_local_model(task, local_model_name)
                Model.run(testsuite, model, cls.model_func_map[task], logger=logger)
                logger.print(f"<<<<< RETRAINED MODEL: {local_model_name}")
            # end if
        # end for
        logger.print('**********')
        return
    
    @classmethod
    def _run_bl_testsuite(cls,
                          task,
                          bl_name,
                          logger,
                          local_model_name=None):
        logger.print(f"***** TASK: {task} *****")
        logger.print(f"***** Baseline: {bl_name} *****")
        testsuite = cls.load_testsuite(Macros.BASELINES[bl_name]["testsuite_file"])

        if local_model_name is None:
            # Run Google nlp model
            # print(f">>>>> MODEL: Google NLP model")
            # GoogleModel.run(testsuite, GoogleModel.sentiment_pred_and_conf, n=Macros.nsamples)
            # print(f"<<<<< MODEL: Google NLP model")
            
            for mname, model in Model.load_models(task):
                logger.print(f">>>>> MODEL: {mname}")
                Model.run(testsuite, model, cls.model_func_map[task], n=Macros.nsamples, logger=logger)
                logger.print(f"<<<<< MODEL: {mname}")
            # end for
            logger.print("**********")
        else:
            logger.print(f">>>>> RETRAINED MODEL: {local_model_name}")
            model = Model.load_local_model(task, local_model_name)
            Model.run(testsuite, model, cls.model_func_map[task], n=Macros.nsamples, logger=logger)
            logger.print(f"<<<<< RETRAINED MODEL: {local_model_name}")
        # end if
        return

    @classmethod
    def run_testsuite(cls,
                      task: str,
                      dataset_name: str,
                      selection_method: str,
                      test_baseline: bool,
                      test_seed: bool,
                      logger,
                      local_model_name: str = None):
        # run models on checklist introduced testsuite format
        bl_name = None
        if test_baseline:
            cls._run_bl_testsuite(task,
                                  'checklist',
                                  logger,
                                  local_model_name=local_model_name)
        elif test_baseline==False and test_seed:
            cls._run_seed_testsuite(task,
                                    dataset_name,
                                    logger,
                                    local_model_name=local_model_name)
        elif test_baseline==False and test_seed==False:
            cls._run_testsuite(task,
                               dataset_name,
                               selection_method,
                               logger,
                               local_model_name=local_model_name)
        # end if
        return

    @classmethod
    def run_on_diff_dataset(cls, task: str, dataset_name: str, test_type: str, logger):
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
            logger.print(f"***** TASK: {task} *****")
            logger.print(f"***** DATASET: {dataset_name} *****")
            if dataset_name=="dynasent":
                srcs = DynasentRoundOne.get_data(Macros.dyna_r1_test_src_file)
                sents = [s[1] for s in srcs]
                labels = [s[-1] for s in srcs]
                for mname, model in Model.load_models(task):
                    logger.print(f">>>>> MODEL: {mname}")
                    preds, pp = run(model, sents)
                    fail_cnt, fail_rate = Utils.compute_failure_rate(task, preds, labels)
                    logger.print(f"Test cases run:\t{len(preds)}")
                    logger.print(f"Fails (rate):\t{fail_cnt} \({fail_rate}\)")
                    logger.print(f"<<<<< MODEL: {mname}")
                # end for
    
            # end if
        # end if
        return


def main(task,
         dataset_name,
         selection_method,
         test_baseline,
         test_type,
         log_file,
         test_seed=False,
         local_model_name=None):
    logger = Logger(logger_file=log_file,
                    logger_name='testmodel')
    test_result_dir = Macros.result_dir / f"test_results_{task}_{dataset_name}_{selection_method}"
    if local_model_name is None:
        Testmodel.run_testsuite(task,
                                dataset_name,
                                selection_method,
                                test_baseline,
                                test_seed,
                                logger)
        if test_baseline:
            test_result_file = test_result_dir / 'test_results_checklist.txt'
        else:
            test_result_file = test_result_dir / 'test_results.txt'
        # end if
        shutil.copyfile(log_file, test_result_file)
        # if test_type=="testsuite":
        #     Testmodel.run_testsuite(task, dataset_name, selection_method, test_baseline, logger)
        #     shutil.copyfile(log_file, 'file2.txt')
        # else:
        #     Testmodel.run_on_diff_dataset(task, dataset_name, selection_method, test_type=test_type, logger=logger)
        # # end if
    else:
        Testmodel.run_testsuite(task,
                                dataset_name,
                                selection_method,
                                test_baseline,
                                test_seed,
                                logger,
                                local_model_name=local_model_name)
        if test_baseline:
            test_result_file = test_result_dir / 'test_results_checklist.txt'
        else:
            test_result_file = test_result_dir / 'test_results.txt'
        # end if
        shutil.copyfile(log_file, test_result_file)
        # if test_type=="testsuite":
        #     Testmodel.run_testsuite(task, dataset_name, selection_method, test_baseline, local_model_name=local_model_name, logger)
        #     shutil.copyfile(log_file, 'file2.txt')
        # else:
        #     Testmodel.run_on_diff_dataset(task, dataset_name, test_type, logger)
        # # end if
    # end if
    
    return

def main_seed(task,
              dataset_name,
              selection_method,
              test_baseline,
              log_file,
              test_seed=True,
              test_type=True,
              local_model_name=None):
    logger = Logger(logger_file=log_file,
                    logger_name='testseed')
    test_result_dir = Macros.result_dir/ f"seeds_{task}_{dataset_name}"
    if local_model_name is None:
        Testmodel.run_testsuite(task,
                                dataset_name,
                                selection_method,
                                test_baseline,
                                test_seed,
                                logger)
        if test_baseline:
            test_result_file = test_result_dir / 'test_results_checklist.txt'
        else:
            test_result_file = test_result_dir / 'test_results.txt'
        # end if
        shutil.copyfile(log_file, test_result_file)
    else:
        Testmodel.run_testsuite(task,
                                dataset_name,
                                selection_method,
                                test_baseline,
                                test_seed,
                                logger,
                                local_model_name=local_model_name)
        if test_baseline:
            test_result_file = test_result_dir / 'test_results_checklist.txt'
        else:
            test_result_file = test_result_dir / 'test_results.txt'
        # end if
        shutil.copyfile(log_file, test_result_file)
    # end if
    return


# if __name__=="__main__":
#     main()
