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
from .Chatgpt import Chatgpt
# from .GoogleModel import GoogleModel

import os
import shutil

class Testmodel:

    model_func_map = {
        "sa": Model.sentiment_pred_and_conf
    }

    num_alict_tcs_for_chatgpt_over_lcs = {
        'Short sentences with neutral adjectives and nouns': 19,
        'Short sentences with sentiment-laden adjectives': 160,
        'Sentiment change over time, present should prevail': 383,
        'Negated negative should be positive or neutral': 67,
        'Negated neutral should still be neutral': 26,
        'Negated positive with neutral content in the middle': 379,
        'Negation of negative at the end, should be positive or neutral': 377,
        'Author sentiment is more important than of others': 383,
        'parsing sentiment in (question, yes) form': 375,
        'Parsing sentiment in (question, no) form': 375
    }

    num_checklist_tcs_for_chatgpt_over_lcs = {
        'Sentiment-laden words in context': 368,
        'neutral words in context': 315,
        'used to, but now': 367,
        'simple negations: not negative': 364,
        'simple negations: not neutral is still neutral': 334,
        'Hard: Negation of positive with neutral stuff in the middle (should be negative)': 278,
        'simple negations: I thought x was negative, but it was not (should be neutral or positive)': 326,
        'my opinion is what matters': 368,
        'Q & A: yes': 366,
        'Q & A: no': 366
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
                       test_result_dir: Path,
                       logger,
                       local_model_name=None):
        logger.print(f"***** TASK: {task} *****")
        cksum_vals = [
            os.path.basename(test_file).split("_")[-1].split(".")[0]
            for test_file in os.listdir(test_result_dir)
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
            print(cksum_val)
            testsuite_files = [
                test_result_dir / f for f in [
                    f"{task}_testsuite_seeds_{cksum_val}.pkl",
                    f"{task}_testsuite_exps_{cksum_val}.pkl",
                ] if os.path.exists(test_result_dir / f)
            ]
            if local_model_name==Macros.openai_chatgpt_engine_name or \
                local_model_name==Macros.openai_chatgpt4_engine_name:
                testsuite_files = [
                    test_result_dir / f for f in [
                        f"{task}_testsuite_tosem_seeds_{cksum_val}.pkl",
                        f"{task}_testsuite_tosem_exps_{cksum_val}.pkl",
                    ] if os.path.exists(test_result_dir / f)
                ]
            # end if
            for testsuite_file in testsuite_files:
                testsuite = cls.load_testsuite(testsuite_file)
                if local_model_name is None:
                    # # Run Google nlp model
                    # print(f">>>>> MODEL: Google NLP model")
                    # GoogleModel.run(testsuite, GoogleModel.sentiment_pred_and_conf, n=Macros.nsamples)
                    # print(f"<<<<< MODEL: Google NLP model")
                    for mname, model in Model.load_models(task):
                        logger.print(f">>>>> MODEL: {mname}")
                        Model.run(testsuite,
                                model,
                                cls.model_func_map[task],
                                logger=logger)
                        logger.print(f"<<<<< MODEL: {mname}")
                    # end for
                else:
                    if local_model_name==Macros.openai_chatgpt_engine_name or \
                        local_model_name==Macros.openai_chatgpt4_engine_name:
                        logger.print(f">>>>> MODEL: {local_model_name}")
                        # lc = list(testsuite.tests.keys())[0].split('::')[-1]
                        # num_samples = cls.num_alict_tcs_for_chatgpt_over_lcs[lc]
                        Chatgpt.run(
                            testsuite,
                            local_model_name,
                            Chatgpt.sentiment_pred_and_conf,
                            n=None,
                            logger=logger
                        )
                        logger.print(f"<<<<< MODEL: {local_model_name}")
                    else:
                        logger.print(f">>>>> RETRAINED MODEL: {local_model_name}")
                        model = Model.load_local_model(task, local_model_name)
                        Model.run(testsuite,
                                model,
                                cls.model_func_map[task],
                                logger=logger)
                        logger.print(f"<<<<< RETRAINED MODEL: {local_model_name}")
                    # end if
                # end if
            # end for
        # end for
        logger.print('**********')
        return

    @classmethod
    def _run_testsuite_fairness(
        cls,
        task: str,
        dataset_name: str,
        selection_method: str,
        test_result_dir: Path,
        logger,
        local_model_name=None
    ):
        logger.print(f"***** TASK: {task} *****")
        cksum_vals = [
            os.path.basename(test_file).split("_")[-1].split(".")[0]
            for test_file in os.listdir(test_result_dir)
            if test_file.startswith(f"{task}_testsuite_fairness_seeds_") and test_file.endswith(".pkl")
        ]
        # cksum_vals = [v for v in cksum_vals if v in ['d3af59d', 'a416a87', '22f987a']]
        for cksum_val in cksum_vals:
            testsuite_files = [
                test_result_dir / f for f in [
                    f"{task}_testsuite_fairness_seeds_{cksum_val}.pkl",
                    f"{task}_testsuite_fairness_exps_{cksum_val}.pkl",
                ] if os.path.exists(test_result_dir / f)
            ]
            for testsuite_file in testsuite_files:
                testsuite = cls.load_testsuite(testsuite_file)
                if local_model_name is None:
                    for mname, model in Model.load_models(task):
                        logger.print(f">>>>> MODEL: {mname}")
                        Model.run(
                            testsuite,
                            model,
                            cls.model_func_map[task],
                            logger=logger
                        )
                        logger.print(f"<<<<< MODEL: {mname}")
                    # end for
                else:
                    logger.print(f">>>>> RETRAINED MODEL: {local_model_name}")
                    model = Model.load_local_model(task, local_model_name)
                    Model.run(
                        testsuite,
                        model,
                        cls.model_func_map[task],
                        logger=logger
                    )
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
                            test_result_dir: Path,
                            logger,
                            local_model_name=None):
        logger.print(f"***** TASK: {task} *****")
        # res_dir = Macros.result_dir / f"seeds_{task}_{dataset_name}"
        cksum_vals = [
            os.path.basename(test_file).split("_")[-1].split(".")[0]
            for test_file in os.listdir(test_result_dir)
            if test_file.startswith(f"{task}_testsuite_seeds_") and test_file.endswith(".pkl")
        ]
        for cksum_val in cksum_vals:
            testsuite_file = test_result_dir / f"{task}_testsuite_seeds_{cksum_val}.pkl"
            testsuite = cls.load_testsuite(testsuite_file)
            if local_model_name is None:
                # # Run Google nlp model
                # print(f">>>>> MODEL: Google NLP model")
                # GoogleModel.run(testsuite, GoogleModel.sentiment_pred_and_conf, n=Macros.nsamples)
                # print(f"<<<<< MODEL: Google NLP model")
                for mname, model in Model.load_models(task):
                    logger.print(f">>>>> MODEL: {mname}")
                    Model.run(testsuite,
                            model,
                            cls.model_func_map[task],
                            logger=logger)
                    logger.print(f"<<<<< MODEL: {mname}")
                # end for
            else:
                if local_model_name==Macros.openai_chatgpt_engine_name or \
                    local_model_name==Macros.openai_chatgpt4_engine_name:
                    testsuite_file = test_result_dir / f"{task}_testsuite_tosem_seeds_{cksum_val}.pkl"
                    testsuite = cls.load_testsuite(testsuite_file)  
                    logger.print(f">>>>> MODEL: {local_model_name}")
                    # lc = testsuite.name.split('::')[-1]
                    # num_samples = cls.num_alict_tcs_for_chatgpt_over_lcs['seed'][lc]
                    Chatgpt.run(
                        testsuite,
                        local_model_name,
                        Chatgpt.sentiment_pred_and_conf,
                        n=None,
                        logger=logger
                    )
                    logger.print(f"<<<<< MODEL: {local_model_name}")
                else:
                    logger.print(f">>>>> RETRAINED MODEL: {local_model_name}")
                    model = Model.load_local_model(task, local_model_name)
                    Model.run(testsuite,
                            model,
                            cls.model_func_map[task],
                            logger=logger)
                    logger.print(f"<<<<< RETRAINED MODEL: {local_model_name}")
                # end if
            # end if
        # end for
        logger.print('**********')
        return
    
    @classmethod
    def _run_bl_testsuite(cls,
                          task,
                          bl_name,
                          test_result_dir,
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
                Model.run(
                    testsuite,
                    model,
                    cls.model_func_map[task],
                    logger=logger
                )
                logger.print(f"<<<<< MODEL: {mname}")
            # end for
            logger.print("**********")
        else:
            if local_model_name==Macros.openai_chatgpt_engine_name or \
                local_model_name==Macros.openai_chatgpt4_engine_name:
                logger.print(f">>>>> MODEL: {local_model_name}")
                num_samples = 368
                tests_to_be_del = list()
                for test in testsuite.tests:
                    if test not in cls.num_checklist_tcs_for_chatgpt_over_lcs.keys():
                        tests_to_be_del.append(test)
                    # end if
                # end for
                for t in tests_to_be_del:
                    del testsuite.tests[t]
                # end if
                Chatgpt.run(
                    testsuite,
                    local_model_name,
                    pred_and_conf_fn=Chatgpt.sentiment_pred_and_conf,
                    n=num_samples,
                    logger=logger
                )
                logger.print(f"<<<<< MODEL: {local_model_name}")
            else:
                logger.print(f">>>>> RETRAINED MODEL: {local_model_name}")
                model = Model.load_local_model(task, local_model_name)
                Model.run(testsuite,
                        model,
                        cls.model_func_map[task],
                        logger=logger)
                logger.print(f"<<<<< RETRAINED MODEL: {local_model_name}")
            # end if
        # end if
        return

    @classmethod
    def _run_bl_testsuite_fairness(
        cls,
        task,
        bl_name,
        test_result_dir,
        logger,
        local_model_name=None
    ):
        logger.print(f"***** TASK: {task} *****")
        logger.print(f"***** Baseline: {bl_name} *****")
        _bl_name = ''
        if bl_name=='checklist':
            _bl_name = 'checklist_fairness'
        # end if
        testsuite = cls.load_testsuite(Macros.BASELINES[_bl_name]["testsuite_file"])

        if local_model_name is None:
            for mname, model in Model.load_models(task):
                logger.print(f">>>>> MODEL: {mname}")
                Model.run(
                    testsuite,
                    model,
                    cls.model_func_map[task],
                    logger=logger
                )
                logger.print(f"<<<<< MODEL: {mname}")
            # end for
            logger.print("**********")
        else:
            logger.print(f">>>>> RETRAINED MODEL: {local_model_name}")
            model = Model.load_local_model(task, local_model_name)
            Model.run(testsuite,
                    model,
                    cls.model_func_map[task],
                    logger=logger)
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
                      test_result_dir: Path,
                      logger,
                      local_model_name: str = None):
        # run models on checklist introduced testsuite format
        bl_name = None
        if test_baseline:
            cls._run_bl_testsuite(task,
                                  'checklist',
                                  test_result_dir,
                                  logger,
                                  local_model_name=local_model_name)
        elif test_baseline==False and test_seed:
            cls._run_seed_testsuite(task,
                                    dataset_name,
                                    test_result_dir,
                                    logger,
                                    local_model_name=local_model_name)
        elif test_baseline==False and test_seed==False:
            cls._run_testsuite(task,
                               dataset_name,
                               selection_method,
                               test_result_dir,
                               logger,
                               local_model_name=local_model_name)
        # end if
        return

    @classmethod
    def run_testsuite_on_chatgpt(
        cls,
        task: str,
        dataset_name: str,
        selection_method: str,
        test_baseline: bool,
        test_seed: bool,
        test_result_dir: Path,
        logger,
        chatgpt_model_name: str
    ):
        # run models on checklist introduced testsuite format
        bl_name = None
        if test_baseline:
            cls._run_bl_testsuite(
                task,
                'checklist',
                test_result_dir,
                logger,
                local_model_name=chatgpt_model_name
            )
        elif test_baseline==False and test_seed:
            cls._run_seed_testsuite(
                task,
                dataset_name,
                test_result_dir,
                logger,
                local_model_name=chatgpt_model_name
            )
        elif test_baseline==False and test_seed==False:
            cls._run_testsuite(
                task,
                dataset_name,
                selection_method,
                test_result_dir,
                logger,
                local_model_name=chatgpt_model_name
            )
        # end if
        return

    @classmethod
    def run_testsuite_fairness(
        cls,
        task: str,
        dataset_name: str,
        selection_method: str,
        test_baseline: bool,
        test_seed: bool,
        test_result_dir: Path,
        logger,
        local_model_name: str = None
    ):
        # run models on checklist introduced testsuite format
        bl_name = None
        if test_baseline:
            cls._run_bl_testsuite_fairness(
                task,
                'checklist',
                test_result_dir,
                logger,
                local_model_name=local_model_name
            )
        elif test_baseline==False and test_seed:
            cls._run_seed_testsuite(
                task,
                dataset_name,
                test_result_dir,
                logger,
                local_model_name=local_model_name
            )
        elif test_baseline==False and test_seed==False:
            cls._run_testsuite_fairness(
                task,
                dataset_name,
                selection_method,
                test_result_dir,
                logger,
                local_model_name=local_model_name
            )
        # end if
        return

    @classmethod
    def run_on_diff_dataset(cls, task: str, dataset_name: str, test_type: str, logger):
        # run models on other type of dataset
        def run(model, data):
            preds_all, pp_all = list(), list()
            batch_size = 16
            for batch in Model.get_batch(data, batch_size):
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
         num_seeds,
         num_trials,
         test_baseline,
         test_type,
         log_file,
         test_seed=False,
         local_model_name=None):
    logger = Logger(logger_file=log_file,
                    logger_name='testmodel')
    _num_trials = '' if num_trials==1 else str(num_trials)
    if num_seeds<0:
        test_result_dir = Macros.result_dir/ f"test_results{_num_trials}_{task}_{dataset_name}_{selection_method}"
    else:
        test_result_dir = Macros.result_dir/ f"test_results{_num_trials}_{task}_{dataset_name}_{selection_method}_{num_seeds}seeds"
    # end if
    if local_model_name is None:
        Testmodel.run_testsuite(task,
                                dataset_name,
                                selection_method,
                                test_baseline,
                                test_seed,
                                test_result_dir,
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
                                test_result_dir,
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
              num_seeds,
              num_trials,
              test_baseline,
              log_file,
              test_seed=True,
              test_type=True,
              local_model_name=None):
    logger = Logger(logger_file=log_file,
                    logger_name='testseed')
    _num_trials = '' if num_trials==1 else str(num_trials)
    if num_seeds<0:
        test_result_dir = Macros.result_dir/ f"seeds{_num_trials}_{task}_{dataset_name}"
    else:
        test_result_dir = Macros.result_dir/ f"seeds{_num_trials}_{task}_{dataset_name}_{num_seeds}seeds"
    # end if
    if local_model_name is None:
        Testmodel.run_testsuite(task,
                                dataset_name,
                                selection_method,
                                test_baseline,
                                test_seed,
                                test_result_dir,
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
                                test_result_dir,
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

def main_tosem(
    task,
    dataset_name,
    selection_method,
    num_seeds,
    num_trials,
    test_baseline,
    test_type,
    log_file,
    test_seed=False,
    local_model_name=None
):
    logger = Logger(logger_file=log_file,
                    logger_name='testmodel_tosem_chatgpt')
    _num_trials = '' if num_trials==1 else str(num_trials)
    if num_seeds<0:
        test_result_dir = Macros.result_dir/ f"test_results{_num_trials}_{task}_{dataset_name}_{selection_method}"
    else:
        test_result_dir = Macros.result_dir/ f"test_results{_num_trials}_{task}_{dataset_name}_{selection_method}_{num_seeds}seeds"
    # end if
    Testmodel.run_testsuite_on_chatgpt(
        task,
        dataset_name,
        selection_method,
        test_baseline,
        test_seed,
        test_result_dir,
        logger,
        local_model_name
    )
    if test_baseline:
        test_result_file = test_result_dir / 'test_results_tosem_checklist.txt'
    else:
        test_result_file = test_result_dir / 'test_results_tosem.txt'
    # end if
    shutil.copyfile(log_file, test_result_file)
    # if test_type=="testsuite":
    #     Testmodel.run_testsuite(task, dataset_name, selection_method, test_baseline, logger)
    #     shutil.copyfile(log_file, 'file2.txt')
    # else:
    #     Testmodel.run_on_diff_dataset(task, dataset_name, selection_method, test_type=test_type, logger=logger)
    # # end if
    
    return

def main_fairness(
    task,
    dataset_name,
    selection_method,
    num_seeds,
    num_trials,
    test_baseline,
    test_type,
    log_file,
    test_seed=False,
    local_model_name=None
):
    logger = Logger(logger_file=log_file,
                    logger_name='testmodel_fairness')
    _num_trials = '' if num_trials==1 else str(num_trials)
    if num_seeds<0:
        test_result_dir = Macros.result_dir/ f"test_results{_num_trials}_{task}_{dataset_name}_{selection_method}_for_fairness"
    else:
        test_result_dir = Macros.result_dir/ f"test_results{_num_trials}_{task}_{dataset_name}_{selection_method}_{num_seeds}seeds_for_fairness"
    # end if
    Testmodel.run_testsuite_fairness(
        task,
        dataset_name,
        selection_method,
        test_baseline,
        test_seed,
        test_result_dir,
        logger,
        local_model_name
    )
    if test_baseline:
        test_result_file = test_result_dir / 'test_results_fairness_checklist.txt'
    else:
        test_result_file = test_result_dir / 'test_results_fairness.txt'
    # end if
    shutil.copyfile(log_file, test_result_file)
    # if test_type=="testsuite":
    #     Testmodel.run_testsuite(task, dataset_name, selection_method, test_baseline, logger)
    #     shutil.copyfile(log_file, 'file2.txt')
    # else:
    #     Testmodel.run_on_diff_dataset(task, dataset_name, selection_method, test_type=test_type, logger=logger)
    # # end if
    
    return



# if __name__=="__main__":
#     main()
