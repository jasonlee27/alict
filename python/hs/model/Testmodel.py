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
from .GoogleModel import GoogleModel
from ..seed.Search import Hatecheck

import os
import shutil

class Testmodel:

    model_func_map = {
        "hs": Model.sentiment_pred_and_conf
    }

    num_alict_tcs_for_chatgpt_over_lcs = {
        'seed': {
            'Slur usage::Hate expressed using slur': 203,
            'Slur usage::Non-hateful use of slur': 278,
            'Profanity usage::Hate expressed using profanity': 283,
            'Profanity usage::Non-Hateful use of profanity': 306,
            'Pronoun reference::Hate expressed through reference in subsequent clauses': 373,
            'Pronoun reference::Hate expressed through reference in subsequent sentences': 373,
            'Negation::Hate expressed using negated positive statement': 381,
            'Negation::Non-hate expressed using negated hateful statement': 377,
            'Phrasing::Hate phrased as a question': 373,
            'Phrasing::Hate phrased as a opinion': 373,
            'Non-hate grp. ident.::Neutral statements using protected group identifiers': 6,
            'Non-hate grp. ident.::Positive statements using protected group identifiers': 57,
            'Counter speech::Denouncements of hate that quote it': 379,
            'Counter speech::Denouncements of hate that make direct reference to it': 377,
        },
        'exp': {
            'Slur usage::Hate expressed using slur': 290,
            'Slur usage::Non-hateful use of slur': 354,
            'Profanity usage::Hate expressed using profanity': 363,
            'Profanity usage::Non-Hateful use of profanity': 366,
            'Pronoun reference::Hate expressed through reference in subsequent clauses': 381,
            'Pronoun reference::Hate expressed through reference in subsequent sentences': 381,
            'Negation::Hate expressed using negated positive statement': 384,
            'Negation::Non-hate expressed using negated hateful statement': 384,
            'Phrasing::Hate phrased as a question': 383,
            'Phrasing::Hate phrased as a opinion': 383,
            'Non-hate grp. ident.::Neutral statements using protected group identifiers': 12,
            'Non-hate grp. ident.::Positive statements using protected group identifiers': 246,
            'Counter speech::Denouncements of hate that quote it': 384,
            'Counter speech::Denouncements of hate that make direct reference to it': 384,
        }
    }

    num_hatecheck_tcs_for_chatgpt_over_lcs = {
        'Slur usage::Hate expressed using slur': 144,
        'Slur usage::Non-hateful use of slur': 111,
        'Profanity usage::Hate expressed using profanity': 140,
        'Profanity usage::Non-Hateful use of profanity': 100,
        'Pronoun reference::Hate expressed through reference in subsequent clauses': 140,
        'Pronoun reference::Hate expressed through reference in subsequent sentences': 133,
        'Negation::Hate expressed using negated positive statement': 140,
        'Negation::Non-hate expressed using negated hateful statement': 133,
        'Phrasing::Hate phrased as a question': 140,
        'Phrasing::Hate phrased as a opinion': 133,
        'Non-hate grp. ident.::Neutral statements using protected group identifiers': 126,
        'Non-hate grp. ident.::Positive statements using protected group identifiers': 189,
        'Counter speech::Denouncements of hate that quote it': 173,
        'Counter speech::Denouncements of hate that make direct reference to it': 141,
    }

    @classmethod
    def load_testsuite(cls, testsuite_file: Path):
        tsuite = suite().from_file(testsuite_file)
        # print(tsuite.info)
        return tsuite

    @classmethod
    def load_hatecheck_testsuite(cls,
                                 hatecheck_data_file: Path = Macros.hatecheck_data_file,
                                 hatecheck_testsuite_file: Path = Macros.hatecheck_testsuite_file):
        if not os.path.exists(str(Macros.hatecheck_testsuite_file)):
            Hatecheck.write_testsuites(hatecheck_data_file, hatecheck_testsuite_file)
        # end if
        return cls.load_testsuite(hatecheck_testsuite_file)

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
        if bl_name == Macros.datasets[Macros.hs_task][-1]: # hatecheck:
            testsuite = cls.load_hatecheck_testsuite()
        else:
            testsuite = cls.load_testsuite(Macros.BASELINES[bl_name]["testsuite_file"])
        # end if
        
        if local_model_name is None:
            # Run Google nlp model
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
            logger.print("**********")
        else:
            if local_model_name==Macros.openai_chatgpt_engine_name or \
                local_model_name==Macros.openai_chatgpt4_engine_name:
                logger.print(f">>>>> MODEL: {local_model_name}")
                # lc = testsuite.name
                # num_samples = cls.num_checklist_tcs_for_chatgpt_over_lcs[lc]
                Chatgpt.run(
                    testsuite,
                    local_model_name,
                    pred_and_conf_fn=Chatgpt.sentiment_pred_and_conf,
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
        # run models on checklist testsuite format
        if test_baseline:
            bl_name = Macros.datasets[Macros.hs_task][-1]
            cls._run_bl_testsuite(task,
                                  bl_name,
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
            bl_name = Macros.datasets[Macros.hs_task][-1]
            cls._run_bl_testsuite(
                task,
                bl_name,
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
         num_seeds,
         num_trials,
         test_baseline,
         log_file,
         test_seed=False,
         local_model_name=None):
    logger = Logger(logger_file=log_file,
                    logger_name='testmodel')
    _num_trials = '' if num_trials<2 else str(num_trials)
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
            test_result_file = test_result_dir / 'test_results_hatecheck.txt'
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
            test_result_file = test_result_dir / 'test_results_hatecheck.txt'
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
    _num_trials = '' if num_trials<2 else str(num_trials)
    if num_seeds<0:
        test_result_dir = Macros.result_dir/ f"seeds{_num_trials}_{task}_{dataset_name}"
    else:
        test_result_dir = Macros.result_dir/ f"seeds{_num_trials}_{task}_{dataset_name}_{num_seeds}seeds"
    # end if
    baseline_name = Macros.datasets[Macros.hs_task][-1]
    if local_model_name is None:
        Testmodel.run_testsuite(task,
                                dataset_name,
                                selection_method,
                                test_baseline,
                                test_seed,
                                test_result_dir,
                                logger)
        if test_baseline:
            test_result_file = test_result_dir / f"test_results_{baseline_name}.txt"
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
            test_result_file = test_result_dir / f"test_results_{baseline_name}.txt"
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
    log_file,
    test_seed=False,
    local_model_name=None
):
    logger = Logger(
        logger_file=log_file,
        logger_name='testmodel_tosem_chatgpt'
    )
    _num_trials = '' if num_trials<2 else str(num_trials)
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
        test_result_file = test_result_dir / 'test_results_tosem_hatecheck.txt'
    else:
        test_result_file = test_result_dir / 'test_results_tosem.txt'
    # end if
    shutil.copyfile(log_file, test_result_file)
    return


# if __name__=="__main__":
#     main()
