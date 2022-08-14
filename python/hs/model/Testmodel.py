# This script is to test models
# using written testsuites

from typing import *
from pathlib import Path

from checklist.editor import Editor
from checklist.test_suite import TestSuite # as suite
from checklist.test_types import MFT #, INV, DIR

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger
# from ..testsuite.Testsuite import Testsuite
from ..testsuite.Search import Hatecheck

from .Model import Model
from .GoogleModel import GoogleModel

import os
import shutil

class Testmodel:

    model_func_map = {
        "hs": Model.hatespeech_pred_and_conf
    }

    @classmethod
    def generate_baseline_testsuite(cls, save_dir: Path):
        sents = Hatecheck.get_sents()
        # save_dir = Macros.result_dir / 'templates_hs_hatecheck_baseline'
        for key, val in Hatecheck.FUNCTIONALITY_MAP.items():
            test_cksum = Utils.get_cksum(key)
            if not os.path.exists(str(save_dir / f'testsuite_{test_cksum}.pkl')):
                t = None
                suite = TestSuite()
                editor = Editor()
                for s in sents:
                    if s['func']==val:
                        if t is None:
                            t = editor.template(s['sent'],
                                                labels=Macros.hs_label_map[s['label']],
                                                save=True)
                        elif t is not None and len(t.data)<Macros.max_num_sents:
                            t += editor.template(s["sent"],
                                                 labels=Macros.hs_label_map[s['label']],
                                                 save=True)
                        # end if
                    # end if
                # end for
                if t is not None:
                    test = MFT(**t)
                    suite.add(test,
                              name=f"hs::SEED::{val}",
                              capability=val.split('_')[0],
                              description=key)
                    num_data = sum([len(suite.tests[k].data) for k in suite.tests.keys()])
                    if num_data>0:
                        suite.save(save_dir / f'testsuite_{test_cksum}.pkl')
                    # end if
                # end if
            # end if
        # end for
        return

    @classmethod
    def load_testsuite(cls, testsuite_file: Path):
        tsuite = TestSuite().from_file(testsuite_file)
        # print(tsuite.info)
        return tsuite

    @classmethod
    def _run_testsuite(cls, task: str, dataset_name: str, selection_method: str, logger, local_model_name=None):
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
                # end if
            # end for
        # end for
        logger.print('**********')
        return
    
    @classmethod
    def _run_bl_testsuite(cls, task, bl_name, logger, local_model_name=None):
        logger.print(f"***** TASK: {task} *****")
        logger.print(f"***** Baseline: {bl_name} *****")
        testsuite_dir = Macros.result_dir / 'templates_hs_hatecheck_baseline'
        testsuite_dir.mkdir(parents=True, exist_ok=True)
        cksum_vals = [
            os.path.basename(test_file).split("_")[-1].split(".")[0]
            for test_file in os.listdir(testsuite_dir)
            if test_file.startswith('testsuite_') and test_file.endswith(".pkl")
        ]
        if not any(cksum_vals):
            cls.generate_baseline_testsuite(testsuite_dir)
            cksum_vals = [
                os.path.basename(test_file).split("_")[-1].split(".")[0]
                for test_file in os.listdir(testsuite_dir)
                if test_file.startswith('testsuite_') and test_file.endswith(".pkl")
            ]
        # end if
        for cksum_val in cksum_vals:
            testsuite = cls.load_testsuite(
                testsuite_dir / f'testsuite_{cksum_val}.pkl'
            )
            for mname, model in Model.load_models(task):
                logger.print(f">>>>> MODEL: {mname}")
                Model.run(testsuite, model, cls.model_func_map[task], n=Macros.nsamples, logger=logger)
                logger.print(f"<<<<< MODEL: {mname}")
            # end for
            logger.print("**********")
        # end if
        return
    
    @classmethod
    def run_testsuite(cls, task: str, dataset_name: str, selection_method: str, test_baseline: bool, logger, local_model_name: str = None):
        # run models on checklist introduced testsuite format
        bl_name = None
        if test_baseline:
            cls._run_bl_testsuite(task, "hatecheck", logger, local_model_name=local_model_name)
        else:
            cls._run_testsuite(task, dataset_name, selection_method, logger, local_model_name=local_model_name)
        # end if
        return

    
def main(task, dataset_name, selection_method, test_baseline, log_file, local_model_name=None):
    logger = Logger(logger_file=log_file,
                    logger_name='testmodel')
    test_result_dir = Macros.result_dir / f"test_results_{task}_{dataset_name}_{selection_method}"
    if local_model_name is None:
        Testmodel.run_testsuite(task, dataset_name, selection_method, test_baseline, logger)
        if test_baseline:
            test_result_file = test_result_dir / 'test_results_hatecheck.txt'
        else:
            test_result_file = test_result_dir / 'test_results.txt'
        # end if
        shutil.copyfile(log_file, test_result_file)
    else:
        Testmodel.run_testsuite(task, dataset_name, selection_method, test_baseline, logger, local_model_name=local_model_name)
        if test_baseline:
            test_result_file = test_result_dir / 'test_results_hatecheck.txt'
        else:
            test_result_file = test_result_dir / 'test_results.txt'
        # end if
        shutil.copyfile(log_file, test_result_file)
    # end if
    
    return


# if __name__=="__main__":
#     main()
