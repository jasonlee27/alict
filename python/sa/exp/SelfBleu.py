# experiment for selfbleu score
# A higher Self-BLEU score implies less diversity of the document

import os
import nltk
import time
import multiprocessing

from tqdm import tqdm
from multiprocessing import Pool
# from functools import partial
from nltk.translate.bleu_score import SmoothingFunction
from ..testsuite.Search import ChecklistTestsuite

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger

NUM_PROCESSES_IN_USE = 3 # os.cpu_count()


class SelfBleu:
    def __init__(self, text_file=None, texts=None, gram=3):
        # the json file used for retraining sa models
        self.texts = None
        if texts is not None and text_file is None:
            self.texts = texts
        elif texts is None and text_file is not None:
            real_data = Utils.read_json(text_file)
            self.texts = real_data['train']['text']
            if 'test' in real_data.keys():
                self.texts.extend(real_data['test']['text'])
            # end if
        # end if
        self.gram = gram
        self.sample_size = 500
        self.reference = None
        self.is_first = True
        self.num_data = len(self.texts)

    def get_reference(self):
        reference = list()
        if self.reference is None:
            self.reference = self.texts
            for t in self.texts:
                reference.append(Utils.tokenize(t))
            # end for
            self.reference = reference
            return reference
        # end if
        return self.reference

    def calc_bleu(self, reference, hypothesis, weight):
        # print(multiprocessing.current_process())
        return nltk.translate.bleu_score.sentence_bleu(
            reference, hypothesis, weight,
            smoothing_function=SmoothingFunction().method1
        )

    def get_score(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        # end if
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(processes=NUM_PROCESSES_IN_USE)
        result = list()
        for d_i in tqdm(range(self.num_data)):
            hypothesis = reference[d_i]
            other = reference[:d_i] + reference[d_i+1:]
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))
        # end for
        score = 0.
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        # end for
        pool.close()
        pool.join()
        return float("{:.3f}".format(score / cnt))
    

def read_our_seeds(task, search_dataset_name, selection_method):
    seed_file = Macros.result_dir / f"seed_inputs_{task}_{search_dataset_name}.json"
    seed_dict = Utils.read_json(seed_file)
    texts_lcs = dict()
    texts_all = list()
    for seeds in seed_dict:
        lc = seeds['requirement']['description']
        seed_sents = [s[1] for s in seeds['seeds']]
        texts_lcs[lc] = seed_sents
        texts_all.extend(seed_sents)
    # end for
    
    # for l in Utils.read_txt(seed_dir / 'cksum_map.txt'):
    #     l_split = l.strip().split('\t')
    #     lc, cksum_val = l_split[0], l_split[1]
    #     seeds = Utils.read_json(template_dir / f"seeds_{cksum_val}.json")
    #     texts = list()
    #     for s in seeds:
    #         texts_all.append(s['input'])
    #         texts.append(s['input'])
    #     # end for
    #     exps_file = template_dir / f"exps_{cksum_val}.json"
    #     if os.path.exists(str(exps_file)):
    #         exps = Utils.read_json(exps_file)
    #         for e in exps:
    #             texts_all.append(e['input'])
    #             texts.append(e['input'])
    #         # end for
    #     # end if
    #     if lc==Macros.OUR_LC_LIST[9]:
    #         texts_lcs['Parsing sentiment in (question, no) form'] = texts
    #     elif lc==Macros.OUR_LC_LIST[10]:
    #         texts_lcs['Parsing sentiment in (question, no) form'].extend(texts)
    #     else:
    #         texts_lcs[lc] = texts
    #     # end if
    # # end for
    return texts_all, texts_lcs

def read_checklist_testcases(task, search_dataset_name, selection_method):
    seed_file = Macros.result_dir / f"seed_input_{task}_{search_dataset_name}.json"
    seed_dict = Utils.read_json(seed_file)
    texts_lcs = dict()
    texts_all = list()
    for seeds in seed_dict:
        lc = seeds['requirement']['description']
        sents = ChecklistTestsuite.get_sents(
            Macros.checklist_sa_dataset_file,
            lc
        )
        _sents = [s[1] for s in sents]
        texts_lcs[lc] = _sents
        texts_all.extend(_sents)
    # end for
    return texts_all, texts_lcs
# def read_checklist_testcases():
#     LC_LIST = Macros.CHECKLIST_LC_LIST
#     tsuite, tsuite_dict = Utils.read_testsuite(Macros.checklist_sa_dataset_file)
#     test_names = list(set(tsuite_dict['test_name']))
#     texts_all = list()
#     texts_lcs = dict()
#     num_data = 0
#     for test_name in test_names:
#         if test_name in LC_LIST:
#             if test_name==Macros.CHECKLIST_LC_LIST[8]:
#                 texts_lcs['Q & A: yes'] = tsuite.tests[test_name].data
#             elif test_name==Macros.CHECKLIST_LC_LIST[9]:
#                 texts_lcs['Q & A: yes'].extend(tsuite.tests[test_name].data)
#             else:
#                 texts_lcs[test_name] = tsuite.tests[test_name].data
#             # end if
#             texts_all.extend(tsuite.tests[test_name].data)
#         # end if
#     # end for
#     return texts_all, texts_lcs


def main_seed(task,
              search_dataset_name,
              selection_method):
    logger_file = Macros.log_dir / f"seeds_{task}_{search_dataset_name}_selfbleu.log"
    logger = Logger(logger_file=logger_file,
                    logger_name='seed_selfbleu_log')
    Macros.selfbleu_result_dir.mkdir(parents=True, exist_ok=True)
    result_file = Macros.selfbleu_result_dir / f"seeds_{task}_{search_dataset_name}_selfbleu.json"
    _, texts_ours = read_our_seeds(task,
                                   search_dataset_name,
                                   selection_method)
    result = dict()
    scores = dict()
    scores_baseline = dict()
    for lc in texts_ours.keys():
        st = time.time()
        logger.print(f"OURS::{lc}", end='::')
        sbleu = SelfBleu(texts=texts_ours[lc])
        scores[lc] = {
            'num_data': sbleu.num_data,
            'score': sbleu.get_score()
        }
        result = {
            'ours': scores,
            'checklist': scores_baseline
        }
        Utils.write_json(result, result_file, pretty_format=True)
        ft = time.time()
        logger.print(f"{round(ft-st,2)}secs", end='::')
        logger.print(f"num_data:{scores[lc]['num_data']}::score:{scores[lc]['score']}")
    # end for

    _, texts_checklist = read_checklist_testcases(task,
                                                  search_dataset_name,
                                                  selection_method)
    for lc in texts_checklist.keys():
        st = time.time()
        logger.print(f"BL::{lc}", end='::')
        sbleu = SelfBleu(texts=texts_checklist[lc])
        scores_baseline[lc] = {
            'num_data': sbleu.num_data,
            'score': sbleu.get_score()
        }
        result = {
            'ours': scores,
            'checklist': scores_baseline
        }
        Utils.write_json(result, result_file, pretty_format=True)
        ft = time.time()
        logger.print(f"{round(ft-st,2)}secs", end='::')
        logger.print(f"num_data:{scores_baseline[lc]['num_data']}::score:{scores_baseline[lc]['score']}")
    # end for

    result = {
        'ours': scores,
        'checklist': scores_baseline
    }
    Utils.write_json(result, result_file, pretty_format=True)
    return

