# experiment for selfbleu score
# A higher Self-BLEU score implies less diversity of the document

# OURS::Sentiment change over time, present should prevail::75294sents::around6secperonesent::125.49hourstocomplete(sequentially)

import os
import nltk
import time
import random
import multiprocessing

from tqdm import tqdm
from multiprocessing import Pool
# from functools import partial
from nltk.translate.bleu_score import SmoothingFunction

from ..testsuite.Search import ChecklistTestsuite
from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger

NUM_PROCESSES_IN_USE = 40 # os.cpu_count()
NUM_TRIALS = 3

class SelfBleu:
    def __init__(self,
                 text_file=None,
                 texts=None,
                 num_data=None,
                 logger=None,
                 gram=3):
        # the json file used for retraining sa models
        self.texts = None
        self.logger = logger
        self.score = 0.
        self.cnt = 0
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
        self.num_data = num_data if len(texts)>num_data else len(texts)

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
        st = time.time()
        # self.logger.print(hypothesis, end='::')
        # print(multiprocessing.current_process())
        score = nltk.translate.bleu_score.sentence_bleu(
            reference, hypothesis, weight,
            smoothing_function=SmoothingFunction().method1
        )
        ft = time.time()
        self.logger.print(f"score{score}::{round(ft-st,3)}sec::pcs{os.getpid()}")
        return score

    def get_score_wo_sample(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        # end if
        weight = tuple((1. / ngram for _ in range(ngram)))
        self.score = 0.
        self.cnt = 0
        result = list()
        pool = Pool(processes=NUM_PROCESSES_IN_USE)
        for d_i in tqdm(range(len(reference))):
            hypothesis = reference[d_i]
            other = reference[:d_i] + reference[d_i+1:]
            result.append(pool.apply_async(self.calc_bleu,
                                           args=(other, hypothesis, weight)))
        # end for
        for i in tqdm(result):
            self.score += i.get()
            self.cnt += 1
        # end for
        pool.close()
        pool.join()
        return round(self.score / self.cnt, 3)
    
    def get_score(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        # end if
        weight = tuple((1. / ngram for _ in range(ngram)))
        raw_scores = list()
        for tr in range(NUM_TRIALS):
            random.seed(tr)
            result = list()
            self.score = 0.
            self.cnt = 0
            # def callback(score):
            #     self.score += score
            #     self.cnt += 1
            #     return
            ref_idxs = list(range(len(reference)))
            random.shuffle(ref_idxs)
            reference_sample = [reference[r_i] for r_i in ref_idxs[:self.num_data]]
            
            pool = Pool(processes=NUM_PROCESSES_IN_USE)
            for d_i in tqdm(range(len(reference_sample))):
                hypothesis = reference_sample[d_i]
                other = reference_sample[:d_i] + reference_sample[d_i+1:]
                # score = self.calc_bleu(other, hypothesis, weight)
                # self.score += score
                # self.cnt += 1
                result.append(pool.apply_async(self.calc_bleu,
                                               args=(other, hypothesis, weight)))
            # end for
            # score = 0.
            # cnt = 0
            for i in tqdm(result):
                self.score += i.get()
                self.cnt += 1
            # end for
            raw_scores.append(self.score / self.cnt)
            pool.close()
            pool.join()
        # end for
        avg_score = Utils.avg(raw_scores)
        med_score = Utils.median(raw_scores)
        std_score = Utils.stdev(raw_scores)
        print(f'get_score done avg:{avg_score}, med:{med_score}, std:{std_score}, cnt:{self.cnt}')
        return {
            'avg': avg_score,
            'median': med_score,
            'stdev': std_score
        }
    

def read_our_seeds(task,
                   search_dataset_name,
                   selection_method='',
                   num_seeds=-1,
                   num_trials=-1):
    if num_trials>0:
        texts_lcs_over_trials = list()
        texts_all_over_trials = list()
        for num_trial in range(num_trials):
            _num_trial = '' if num_trial==0 else str(num_trial+1)
            seed_file = Macros.result_dir / f"cfg_expanded_inputs{_num_trial}_{task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds.json"
            seed_dict = Utils.read_json(seed_file)
            texts_lcs = dict()
            texts_all = list()
            for seeds in seed_dict:
                lc = seeds['requirement']['description']
                seed_sents = [s for s in seeds['inputs'].keys()]
                texts_lcs[lc] = seed_sents
                texts_all.extend(seed_sents)
            # end for
            texts_lcs_over_trials.append(texts_lcs)
            texts_all_over_trials.append(texts_all)
        # end for
        return texts_all_over_trials, texts_lcs_over_trials
    else:
        seed_res_dir_name = f"seeds_{task}_{search_dataset_name}"
        seed_dir = Macros.result_dir / seed_res_dir_name
        cksums = Utils.read_txt(seed_dir / 'cksum_map.txt')
        texts_lcs = dict()
        texts_all = list()
        for l in cksums:
            lc, cksum_val = l.split('\t')[0].strip(), l.split('\t')[1].strip()
            seed_file = seed_dir / f"seed_{cksum_val}.json"
            seed_dict = Utils.read_json(seed_file)
            seed_sents = [s[1] for s in seed_dict['seeds']]
            texts_lcs[lc] = seed_sents
            texts_all.extend(seed_sents)
        # end for
        return texts_all, texts_lcs
    # end if
        

def read_our_exps(task,
                  search_dataset_name,
                  selection_method,
                  num_seeds,
                  num_trials):
    if num_seeds<0:
        seed_file = Macros.result_dir / f"cfg_expanded_inputs{num_trials}_{task}_{dataset_name}_{selection_method}.json"
    else:
        seed_file = Macros.result_dir / f"cfg_expanded_inputs{num_trials}_{task}_{dataset_name}_{selection_method}_{num_seeds}seeds.json"
    # end if
    seed_dict = Utils.read_json(seed_file)
    texts_lcs = dict()
    texts_all = list()
    for seeds in seed_dict:
        lc = seeds['requirement']['description']
        exp_sents = list()
        for s in seeds['inputs'].keys():
            exp_sents.extend([e[5] for e in seeds['inputs'][s]['exp_inputs']])
        # end for
        texts_lcs[lc] = exp_sents
        texts_all.extend(exp_sents)
    # end for
    return texts_all, texts_lcs

def read_checklist_testcases(task, search_dataset_name):
    seed_res_dir_name = f"seeds_{task}_{search_dataset_name}"
    seed_dir = Macros.result_dir / seed_res_dir_name
    cksums = Utils.read_txt(seed_dir / 'cksum_map.txt')
    texts_lcs = dict()
    texts_all = list()
    for l in cksums:
        lc, cksum_val = l.split('\t')[0].strip(), l.split('\t')[1].strip()
        sents = ChecklistTestsuite.get_sents(
            Macros.checklist_sa_dataset_file,
            lc
        )
        _sents = [s[1] for s in sents]
        texts_lcs[lc] = _sents
        texts_all.extend(_sents)
    # end for
    return texts_all, texts_lcs


def main_seed(task,
              search_dataset_name,
              selection_method,
              num_seeds,
              num_trials):
    if num_seeds>0:
        logger_file = Macros.log_dir / f"seeds_over{num_trials}_{task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds_selfbleu.log"
        result_file = Macros.selfbleu_result_dir / f"seeds_over{num_trials}_{task}_{search_dataset_name}_{selection_method}_{num_seeds}seeds_selfbleu.json"
        logger = Logger(logger_file=logger_file,
                        logger_name='seed_selfbleu_log')
        Macros.selfbleu_result_dir.mkdir(parents=True, exist_ok=True)
        _, texts_checklist = read_checklist_testcases(task,
                                                      search_dataset_name)
        _, texts_ours = read_our_seeds(task,
                                       search_dataset_name,
                                       selection_method=selection_method,
                                       num_seeds=num_seeds,
                                       num_trials=num_trials)
        if os.path.exists(str(result_file)):
            scores = Utils.read_json(result_file)
        else:
            scores = dict()
        # end if
        for lc in texts_checklist.keys():
            if lc not in scores.keys():
                logger.print(f"OURS::{lc}")
                our_sents, bl_sents = list(), list()
                scores[lc] = {
                    'ours': {
                        'selfbleu_scores': list()
                    },
                    'bl': {
                        'selfbleu_scores': list()
                    }
                }
                for num_trial in range(num_trials):
                    random.seed(num_trial)
                    _texts_ours = texts_ours[num_trial]
                    _texts_ours_lc = _texts_ours[lc]
                    texts_checklist_lc = texts_checklist[lc]
                    if len(_texts_ours_lc)<len(texts_checklist_lc):
                        num_samples = len(_texts_ours_lc)
                        r_idxs = list(range(len(texts_checklist_lc)))
                        random.shuffle(r_idxs)
                        our_sents = _texts_ours_lc
                        bl_sents = [texts_checklist_lc[r_i] for r_i in r_idxs[:num_samples]]
                    else:
                        num_samples = len(texts_checklist_lc)
                        r_idxs = list(range(len(_texts_ours_lc)))
                        random.shuffle(r_idxs)
                        our_sents = [_texts_ours_lc[r_i] for r_i in r_idxs[:num_samples]]
                        bl_sents = texts_checklist_lc
                    # end if
                    scores[lc]['num_data'] = len(our_sents)
                    sbleu = SelfBleu(texts=our_sents,
                                     num_data=len(our_sents),
                                     logger=logger)
                    score = sbleu.get_score_wo_sample()
                    scores[lc]['ours']['selfbleu_scores'].append(score)
                    sbleu_bl = SelfBleu(texts=bl_sents,
                                        num_data=len(bl_sents),
                                        logger=logger)
                    score_bl = sbleu_bl.get_score_wo_sample()
                    scores[lc]['bl']['selfbleu_scores'].append(score_bl)
                # end for
                logger.print(f"{scores[lc]}")
                scores[lc]['ours']['avg_score'] = Utils.avg(scores[lc]['ours']['selfbleu_scores'])
                scores[lc]['ours']['med_score'] = Utils.median(scores[lc]['ours']['selfbleu_scores'])
                scores[lc]['ours']['std_score'] = Utils.stdev(scores[lc]['ours']['selfbleu_scores'])
                scores[lc]['bl']['avg_score'] = Utils.avg(scores[lc]['bl']['selfbleu_scores'])
                scores[lc]['bl']['med_score'] = Utils.median(scores[lc]['bl']['selfbleu_scores'])
                scores[lc]['bl']['std_score'] = Utils.stdev(scores[lc]['bl']['selfbleu_scores'])
                Utils.write_json(scores, result_file, pretty_format=True)
            # end if
        # end for
    else:
        logger_file = Macros.log_dir / f"seeds_{task}_{search_dataset_name}_selfbleu.log"
        result_file = Macros.selfbleu_result_dir / f"seeds_{task}_{search_dataset_name}_selfbleu.json"
        logger = Logger(logger_file=logger_file,
                        logger_name='seed_selfbleu_log')
        Macros.selfbleu_result_dir.mkdir(parents=True, exist_ok=True)
        
        _, texts_checklist = read_checklist_testcases(task,
                                                      search_dataset_name)
        _, texts_ours = read_our_seeds(task,
                                       search_dataset_name)
        # _, texts_ours = read_our_exps(task,
        #                               search_dataset_name,
        #                               selection_method,
        #                               num_seeds,
        #                               num_trials)

        if os.path.exists(str(result_file)):
            result = Utils.read_json(result_file)
        else:
            result = {
                'ours': dict(),
                'checklist': dict()
            }
            # end if
            scores = dict()
            scores_baseline = dict()
            for lc in texts_ours.keys():
                if lc not in result['ours'].keys():
                    st = time.time()
                    logger.print(f"OURS::{lc}")
                    sbleu = SelfBleu(texts=texts_ours[lc],
                                     num_data=len(texts_checklist[lc]),
                                     logger=logger)
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
                # end if
            # end for
    
        for lc in texts_checklist.keys():
            if lc not in result['checklist'].keys():
                st = time.time()
                logger.print(f"BL::{lc}", end='::')
                sbleu = SelfBleu(texts=texts_checklist[lc],
                                 num_data=len(texts_checklist[lc]),
                                 logger=logger)
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
            # end if
        # end for

        result = {
            'ours': scores,
            'checklist': scores_baseline
        }
        Utils.write_json(result, result_file, pretty_format=True)
    # end if
    return

