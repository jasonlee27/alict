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

from ..seed.Search import ChecklistTestsuite
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
                   selection_method):
    texts_lcs = dict()
    texts_all = list()
    seed_dir = Macros.result_dir / f"templates_{task}_{search_dataset_name}_{selection_method}"
    seed_files = [
        f for f in os.listdir(str(seed_dir))
        if f.startswith('cfg_expanded_inputs_') and f.endswith('.json')
    ]
    for seed_file in seed_files:
        seed_dict = Utils.read_json(seed_dir / seed_file)
        lc = seed_dict['requirement']['description']
        seed_sents = [s for s in seed_dict['inputs'].keys()]
        texts_lcs[lc] = seed_sents
        texts_all.extend(seed_sents)
    # end for
    return texts_all, texts_lcs
        
def read_our_exps(task,
                  search_dataset_name,
                  selection_method):
    texts_lcs = dict()
    seed_exp_map = dict()
    seed_dir = Macros.result_dir / f"templates_{task}_{search_dataset_name}_{selection_method}"
    seed_files = [
        f for f in os.listdir(str(seed_dir))
        if f.startswith('cfg_expanded_inputs_') and f.endswith('.json')
    ]
    for seed_file in seed_files:
        seed_dict = Utils.read_json(seed_dir / seed_file)
        lc = seed_dict['requirement']['description']
        if lc not in seed_exp_map.keys():
            seed_exp_map[lc] = dict()
        # end if
        exp_sents = list()
        for s in seed_dict['inputs'].keys():
            exp_sents_per_seed = [e[5] for e in seed_dict['inputs'][s]['exp_inputs']]
            if s not in seed_exp_map[lc].keys():
                seed_exp_map[lc][s] = exp_sents_per_seed
            # end if
            exp_sents.extend(exp_sents_per_seed)
        # end for
        texts_lcs[lc] = exp_sents
    # end for
    return seed_exp_map, texts_lcs

def read_checklist_testcases(task, search_dataset_name, selection_method):
    seed_res_dir_name = f"templates_{task}_{search_dataset_name}_{selection_method}"
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
              selection_method):
    num_trials = 10
    num_samples = [50, 100, 150, 200]
    logger_file = Macros.log_dir / f"seeds_{task}_{search_dataset_name}_{selection_method}_selfbleu.log"
    result_file = Macros.selfbleu_result_dir / f"seeds_{task}_{search_dataset_name}_{selection_method}_selfbleu.json"
    logger = Logger(logger_file=logger_file,
                    logger_name='seed_selfbleu_log')
    Macros.selfbleu_result_dir.mkdir(parents=True, exist_ok=True)
    _, texts_checklist = read_checklist_testcases(task,
                                                  search_dataset_name,
                                                  selection_method)
    _, texts_seed = read_our_seeds(task,
                                   search_dataset_name,
                                   selection_method)
    seed_exp_map, _ = read_our_exps(task,
                                    search_dataset_name,
                                    selection_method)
    # if os.path.exists(str(result_file)):
    #     scores = Utils.read_json(result_file)
    # else:
    #     scores = dict()
    # # end if
    scores = dict()
    for lc in texts_checklist.keys():
        if lc not in scores.keys():
            logger.print(f"OURS::{lc}")
            our_sents, bl_sents = list(), list()
            scores[lc] = {
                'ours_seed': {
                    f"{num_sample}sample": {
                        'selfbleu_scores': list()
                    }
                    for num_sample in num_samples
                },
                'ours_seed_exp': {
                    f"{num_sample}sample": {
                        'selfbleu_scores': list()
                    }
                    for num_sample in num_samples
                },
                'bl': {
                    f"{num_sample}sample": {
                        'selfbleu_scores': list()
                    }
                    for num_sample in num_samples
                }
            }
            for num_sample in num_samples:
                # scores[lc]['ours'][f"{num_sample}sample"]['selfbleu_scores'] = list()
                for num_trial in range(num_trials):
                    random.seed(num_trial)
                    # _num_sample = min([
                    #     num_sample,
                    #     len(texts_seed_ours[lc]),
                    #     len(texts_checklist[lc])
                    # ])
                    seed_sents = random.sample(texts_seed[lc], min(len(texts_seed[lc]), num_sample))
                    texts_exp = list()
                    for s in seed_sents:
                        texts_exp.extend(seed_exp_map[lc][s])
                    # end for
                    bl_sents = random.sample(texts_checklist[lc], min(len(texts_checklist[lc]), num_sample))
                    exp_sents = random.sample(texts_exp, min(len(texts_exp), num_sample))
                    sbleu_seed = SelfBleu(texts=seed_sents,
                                          num_data=len(seed_sents),
                                          logger=logger)
                    score_seed = sbleu.get_score_wo_sample()
                    scores[lc]['ours_seed'][f"{num_sample}sample"]['selfbleu_scores'].append(score_seed)
                    sbleu_seed_exp = SelfBleu(texts=seed_sents+exp_sents,
                                         num_data=len(seed_sents+exp_sents),
                                         logger=logger)
                    score_seed_exp = sbleu_exp.get_score_wo_sample()
                    scores[lc]['ours_seed_exp'][f"{num_sample}sample"]['selfbleu_scores'].append(score_seed_exp)
                    sbleu_bl = SelfBleu(texts=bl_sents,
                                        num_data=len(bl_sents),
                                        logger=logger)
                    score_bl = sbleu_bl.get_score_wo_sample()
                    scores[lc]['bl'][f"{num_sample}sample"]['selfbleu_scores'].append(score_bl)
                # end for
                logger.print(f"{scores[lc]}")
                scores[lc]['ours_seed'][f"{num_sample}sample"]['avg_score'] = Utils.avg(scores[lc]['ours_seed'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['ours_seed'][f"{num_sample}sample"]['med_score'] = Utils.median(scores[lc]['ours_seed'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['ours_seed'][f"{num_sample}sample"]['std_score'] = Utils.stdev(scores[lc]['ours_seed'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['ours_seed_exp'][f"{num_sample}sample"]['avg_score'] = Utils.avg(scores[lc]['ours_seed_exp'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['ours_seed_exp'][f"{num_sample}sample"]['med_score'] = Utils.median(scores[lc]['ours_seed_exp'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['ours_seed_exp'][f"{num_sample}sample"]['std_score'] = Utils.stdev(scores[lc]['ours_seed_exp'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['bl'][f"{num_sample}sample"]['avg_score'] = Utils.avg(scores[lc]['bl'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['bl'][f"{num_sample}sample"]['med_score'] = Utils.median(scores[lc]['bl'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['bl'][f"{num_sample}sample"]['std_score'] = Utils.stdev(scores[lc]['bl'][f"{num_sample}sample"]['selfbleu_scores'])
                Utils.write_json(scores, result_file, pretty_format=True)
            # end for
        # end if
    # end for
    return

