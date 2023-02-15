# experiment for selfbleu score
# A higher Self-BLEU score implies less diversity of the document

# OURS::Sentiment change over time, present should prevail::75294sents::around6secperonesent::125.49hourstocomplete(sequentially)

import re
import os
import math
import nltk
import time
import random
import multiprocessing

from tqdm import tqdm
from multiprocessing import Pool
# from functools import partial
from nltk.translate.bleu_score import SmoothingFunction

from ..seed.Search import Hatecheck
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

def read_hatecheck_testcases(task, search_dataset_name, selection_method):
    seed_res_dir_name = f"templates_{task}_{search_dataset_name}_{selection_method}"
    seed_dir = Macros.result_dir / seed_res_dir_name
    cksums = Utils.read_txt(seed_dir / 'cksum_map.txt')
    texts_lcs = dict()
    texts_all = list()
    for l in cksums:
        lc, cksum_val = l.split('\t')[0].strip(), l.split('\t')[1].strip()
        sents = Hatecheck.get_sents(
            Macros.hatecheck_data_file,
        )
        func_name = [
            Hatecheck.FUNCTIONALITY_MAP[key]
            for key in Hatecheck.FUNCTIONALITY_MAP.keys()
            if key.split('::')[-1]==lc
        ][0]
        _sents = [s['sent'] for s in sents if s['func']==func_name or s['func'] in func_name]
        texts_lcs[lc] = _sents
        texts_all.extend(_sents)
    # end for
    return texts_all, texts_lcs


def main_sample(task,
                search_dataset_name,
                selection_method):
    num_trials = 5
    num_samples = [200, 400, 600, 800, 1000]
    logger_file = Macros.log_dir / f"seed_exp_bl_sample_{task}_{search_dataset_name}_{selection_method}_selfbleu.log"
    result_file = Macros.selfbleu_result_dir / f"seed_exp_bl_sample_{task}_{search_dataset_name}_{selection_method}_selfbleu.json"
    logger = Logger(logger_file=logger_file,
                    logger_name='seed_selfbleu_log')
    Macros.selfbleu_result_dir.mkdir(parents=True, exist_ok=True)
    _, texts_hatecheck = read_hatecheck_testcases(task,
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
    for lc in texts_hatecheck.keys():
        if lc not in scores.keys():
            logger.print(lc)
            our_sents, bl_sents = list(), list()
            scores[lc] = {
                'ours_seed': {
                    f"{num_sample}sample": {
                        'num_data': list(),
                        'selfbleu_scores': list()
                    }
                    for num_sample in num_samples
                },
                'ours_exp': {
                    f"{num_sample}sample": {
                        'num_data': list(),
                        'selfbleu_scores': list()
                    }
                    for num_sample in num_samples
                },
                'bl': {
                    f"{num_sample}sample": {
                        'num_data': list(),
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
                    #     len(texts_hatecheck[lc])
                    # ])
                    seed_sents = random.sample(texts_seed[lc], min(len(texts_seed[lc]), num_sample))
                    exp_sents = list()
                    for s in seed_sents:
                        if any(seed_exp_map[lc].get(s, list())):
                            exp_sent = random.sample(seed_exp_map[lc][s], 1)
                            exp_sents.extend(exp_sent)
                        else:
                            exp_sents.append(s)
                        # end if
                    # end for
                    # exp_sents = random.sample(exp_sents,
                    #                           min(len(exp_sents), num_sample))
                    bl_sents = random.sample(texts_hatecheck[lc],
                                             min(len(texts_hatecheck[lc]), num_sample))
                    sbleu_seed = SelfBleu(texts=seed_sents,
                                          num_data=len(seed_sents),
                                          logger=logger)
                    score_seed = sbleu_seed.get_score_wo_sample()
                    sbleu_exp = SelfBleu(texts=exp_sents,
                                         num_data=len(exp_sents),
                                         logger=logger)
                    score_exp = sbleu_exp.get_score_wo_sample()
                    sbleu_bl = SelfBleu(texts=bl_sents,
                                        num_data=len(bl_sents),
                                        logger=logger)
                    score_bl = sbleu_bl.get_score_wo_sample()
                    scores[lc]['ours_seed'][f"{num_sample}sample"]['num_data'].append(len(seed_sents))
                    scores[lc]['ours_exp'][f"{num_sample}sample"]['num_data'].append(len(exp_sents))
                    scores[lc]['bl'][f"{num_sample}sample"]['num_data'].append(len(bl_sents))
                    scores[lc]['ours_seed'][f"{num_sample}sample"]['selfbleu_scores'].append(score_seed)
                    scores[lc]['ours_exp'][f"{num_sample}sample"]['selfbleu_scores'].append(score_exp)
                    scores[lc]['bl'][f"{num_sample}sample"]['selfbleu_scores'].append(score_bl)
                # end for
                logger.print(f"{scores[lc]}")
                scores[lc]['ours_seed'][f"{num_sample}sample"]['avg_score'] = Utils.avg(scores[lc]['ours_seed'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['ours_seed'][f"{num_sample}sample"]['med_score'] = Utils.median(scores[lc]['ours_seed'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['ours_seed'][f"{num_sample}sample"]['std_score'] = Utils.stdev(scores[lc]['ours_seed'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['ours_exp'][f"{num_sample}sample"]['avg_score'] = Utils.avg(scores[lc]['ours_exp'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['ours_exp'][f"{num_sample}sample"]['med_score'] = Utils.median(scores[lc]['ours_exp'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['ours_exp'][f"{num_sample}sample"]['std_score'] = Utils.stdev(scores[lc]['ours_exp'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['bl'][f"{num_sample}sample"]['avg_score'] = Utils.avg(scores[lc]['bl'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['bl'][f"{num_sample}sample"]['med_score'] = Utils.median(scores[lc]['bl'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['bl'][f"{num_sample}sample"]['std_score'] = Utils.stdev(scores[lc]['bl'][f"{num_sample}sample"]['selfbleu_scores'])
                Utils.write_json(scores, result_file, pretty_format=True)
            # end for
        # end if
    # end for
    return


def main_mtnlp(task,
               search_dataset_name,
               selection_method):
    st = time.time()
    num_trials = 5
    logger_file = Macros.log_dir / f"mtnlp_{task}_{search_dataset_name}_{selection_method}_selfbleu.log"
    seed_dir = Macros.result_dir / f"templates_{task}_{search_dataset_name}_{selection_method}"
    mtnlp_dir = Macros.download_dir / 'MT-NLP'
    mtnlp_res_dir =  Macros.result_dir / 'mtnlp' / f"{task}_{search_dataset_name}_{selection_method}_sample"
    result_file = mtnlp_res_dir / f"mtnlp_sample_{task}_{search_dataset_name}_{selection_method}_selfbleu.json"
    logger = Logger(logger_file=logger_file,
                    logger_name='mtnlp_mutation_log')
    mtnlp_files = sorted([
        f for f in os.listdir(mtnlp_res_dir)
        if f.startswith('mutations_s2lct_seed_samples') and f.endswith('.json')
    ])
    seed_lcs = dict()
    seed_sents = list()
    mt_sents = list()
    sample_files = list()
    for mtnlp_file in mtnlp_files:
        mt_res = Utils.read_json(mtnlp_res_dir / mtnlp_file)
        sample_file = mtnlp_dir / mt_res['sample_file']
        sample_files.append(mt_res['sample_file'])
        file_ind = re.search('raw_file(\d)\.txt', mt_res['sample_file']).group(1)
        _seed_sents = list(mt_res['mutations'].keys())
        seed_sents.extend(_seed_sents)
        search_lns = [
            l.strip()
            for l in Utils.read_txt(mtnlp_dir / f"{task}_seed_samples_raw_file{file_ind}.txt")
            if l.strip()!=''
        ]
        for s in _seed_sents:
            for l in search_lns:
                if l.split('::')[0].strip()==s:
                    seed_lcs[s] = l.split('::')[-1].strip()
                # end if
            # end for
            ana_mt_sents = mt_res['mutations'][s]['ana']
            act_mt_sents = mt_res['mutations'][s]['act']
            if any(ana_mt_sents+act_mt_sents):
                mt_sents.extend(ana_mt_sents+act_mt_sents)
            # end if
        # end for        
    # end for
    
    exp_sents = list()
    req_dir = Macros.result_dir / 'reqs'
    req_file = req_dir / 'requirements_desc_hs.txt'
    for s in seed_lcs.keys():
        lc_desc = seed_lcs[s].strip()
        lc_cksum = Utils.get_cksum(lc_desc)
        _lc_cksum = Utils.get_cksum(lc_desc.lower())
        seed_file = seed_dir / f"cfg_expanded_inputs_{lc_cksum}.json"
        if os.path.exists(seed_dir / f"cfg_expanded_inputs_{_lc_cksum}.json") and \
           not os.path.exists(seed_dir / f"cfg_expanded_inputs_{lc_cksum}.json"):
            seed_file = seed_dir / f"cfg_expanded_inputs_{_lc_cksum}.json"    
        # end if
        cfg_res = Utils.read_json(seed_file)
        if cfg_res is not None:
            # tokens = Utils.tokenize(s)
            # s = Utils.detokenize(tokens)
            if s in cfg_res['inputs'].keys():
                for exp in cfg_res['inputs'][s]['exp_inputs']:
                    exp_sent = exp[5]
                    exp_sents.append(exp_sent)
                # end for
            # end if
        # end if
    # end for

    logger.print(f"OURS_SELFBLEU_SAMPLE::mtnlp::")
    scores = {
        'sample_file': sample_files,
        'ours_exp': {
            'num_data': len(exp_sents),
            'sample_size': list(),
            'scores': list()
        },
        'mtnlp': {
            'num_data': len(mt_sents),
            'sample_size': list(),
            'scores': list()
        }
    }
    for t in tqdm(range(num_trials)):
        random.seed(t)
        sample_exp_sents = random.sample(exp_sents,
                                         min(len(seed_sents), len(exp_sents)))
        sample_mt_sents = random.sample(mt_sents,
                                        min(len(seed_sents), len(mt_sents)))
        sbleu_exp = SelfBleu(texts=sample_exp_sents,
                             num_data=len(sample_exp_sents),
                             logger=logger)
        score_exp = sbleu_exp.get_score_wo_sample()
        sbleu_mt = SelfBleu(texts=sample_mt_sents,
                            num_data=len(sample_mt_sents),
                            logger=logger)
        score_mt = sbleu_mt.get_score_wo_sample()
        scores['ours_exp']['sample_size'].append(len(sample_exp_sents))
        scores['ours_exp']['scores'].append(score_exp)
        scores['mtnlp']['sample_size'].append(len(sample_mt_sents))
        scores['mtnlp']['scores'].append(score_mt)
    # end for
    Utils.write_json(scores, result_file, pretty_format=True)
    return


def main_hatecheck(task,
                   selection_method):
    num_trials = 1
    num_samples = [200] # [10, 50, 100, 150, 200]
    logger_file = Macros.log_dir / f"seed_exp_bl_all_{task}_hatecheck_{selection_method}_selfbleu.log"
    result_file = Macros.selfbleu_result_dir / f"seed_exp_bl_all_{task}_hatecheck_{selection_method}_selfbleu.json"
    logger = Logger(logger_file=logger_file,
                    logger_name='seed_selfbleu_log')
    Macros.pdr_cov_result_dir.mkdir(parents=True, exist_ok=True)

    seed_file = f"cfg_expanded_inputs_{task}_hatecheck_{selection_method}.json"
    _, texts_seed = read_our_seeds(task,
                                   'hatecheck',
                                   selection_method)
    seed_exp_map, _ = read_our_exps(task,
                                    'hatecheck',
                                    selection_method)
    scores = dict()
    for lc in texts_seed.keys():
        if lc not in scores.keys():
            logger.print(lc)
            our_sents, bl_sents = list(), list()
            # max_num_samples = 1000
            # num_samples = list(range(100, max_num_samples+100, 100))
            scores[lc] = {
                'hatecheck': {
                    f"{num_sample}sample": {
                        'selfbleu_scores': list()
                    }
                    for num_sample in num_samples
                },
                'hatecheck_exp': {
                    f"{num_sample}sample": {
                        'selfbleu_scores': list()
                    }
                    for num_sample in num_samples
                }
            }
            for num_sample in num_samples:
                for num_trial in range(num_trials):
                    random.seed(num_trial)
                    seed_sents = random.sample(texts_seed[lc], min(len(texts_seed[lc]), num_sample))
                    exp_sents = list()
                    for s in texts_seed[lc]:
                        if any(seed_exp_map[lc].get(s, list())):
                            exp_sent = random.sample(texts_seed[lc]+[s], 1)
                            exp_sents.extend(exp_sent)
                        else:
                            exp_sents.append(s)
                        # end if
                    # end for
                    # seed_exp_sents = random.sample(texts_seed[lc]+texts_exp,
                    #                                min(len(texts_seed[lc]+texts_exp), num_sample))
                    # seed_exp_sents = seed_sents+exp_sents
                    sbleu_seed = SelfBleu(texts=seed_sents,
                                          num_data=len(seed_sents),
                                          logger=logger)
                    score_seed = sbleu_seed.get_score_wo_sample()
                    scores[lc]['hatecheck'][f"{num_sample}sample"]['selfbleu_scores'].append(score_seed)
                    if any(exp_sents):
                        sbleu_seed_exp = SelfBleu(texts=exp_sents,
                                                  num_data=len(exp_sents),
                                                  logger=logger)
                        score_seed_exp = sbleu_seed_exp.get_score_wo_sample()
                    else:
                        score_seed_exp = 0.
                    # end if
                    scores[lc]['hatecheck_exp'][f"{num_sample}sample"]['selfbleu_scores'].append(score_seed_exp)
                # end for
                logger.print(f"{scores[lc]}")
                scores[lc]['hatecheck'][f"{num_sample}sample"]['avg_score'] = Utils.avg(scores[lc]['hatecheck'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['hatecheck'][f"{num_sample}sample"]['med_score'] = Utils.median(scores[lc]['hatecheck'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['hatecheck'][f"{num_sample}sample"]['std_score'] = Utils.stdev(scores[lc]['hatecheck'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['hatecheck_exp'][f"{num_sample}sample"]['avg_score'] = Utils.avg(scores[lc]['hatecheck_exp'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['hatecheck_exp'][f"{num_sample}sample"]['med_score'] = Utils.median(scores[lc]['hatecheck_exp'][f"{num_sample}sample"]['selfbleu_scores'])
                scores[lc]['hatecheck_exp'][f"{num_sample}sample"]['std_score'] = Utils.stdev(scores[lc]['hatecheck_exp'][f"{num_sample}sample"]['selfbleu_scores'])
                Utils.write_json(scores, result_file, pretty_format=True)
            # end for
        # end if
    # end for
    return
