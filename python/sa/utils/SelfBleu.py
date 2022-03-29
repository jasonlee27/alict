import os
import nltk
import multiprocessing

from multiprocessing import Pool
from functools import partial
from nltk.translate.bleu_score import SmoothingFunction

from .Macros import Macros
from .Utils import Utils

NUM_PROCESSES_IN_USE=8



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
        self.num_data = -1

    def get_score(self):
        self.get_reference()
        return self.get_bleu()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            for t in self.texts:
                text = Utils.tokenize(t)
                reference.append(t)
            # end for
            self.reference = reference
            self.num_data = len(reference)
            return reference
        else:
            return self.reference
        # end if

    def calc_bleu(self, hyp_i, reference, weight):
        # print(multiprocessing.current_process())
        hypothesis = Utils.tokenize(self.texts[hyp_i])
        return nltk.translate.bleu_score.sentence_bleu(
            reference, hypothesis, weight,
            smoothing_function=SmoothingFunction().method1
        )

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        hypothesis = self.texts
        pool_obj = Pool(processes=NUM_PROCESSES_IN_USE)
        mp_calc_bleu = partial(self.calc_bleu,
                               reference=reference,
                               weight=weight)
        scores = pool_obj.map(mp_calc_bleu, range(self.num_data))
        bleu.extend(scores)
        return float("{:.3f}".format(sum(bleu) / len(bleu)))


def read_our_testcases(task, search_dataset_name, selection_method):
    template_dir = Macros.result_dir / f"templates_{task}_{search_dataset_name}_{selection_method}"
    texts_all = list()
    texts_lcs = dict()
    for l in Utils.read_txt(template_dir / 'cksum_map.txt'):
        l_split = l.strip().split('\t')
        lc, cksum_val = l_split[0], l_split[1]
        seeds = Utils.read_json(template_dir / f"seeds_{cksum_val}.json")
        texts = list()
        for s in seeds:
            texts_all.append(s['input'])
            texts.append(s['input'])
        # end for
        exps_file = template_dir / f"exps_{cksum_val}.json"
        if os.path.exists(str(exps_file)):
            exps = Utils.read_json(exps_file)
            texts_all.append(s['input'])
            texts.append(s['input'])
        # end if
        texts_lcs[lc] = texts
    # end for
    return texts_all, texts_lcs

def read_checklist_testcases():
    LC_LIST = [
        'Sentiment-laden words in context',
        'neutral words in context'
        'used to, but now'
        'simple negations: not negative',
        'simple negations: not neutral is still neutral',
        'Hard: Negation of positive with neutral stuff in the middle (should be negative)'
        'negation of neutral with neutral in the middle, should still neutral'
        'simple negations: I thought x was negative, but it was not (should be neutral or positive)',
        'my opinion is what matters',
        'Q & A: yes',
        'Q & A: yes (neutral)',
        'Q & A: no',
    ]
    tsuite, tsuite_dict = Utils.read_testsuite(Macros.checklist_sa_dataset_file)
    test_names = list(set(tsuite_dict['test_name']))
    texts_all = list()
    texts_lcs = dict()
    num_data = 0
    for test_name in test_names:
        if test_name in LC_LIST:
            texts_lcs[test_name] = tsuite.tests[test_name].data
            texts_all.extend(texts_lcs[test_name])
        # end if
    # end for
    return texts_all, texts_lcs


def main(task, search_dataset_name, selection_method):
    _, texts_ours = read_our_testcases(task, search_dataset_name, selection_method)
    scores = list()
    for lc in texts_ours.keys():
        sbleu = SelfBleu(texts=texts_ours[lc])
        scores.append({
            'lc': lc,
            'num_data': sbleu.num_data,
            'scores': sbleu.get_score()
        })
    # end for

    _, texts_checklist = read_checklist_testcases()
    scores_baseline = list()
    for lc in texts_checklist.keys():
        sbleu = SelfBleu(texts=texts_checklist[lc])
        scores_baseline.append({
            'lc': lc,
            'num_data': sbleu.num_data,
            'scores': sbleu.get_score()
        })
    # end for

    Macros.selfbleu_result_dir.mkdir(parents=True, exist_ok=True)
    result_file = Macros.selfbleu_result_dir / f"{task}_{search_dataset_name}_{selection_method}_testcase_selfbleu.json"
    result = {
        'scores': scores,
        'baseline_name': 'checklist',
        'baseline_scores': scores_baseline
    }
    Utils.write_json(result, result_file, pretty_format=True)
    return

