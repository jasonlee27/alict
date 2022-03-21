import os
from multiprocessing import Pool

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from .Macros import Macros
from .Utils import Utils

class SelfBleu:
    def __init__(self, test_file, gram=3):
        # the json file used for retraining sa models
        self.test_file = test_file
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
            real_data = Utils.read_json(self.test_file)
            for text in real_data['train']['text']:
                text = Utils.tokenize(text)
                reference.append(text)
            # end for
            if 'test' in real_data.keys():
                for text in real_data['test']['text']:
                    text = Utils.tokenize(text)
                    reference.append(text)
                # end for
            # end if
            self.reference = reference
            self.num_data = len(reference)
            return reference
        else:
            return self.reference
        # end if

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(
            reference, hypothesis, weight,
            smoothing_function=SmoothingFunction().method1
        )

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        test_data = Utils.read_json(self.test_file)
        for hypothesis in test_data['train']['text']:
            hypothesis = Utils.tokenize(hypothesis)
            bleu.append(self.calc_bleu(reference, hypothesis, weight))
        # end for
        if 'test' in test_data.keys():
            for hypothesis in test_data['test']['text']:
                hypothesis = Utils.tokenize(hypothesis)
                bleu.append(self.calc_bleu(reference, hypothesis, weight))
            # end for
        # end if
        return sum(bleu) / len(bleu)


def main(task, search_dataset_name, selection_method):
    testcase_file = Macros.retrain_dataset_dir / f"{task}_{search_dataset_name}_{selection_method}_testcase.json"
    sbleu = SelfBleu(test_file=testcase_file)
    sbleu_score = sbleu.get_score()

    checklist_testcase_file = Macros.checklist_sa_testcase_file
    sbleu_baseline = SelfBleu(test_file=checklist_testcase_file)
    sbleu_baseline_score = sbleu.get_score()

    Macros.selfbleu_result_dir.mkdir(parents=True, exist_ok=True)
    result_file = Macros.selfbleu_result_dir / "{task}_{search_dataset_name}_{selection_method}_testcase_selfbleu.json"
    result = {
        'num_data': sbleu.num_data,
        'score': sbleu_score,
        'baseline_name': 'checklist',
        'baseline_num_data': sbleu_baseline.num_data,
        'baseline_score': sbleu_baseline_score
    }
    Utils.write_json(result, result_file, pretty_format=True)
    return

