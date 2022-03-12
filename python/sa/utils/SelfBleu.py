import os
from multiprocessing import Pool

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from .Macros import Macros
from .Utils import Utils

class SelfBleu:
    def __init__(self, test_file, gram=3):
        super().__init__()
        # the json file used for retraining sa models
        self.test_file = test_file
        self.gram = gram
        self.sample_size = 500
        self.reference = None
        self.is_first = True

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
            for text in real_data['test']['text']:
                text = Utils.tokenize(text)
                reference.append(text)
            # end for
            self.reference = reference
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
        for hypothesis in real_data['train']['text']:
            hypothesis = Utils.tokenize(hypothesis)
            bleu.append(self.calc_bleu(reference, hypothesis, weight))
        # end for
        return sum(bleu) / len(bleu)