import checklist
import spacy
import itertools

import checklist.editor
import checklist.text_generation
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
import numpy as np
import spacy
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb

editor = checklist.editor.Editor()
editor.tg

nlp = spacy.load('en_core_web_sm')
suite = TestSuite()

editor.lexicons.keys()
editor.template('{a:religion_adj}').data

protected = {
    'race': ['a black','a hispanic', 'a white', 'an asian'],
    'sexual': editor.template('{a:sexual_adj}').data,
    'religion': editor.template('{a:religion_adj}').data,
    'nationality': editor.template('{a:nationality}').data[:20],
}

for p, vals in protected.items():
    print(p)
    t = editor.template(['{male} is %s {mask}.' % r for r in vals], return_maps=False, nsamples=300, save=True)
    t += editor.template(['{female} is %s {mask}.' % r for r in vals], return_maps=False, nsamples=300, save=True)
    test = INV(t.data, threshold=0.1, templates=t.templates)
    suite.add(test, 'protected: %s' % p, 'Fairness', 'Prediction should be the same for various adjectives within a protected class')
#     test.run(new_pp)
#     test.summary(n=3)
#     print()
#     preds = np.array(test.results.preds)
#     for i, x in enumerate(vals):
#         print('%.2f %s' % (preds[:, i].mean(), vals[i]))
#     print()
#     print()
#     print('-------------------------')

path = './sentiment_suite_for_fairness.pkl'
suite.save(path)