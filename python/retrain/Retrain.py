# This script is to re-test models using failed cases
# found from testsuite results

from typing import *
from pathlib import Path

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from checklist.test_suite import TestSuite as suite
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score

from ..utils.Macros import Macros
from ..utils.Utils import Utils
# from Testsuite import Testsuite
from ..model.Model import Model
from .Dataset import Dataset

import os
import torch
import random
import numpy as np


class ChecklistTestcases:
    
    @classmethod
    def write_checklist_testcase(cls, save_file):
        tsuite, tsuite_dict = Utils.read_testsuite(Macros.checklist_sa_dataset_file)
        test_names = list(set(tsuite_dict['test_name']))
        test_data = dict()
        for test_name in test_names:
            test_data[test_name] = {
                'sents': tsuite.tests[test_name].data,
                'labels': tsuite.tests[test_name].labels
            }
            num_data = len(test_data[test_name]['sents'])
            if type(test_data[test_name]['labels'])!=list:
                test_data[test_name]['labels'] = [test_data[test_name]['labels']]*num_data
            # end if
            type_list = ['test']*num_data
            num_train_data = int(num_data*Macros.TRAIN_RATIO)
            num_train_data += ((num_data*Macros.TRAIN_RATIO)%num_train_data)>0
            type_list[:num_train_data] = ['train']*num_train_data
            random.shuffle(type_list)
            test_data[test_name]['types'] = type_list

            # set data labels in a range between 0. and 1. from 0,1,2
            test_data[test_name]['labels'] = [0.5*float(l) for l in test_data[test_name]['labels']]
        # end for

        dataset = dict()
        for idx in range(len(tsuite_dict['text'])):
            test_sent = tsuite_dict['text'][idx]
            test_name = tsuite_dict['test_name'][idx]
            test_case = tsuite_dict['test_case'][idx]
            sent_idx = test_data[test_name]['sents'].index(test_sent)
            label = test_data[test_name]['labels'][sent_idx]
            _type = test_data[test_name]['types'][sent_idx]
            if _type=='train':
                if 'train' not in dataset.keys():
                    dataset['train'] = dict()
                    dataset['train']['text'] = [test_sent]
                    dataset['train']['label'] = [label]
                    dataset['train']['test_name'] = [test_name]
                else:
                    dataset['train']['text'].append(test_sent)
                    dataset['train']['label'].append(label)
                    dataset['train']['test_name'].append(test_name)
                # end if
            else:
                if 'test' not in dataset.keys():
                    dataset['test'] = dict()
                    dataset['test']['text'] = [test_sent]
                    dataset['test']['label'] = [label]
                    dataset['test']['test_name'] = [test_name]
                else:
                    dataset['test']['text'].append(test_sent)
                    dataset['test']['label'].append(label)
                    dataset['test']['test_name'].append(test_name)
                # end if
            # end if
        # end for
        Utils.write_json(dataset, save_file, pretty_format=True)
        return dataset
    
    @classmethod
    def get_checklist_testcase(cls):
        if not os.path.exists(Macros.checklist_sa_testcase_file):
            Macros.checklist_result_dir.mkdir(parents=True, exist_ok=True)
            return cls.write_checklist_testcase(
                Macros.checklist_sa_testcase_file
            )
        # end if
        return Utils.read_json(
            Macros.checklist_sa_testcase_file
        )


class Retrain:

    def __init__(self, model_name, num_labels, dataset_file):
        self.model = self.load_model(model_name)
        self.num_labels = num_labels
        self.tokenizer = self.load_tokenizer(model_name)
        self.train_dataset, self.eval_dataset = self.get_tokenized_dataset(dataset_file)
        self.output_dir = Macros.retrain_output_dir / model_name.replace("/", "-")
        
    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def load_model(self, model_name):
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def get_tokenized_dataset(self, dataset_file):
        raw_dataset = Utils.read_json(dataset_file)
        train_texts = self.tokenizer(raw_dataset['train']['text'], truncation=True, padding=True)
        train_labels = raw_dataset['train']['label']
        eval_texts = self.tokenizer(raw_dataset['test']['text'], truncation=True, padding=True)
        eval_labels = raw_dataset['test']['label']
        train_dataset = Dataset(train_texts, labels=train_labels, num_labels=self.num_labels)
        eval_dataset = Dataset(eval_texts, labels=eval_labels, num_labels=self.num_labels)
        return train_dataset, eval_dataset

    def compute_metrics(self, p):
        scores, labels = p
        pr, preds_all, pp_all = list(), list(), list()
        labels_all = list()
        preds = torch.nn.functional.softmax(torch.tensor(scores), dim=-1)
        for pred, label in zip(preds, labels):
            if np.array_equal(label,[0.,1.]): # positive
                pr.append(pred[-1])
                labels_all.append(2.)
            elif np.array_equal(label,[0.,0.,1.]): # positive
                pr.append(pred[-1])
                labels_all.append(2.)
            elif np.array_equal(label,[1.,0.]): # negative
                pr.append(1.-pred[-1])
                labels_all.append(0.)
            elif np.array_equal(label,[1.,0.,0.]): # negative
                pr.append(pred[0])
                labels_all.append(0.)
            elif np.array_equal(label,[0.5,0.5]): # neutral
                pr.append(pred[-1])
                labels_all.append(1.)
            elif np.array_equal(label,[0.,1.,0.]): # neutral
                pr.append(pred[1])
                labels_all.append(1.)
            # end if
        # end for
        pr = np.array(pr)
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
        accuracy = accuracy_score(y_true=labels_all, y_pred=preds_all)
        recall = recall_score(y_true=labels_all, y_pred=preds_all, average='weighted')
        precision = precision_score(y_true=labels_all, y_pred=preds_all, average='weighted')
        f1 = f1_score(y_true=labels_all, y_pred=preds_all, average='weighted')
        
        # accuracy = accuracy_score(y_true=labels, y_pred=pred)
        # recall = recall_score(y_true=labels, y_pred=pred)
        # precision = precision_score(y_true=labels, y_pred=pred)
        # f1 = f1_score(y_true=labels, y_pred=pred)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=1.0,
            per_device_train_batch_size=4,
            do_train=True
        )       
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        trainer.train()
        return

    def evaluate(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_eval_batch_size=4,
            do_eval=True
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        results = trainer.evaluate()
        return results

    def test(cls):
        pass
    
    @classmethod
    def get_checklist_testcase(cls):
        return ChecklistTestcases.get_checklist_testcase()

    
def retrain(model_name, num_labels, dataset_file):
    retrainer = Retrain(model_name, num_labels, dataset_file)
    eval_result_before = retrainer.evaluate()
    print(f"BEFORE_TRAIN:\n{eval_result_before}")
    retrainer.train()
    eval_result_after = retrainer.evaluate()
    print(f"AFTER_TRAIN:\n{eval_result_after}")
    return eval_result_before, eval_result_after
