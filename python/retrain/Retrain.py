# This script is to re-test models using failed cases
# found from testsuite results

from typing import *
from pathlib import Path

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from checklist.test_suite import TestSuite as suite

from ..utils.Macros import Macros
from ..utils.Utils import Utils
# from Testsuite import Testsuite
from ..model.Model import Model
from .Dataset import Dataset

import os
import random
import numpy as np


class ChecklistTestcases:
    
    @classmethod
    def write_checklist_testcase(cls, save_file):
        def example_to_dict_fn(data):
            return { 'test_sent': data }
        tsuite = suite().from_file(Macros.checklist_sst_dataset_file)
        tsuite_dict = tsuite.to_dict(example_to_dict_fn=example_to_dict_fn)
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
        # end for

        dataset = dict()
        for idx in range(len(tsuite_dict['test_sent'])):
            test_sent = tsuite_dict['test_sent'][idx]
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
        if os.path.exists(Macros.checklist_sst_testcase_file):
            Macros.checklist_result_dir.mkdir(parents=True, exist_ok=True)
            return cls.write_checklist_testcase(
                Macros.checklist_sst_testcase_file
            )
        # end if
        return Utils.read_json(
            Macros.checklist_sst_testcase_file
        )


class Retrain:

    def __init__(self, model_name, dataset_file):
        self.model = self.load_model(model_name)
        self.tokenizer = self.load_tokenizer(model_name)
        self.train_dataset, self.eval_dataset = self.get_tokenized_dataset(dataset_file)
        self.output_dir = Macros.retrain_output_dir / model_name.replace("/", "-")
        
    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def load_model(self, model_name):
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def get_tokenized_dataset(self, dataset_file):
        raw_dataset = Utils.read_json(dataset_file)
        train_texts = self.tokenizer(raw_dataset['train']['text'], truncatiion=True, padding=True)
        train_labels = raw_dataset['train']['label']
        eval_texts = self.tokenizer(raw_dataset['test']['text'], truncatiion=True, padding=True)
        eval_labels = raw_dataset['test']['label']
        train_dataset = Dataset(train_texts, train_labels)
        eval_dataset = Dataset(eval_texts, eval_labels)
        return train_dataset, eval_dataset

    def compute_metrics(self, p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)
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

    
def retrain(model_name, dataset_file):
    retrainer = Retrain(model_name, dataset_file)
    retrainer.train()
    eval_result = retrainer.evaluate()
    return eval_result
