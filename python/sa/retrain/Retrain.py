# This script is to re-test models using failed cases
# found from testsuite results


from typing import *
from pathlib import Path

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from checklist.test_suite import TestSuite as suite
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..utils.Logger import Logger
# from Testsuite import Testsuite
from ..model.Testmodel import Testmodel
from ..model.Model import Model
from .Dataset import Dataset

import os
import torch
import shutil
import random
import numpy as np


class Sst2:
    
    # SST2 dataset is used for retraining with our generated testcases.
    # For retraining, we only use train.tsv file.

    @classmethod
    def get_sents(cls, sent_file: Path):
        # sent_file: train.tsv.
        # Each line in the file consists of (sentence, label)
        sents = Utils.read_sv(sent_file, delimeter='\t', is_first_attributes=True)
        sents = sents['lines']
        result = list()
        for s_i, s in enumerate(sents):
            result.append((s_i,s[0],s[1]))
        # end for
        return result
    
    @classmethod
    def write_sst2_train_sents(cls, save_file):
        sst2_train_file = Macros.sst2_dir / 'train.tsv'
        sents = cls.get_sents(sst2_train_file)
        dataset = {
            'train': {
                'text': list(),
                'label': list()
            }
        }
        for s in sents:
            dataset['train']['text'].append(s[1])
            dataset['train']['label'].append(s[2])
        # end for
        Utils.write_json(dataset, save_file, pretty_format=True)
        return dataset
    
    @classmethod
    def get_trainset_for_retrain(cls):
        if not os.path.exists(Macros.sst2_sa_trainset_file):
            Macros.retrain_dataset_dir.mkdir(parents=True, exist_ok=True)
            return cls.write_sst2_train_sents(
                Macros.sst2_sa_trainset_file,
            )
        # end if
        return Utils.read_json(
            Macros.sst2_sa_trainset_file
        )


class SstTestcases:

    LC_NOT_INCLUDED_LIST = [
        'parsing positive sentiment in (question, no) form',
    ]

    @classmethod
    def write_sst_testcase(cls, task, selection_method, save_file, upscale_by_baseline=True, baseline='checklist'):
        test_results_dir = Macros.result_dir / f"test_results_{task}_sst_{selection_method}"
        cksum_vals = [
            os.path.basename(testsuite_file).split("_")[-1].split(".")[0]
            for testsuite_file in os.listdir(test_results_dir)
            if testsuite_file.startswith(f"{task}_testsuite_seeds_") and testsuite_file.endswith(".pkl")
        ]
        
        test_data = dict()
        for cksum_val in cksum_vals:
            testsuite_files = [
                f"{task}_testsuite_seeds_{cksum_val}.pkl",
                f"{task}_testsuite_exps_{cksum_val}.pkl"
            ]
            test_name = ''
            testsuite_files = [tf for tf in testsuite_files if os.path.exists(test_results_dir / tf)]
            for f_i, testsuite_file in enumerate(testsuite_files):
                tsuite, tsuite_dict = Utils.read_testsuite(test_results_dir / testsuite_file)
                for tn in list(set(tsuite_dict['test_name'])):
                    test_name = tn.split('::')[-1]
                    if tsuite.tests[tn].labels is not None and test_name.lower() not in cls.LC_NOT_INCLUDED_LIST:
                        if test_name not in test_data.keys():
                            test_data[test_name] = {
                                'sents': tsuite.tests[tn].data,
                                'labels': tsuite.tests[tn].labels
                            }
                        else:
                            test_data[test_name]['sents'].extend(tsuite.tests[tn].data)
                            test_data[test_name]['labels'].extend(tsuite.tests[tn].labels)
                        # end if
                    # end if
                # end for
            # end for
            if test_name in test_data.keys():
                num_data = len(test_data[test_name]['sents'])
                type_list = ['test']*num_data
                num_train_data = int(num_data*Macros.TRAIN_RATIO)
                num_train_data += ((num_data*Macros.TRAIN_RATIO)%num_train_data)>0
                type_list[:num_train_data] = ['train']*num_train_data
                random.shuffle(type_list)
                test_data[test_name]['types'] = type_list
                
                # set data labels in a range between 0. and 1. from 0,1,2
                test_data[test_name]['labels'] = [0.5*float(l) for l in test_data[test_name]['labels']]
            # end if
        # end for
        
        dataset = dict()
        for test_name in test_data.keys():
            for idx in range(len(test_data[test_name]['sents'])):
                test_sent = test_data[test_name]['sents'][idx]
                label = test_data[test_name]['labels'][idx]
                _type = test_data[test_name]['types'][idx]
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
        # end for
        num_train_data = len(dataset['train']['text'])

        # for balancing number of data for retraining with baseline,
        # we compute number of baseline dataset (checklist) and upscale
        # the number of our dataset with the ratio of # dataset between baseline and ours
        upscale_rate = 1
        if upscale_by_baseline:
            if baseline=='checklist':
                baseline_dataset = ChecklistTestcases.get_testcase_for_retrain(
                    Macros.checklist_sa_testcase_file
                )
                num_baseline_train_data = len(baseline_dataset['train']['text'])
                upscale_rate = num_baseline_train_data // num_train_data
                if upscale_rate > 1:
                    texts, labels, test_names = list(), list(), list()
                    for d_i in range(num_train_data):
                        t = dataset['train']['text'][d_i]
                        l = dataset['train']['label'][d_i]
                        tn = dataset['train']['test_name'][d_i]
                        texts.extend([t]*upscale_rate)
                        labels.extend([l]*upscale_rate)
                        test_names.extend([tn]*upscale_rate)
                    # end for
                    num_balanced_train_data = list(range(len(texts)))
                    random.shuffle(num_balanced_train_data)
                    dataset['train'] = {
                        'text': [texts[d_i] for d_i in num_balanced_train_data],
                        'label': [labels[d_i] for d_i in num_balanced_train_data],
                        'test_name': [test_names[d_i] for d_i in num_balanced_train_data]
                    }
                # end if
            # end if
        # end if
        Utils.write_json(dataset, save_file, pretty_format=True)
        return dataset

    @classmethod
    def get_testcase_for_retrain(cls,
                                 task,
                                 selection_method,
                                 upscale_by_baseline=True,
                                 baseline='checklist'):
        testcase_file = Macros.retrain_dataset_dir / f"{task}_sst_{selection_method}_testcase.json"
        if not os.path.exists(testcase_file):
            Macros.retrain_dataset_dir.mkdir(parents=True, exist_ok=True)
            return cls.write_sst_testcase(
                task,
                selection_method,
                testcase_file,
                upscale_by_baseline=upscale_by_baseline,
                baseline=baseline
            )
        # end if
        return Utils.read_json(testcase_file)

    
class ChecklistTestcases:

    LC_LIST = [
        'sentiment-laden words in context',
        'neutral words in context',
        'used to, but now',
        'simple negations: not neutral is still neutral',
        'hard: negation of positive with neutral stuff in the middle (should be negative)',
        'my opinion is what matters',
        'q & a: yes',
        'q & a: yes (neutral)',
    ]
    
    @classmethod
    def write_checklist_testcase(cls, save_file):
        tsuite, tsuite_dict = Utils.read_testsuite(Macros.checklist_sa_dataset_file)
        test_names = list(set(tsuite_dict['test_name']))
        test_data = dict()
        num_data = 0
        for test_name in test_names:
            if test_name.lower() in cls.LC_LIST and tsuite.tests[test_name].labels is not None:
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
            # end if
        # end for
        
        dataset = dict()
        for test_name in test_data.keys():
            test_sents = test_data[test_name]['sents']
            labels = test_data[test_name]['labels']
            _types = test_data[test_name]['types']
            for d_i in range(len(test_sents)):
                if _types[d_i]=='train':
                    if 'train' not in dataset.keys():
                        dataset['train'] = dict()
                        dataset['train']['text'] = [test_sents[d_i]]
                        dataset['train']['label'] = [labels[d_i]]
                        dataset['train']['test_name'] = [test_name]
                    else:
                        dataset['train']['text'].append(test_sents[d_i])
                        dataset['train']['label'].append(labels[d_i])
                        dataset['train']['test_name'].append(test_name)
                    # end if
                else:
                    if 'test' not in dataset.keys():
                        dataset['test'] = dict()
                        dataset['test']['text'] = [test_sents[d_i]]
                        dataset['test']['label'] = [labels[d_i]]
                        dataset['test']['test_name'] = [test_name]
                    else:
                        dataset['test']['text'].append(test_sents[d_i])
                        dataset['test']['label'].append(labels[d_i])
                        dataset['test']['test_name'].append(test_name)
                    # end if
                # end if
            # end for
        # end for
        Utils.write_json(dataset, save_file, pretty_format=True)
        return dataset
    
    @classmethod
    def get_testcase_for_retrain(cls, task):
        if not os.path.exists(Macros.checklist_sa_testcase_file):
            Macros.retrain_dataset_dir.mkdir(parents=True, exist_ok=True)
            return cls.write_checklist_testcase(
                Macros.checklist_sa_testcase_file,
            )
        # end if
        return Utils.read_json(
            Macros.checklist_sa_testcase_file
        )


class Retrain:

    LC_MAP = {
        "short sentences with sentiment-laden adjectives": [ChecklistTestcases.LC_LIST[0]],
        "short sentences with neutral adjectives and nouns": [ChecklistTestcases.LC_LIST[1]],
        "sentiment change over time, present should prevail": [ChecklistTestcases.LC_LIST[2]],
        "negated neutral should still be neutral": [ChecklistTestcases.LC_LIST[3]],
        "negated positive with neutral content in the middle": [ChecklistTestcases.LC_LIST[4]],
        "author sentiment is more important than of others": [ChecklistTestcases.LC_LIST[5]],
        "parsing sentiment in (question, yes) form": [ChecklistTestcases.LC_LIST[6],
                                                      ChecklistTestcases.LC_LIST[7]],
    }
    
    def __init__(self,
                 task,
                 model_name,
                 selection_method,
                 label_vec_len,
                 dataset_file,
                 eval_dataset_file,
                 output_dir,
                 lc_desc=None,
                 logger=None):
        self.task = task
        self.model_name = model_name
        self.selection_method = selection_method
        self.model = self.load_model(model_name)
        self.dataset_file = dataset_file # dataset for train
        self.eval_dataset_file = eval_dataset_file # dataset_for val
        self.dataset_name = os.path.basename(str(self.dataset_file)).split('_testcase.json')[0]
        self.label_vec_len = label_vec_len
        self.tokenizer = self.load_tokenizer(model_name)
        self.logger = logger
        self.train_dataset, self.eval_dataset, self.eval_on_train_dataset = \
            self.get_tokenized_dataset(dataset_file, eval_dataset_file, lc_desc=lc_desc)
        model_dir_name = model_name.replace("/", "-")
        self.output_dir = output_dir
        self.batch_size = 16
        self.num_epochs = 2.
        self.lc_desc = lc_desc
        
    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def load_model(self, model_name):
        return AutoModelForSequenceClassification.from_pretrained(model_name)

    def get_train_data_by_lc_types(self, raw_dataset, lc_desc):
        if type(lc_desc)==str:
            return {
                'train': {
                    'text': [raw_dataset['train']['text'][t_i] for t_i, t in enumerate(raw_dataset['train']["test_name"]) if t==lc_desc],
                    'label': [raw_dataset['train']['label'][t_i] for t_i, t in enumerate(raw_dataset['train']["test_name"]) if t==lc_desc]
                }
            }
        elif type(lc_desc)==list:
            texts = list()
            labels = list()
            for lc in lc_desc:
                texts.extends([raw_dataset['train']['text'][t_i] for t_i, t in enumerate(raw_dataset['train']["test_name"]) if t==lc])
                labels.extends([raw_dataset['train']['label'][t_i] for t_i, t in enumerate(raw_dataset['train']["test_name"]) if t==lc])
            # end for
            return {
                'train': {
                    'text': texts, 'label': labels
                }
            }

    def get_lc_descs(self, lc_desc):
        if type(lc_desc)==str:
            if lc_desc.lower() in self.LC_MAP.keys():
                return self.LC_MAP[lc_desc]
            else:
                return [key for key, val in self.LC_MAP.items() if lc_desc.lower() in val]
            # end if
        elif type(lc_desc)==list:
            res = list()
            for lc in lc_desc:
                res.extend([key for key, val in self.LC_MAP.items() if lc.lower() in val])
            # end for
            return list(set(res))
        # end if

    def get_eval_data_by_lc_types(self, raw_dataset, lc_desc):
        eval_descs = self.get_lc_descs(lc_desc)
        texts, labels = list(), list()
        for t_i, t in enumerate(raw_dataset['train']["test_name"]):
            if t.lower() in eval_descs and raw_dataset['train']['text'][t_i] not in texts:
                texts.append(raw_dataset['train']['text'][t_i])
                labels.append(raw_dataset['train']['label'][t_i])
            # end if
        # end for
        return {
            'text': texts,
            'label': labels
        }
    
    def get_testcases(self, dataset_file, lc_desc=None):
        raw_testcases = Utils.read_json(dataset_file)
        if lc_desc is not None:
            return self.get_train_data_by_lc_types(raw_testcases, lc_desc)
        # end if
        return raw_testcases
    
    def get_eval_testcases(self, eval_dataset_file, lc_desc=None):
        raw_testcases = Utils.read_json(eval_dataset_file)
        if lc_desc is not None:
            return self.get_eval_data_by_lc_types(raw_testcases, lc_desc)
        # end if
        texts, labels = list(), list()
        for t_i, t in enumerate(raw_testcases['train']["test_name"]):
            if raw_testcases['train']['text'][t_i] not in texts:
                texts.append(raw_testcases['train']['text'][t_i])
                labels.append(raw_testcases['train']['label'][t_i])
            # end if
        # end for
        return {
            'text': texts,
            'label': labels
        }
    
    def merge_train_data(self, sst_train_data, testcases):
        for t_i, t in enumerate(testcases['train']['text']):
            l = testcases['train']['label'][t_i]
            sst_train_data['train']['text'].append(t)
            sst_train_data['train']['label'].append(l)
        # end for
        return sst_train_data
        
    def get_tokenized_dataset(self, dataset_file, eval_dataset_file, lc_desc=None):
        sst2_train_dataset = Sst2.get_trainset_for_retrain()
        raw_testcases = self.get_testcases(dataset_file, lc_desc=lc_desc)
        
        raw_dataset = self.merge_train_data(sst2_train_dataset, raw_testcases)
        train_texts = self.tokenizer(raw_dataset['train']['text'], truncation=True, padding=True)
        train_labels = raw_dataset['train']['label']
        train_dataset = Dataset(train_texts, labels=train_labels, label_vec_len=self.label_vec_len)
        
        eval_testcases = self.get_eval_testcases(eval_dataset_file, lc_desc=lc_desc)
        eval_texts = self.tokenizer(eval_testcases['text'], truncation=True, padding=True)
        eval_labels = eval_testcases['label']
        eval_dataset = Dataset(eval_texts, labels=eval_labels, label_vec_len=self.label_vec_len)
        
        eval_on_train_texts = self.tokenizer(raw_testcases['train']['text'], truncation=True, padding=True)
        eval_on_train_labels = raw_testcases['train']['label']
        eval_on_train_dataset = Dataset(eval_on_train_texts, labels=eval_on_train_labels, label_vec_len=self.label_vec_len)
        
        num_testcases = len(raw_testcases['train']['text'])
        num_sst2_train_dataset = len(sst2_train_dataset['train']['text'])
        num_eval_testcases = len(eval_testcases['text'])
        print(f"#TrainData: SST2:{num_sst2_train_dataset}, Testcases:{num_testcases}, #EvalData: {num_eval_testcases}")
        if self.logger is not None:
            self.logger.print(f"#TrainData: SST2:{num_sst2_train_dataset}, Testcases:{num_testcases}, #EvalData: {num_eval_testcases}")
        # end if
        return train_dataset, eval_dataset, eval_on_train_dataset
    
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
        return {
            "num_data": len(preds_all),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def train(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            do_train=True,
            save_total_limit=3,
            save_strategy='epoch',
            evaluation_strategy='epoch',
            load_best_model_at_end=True
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        trainer.train()
        self.tokenizer.save_pretrained(self.output_dir)
        return
    
    def evaluate(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_eval_batch_size=self.batch_size,
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
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_on_train_dataset,
            compute_metrics=self.compute_metrics,
        )
        results_on_train = trainer.evaluate()
        return results, results_on_train
    
    def load_retrained_model(self):
        checkpoints = sorted([d for d in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir,d)) and d.startswith('checkpoint-')], key=lambda x: int(x.split('checkpoint-')[-1]))
        checkpoint_dir = self.output_dir / checkpoints[-1]
        tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        _task, _ = Model.model_map[self.task]
        return pipeline(
            _task,
            model=str(checkpoint_dir),
            tokenizer=tokenizer,
            framework="pt",
            device=0
        )

    def run_model_on_testsuite(self,
                               testsuite,
                               model,
                               pred_and_conf_fn=None,
                               n=Macros.nsamples,
                               logger=None):
        Model.run(testsuite,
                  model,
                  pred_and_conf_fn,
                  print_fn=None,
                  format_example_fn=None,
                  n=n,
                  logger=logger)
        return

    def test_on_our_testsuites(self, logger=None):
        _print = print
        if logger is not None:
            _print = logger.print
        # end if
        model = self.load_retrained_model()
        eval_dataset_name = os.path.basename(str(self.eval_dataset_file)).split('_testcase.json')[0]
        cksum_vals = [
            os.path.basename(test_file).split('_')[-1].split('.')[0]
            for test_file in os.listdir(Macros.result_dir / f"test_results_{eval_dataset_name}")
            if test_file.startswith(f"{self.task}_testsuite_seeds_") and test_file.endswith('.pkl')
        ]
        _print(f"***** Eval on Train Testsuite: ours *****")
        for cksum_val in cksum_vals:
            testsuite_files = [
                Macros.result_dir / f"test_results_{eval_dataset_name}" / f for f in [
                    f"{self.task}_testsuite_seeds_{cksum_val}.pkl",
                    f"{self.task}_testsuite_exps_{cksum_val}.pkl",
                ] if os.path.exists(Macros.result_dir / f"test_results_{eval_dataset_name}" / f)
            ]
            for testsuite_file in testsuite_files:
                testsuite = Testmodel.load_testsuite(testsuite_file)
                _print(f">>>>> RETRAINED MODEL: {self.model_name}")
                self.run_model_on_testsuite(
                    testsuite,
                    model,
                    pred_and_conf_fn=Testmodel.model_func_map[self.task],
                    n=Macros.nsamples,
                    logger=logger
                )
                _print(f"<<<<< RETRAINED MODEL: {self.model_name}")
            # end for
        # end for
        return

    def test_on_checklist_testsuite(self, logger=None):
        _print = print
        if logger is not None:
            _print = logger.print
        # end if
        _print(f"***** Eval on Train Testsuite: checklist *****")
        _print(f">>>>> RETRAINED MODEL: {self.model_name}")
        model = self.load_retrained_model()
        testsuite = Testmodel.load_testsuite(Macros.BASELINES[Macros.datasets[Macros.sa_task][1]]['testsuite_file'])
        self.run_model_on_testsuite(testsuite, model, Testmodel.model_func_map[self.task], n=Macros.nsamples, logger=logger)
        _print(f"<<<<< RETRAINED MODEL: {self.model_name}")
        return

    @classmethod
    def get_sst2_testcase_for_retrain(cls, task):
        return Sst2.get_trainset_for_retrain()
    
    @classmethod
    def get_sst_testcase_for_retrain(cls, task, selection_method):
        return SstTestcases.get_testcase_for_retrain(task, selection_method)
    
    @classmethod
    def get_checklist_testcase_for_retrain(cls, task):
        return ChecklistTestcases.get_testcase_for_retrain(task)

def remove_checkpoints(output_dir):
    if os.path.exists(output_dir):
        for d in os.listdir(output_dir):
            if os.path.isdir(os.path.join(output_dir,d)) and d.startswith('checkpoint-'):
                cmd = f"rm -rf {os.path.join(output_dir,d)}"
                os.system(cmd)
            # end if
        # end for
    # end if
    return 

def _retrain_by_lc_types(task,
                         model_name,
                         dataset_name,
                         selection_method,
                         label_vec_len,
                         dataset_file,
                         output_dir,
                         eval_dataset_file,
                         testing_on_testsuite,
                         log_file=None):
    logger = None
    if log_file is not None:
        logger = Logger(logger_file=log_file,
                        logger_name='retrain_over_lcs')
    # end if
    raw_dataset = Utils.read_json(dataset_file)
    eval_result = dict()
    lcs = sorted(list(set(raw_dataset['train']["test_name"])))
    if datset_naem==Macros.datasets[Macros.sa_task][1]:
        _lcs = [lc for lc in lcs if not lc.startswith('Q & A: yes')]
        _lcs.append(['Q & A: yes', 'Q & A: yes (neutral)'])
        lcs = _lcs
        del _lcs
    # end if
    for lc_i, lc_desc in enumerate(lcs):
        print(f">>>>> Retrain: LC<{lc_desc}>+SST2")
        if logger is not None:
            logger.print(f">>>>> Retrain: LC<{lc_desc}>+SST2")
        # end if
        remove_checkpoints(output_dir)
        lc_cksum = Utils.get_cksum(str(lc_desc))
        retrainer = Retrain(task,
                            model_name,
                            selection_method,
                            label_vec_len,
                            dataset_file,
                            eval_dataset_file,
                            output_dir,
                            lc_desc=lc_desc,
                            logger=logger)
        eval_result_before, eval_result_on_train_before = retrainer.evaluate()
        retrainer.train()
        eval_result_after, eval_result_on_train_after = retrainer.evaluate()
        eval_result[str(lc_desc)] = {
            'eval': {
                'before': eval_result_before,
                'after': eval_result_after
            },
            'eval_on_train': {
                'before': eval_result_on_train_before,
                'after': eval_result_on_train_after
            }
        }
        Utils.write_json(eval_result, output_dir / f"eval_results_lcs.json", pretty_format=True)
        if testing_on_testsuite:
            if dataset_name.lower()==Macros.datasets[Macros.sa_task][0]:
                retrainer.test_on_checklist_testsuite(logger=logger)
            elif dataset_name==Macros.datasets[Macros.sa_task][1]:
                retrainer.test_on_our_testsuites(logger=logger)
            # end if
            shutil.copyfile(log_file, output_dir / "eval_on_testsuite_results_lcs.txt")
        # end if
        print(f"<<<<< Retrain: LC<{lc_desc}>+SST2")
        if logger is not None:
            logger.print(f"<<<<< Retrain: LC<{lc_desc}>+SST2")
        # end if
    # end for
    return eval_result

def _retrain_all(task,
                 model_name,
                 dataset_name,
                 selection_method,
                 label_vec_len,
                 dataset_file,
                 output_dir,
                 eval_dataset_file,
                 testing_on_testsuite=True,
                 log_file=None):
    print(f">>>>> Retrain: ALL+SST2")
    if log_file is None:
        logger = None
    else:
        logger = Logger(logger_file=log_file,
                        logger_name='retrain_all')
        logger.print(f">>>>> Retrain: ALL+SST2")
    # end if
    remove_checkpoints(output_dir)
    retrainer = Retrain(task,
                        model_name,
                        selection_method,
                        label_vec_len,
                        dataset_file,
                        eval_dataset_file,
                        output_dir,
                        lc_desc=None,
                        logger=logger)
    eval_result_before, eval_result_on_train_before = retrainer.evaluate()
    retrainer.train()
    eval_result_after, eval_result_on_train_after = retrainer.evaluate()
    eval_result = {
        'eval': {
            'before': eval_result_before,
            'after': eval_result_after
        },
        'eval_on_train': {
            'before': eval_result_on_train_before,
            'after': eval_result_on_train_after
        }
    }
    Utils.write_json(eval_result, output_dir / "eval_results_all.json", pretty_format=True)
    if testing_on_testsuite:
        if dataset_name.lower()==Macros.datasets[Macros.sa_task][0]:
            retrainer.test_on_checklist_testsuite(logger=logger)
        elif dataset_name==Macros.datasets[Macros.sa_task][1]:
            retrainer.test_on_our_testsuites(logger=logger)
        # end if
        shutil.copyfile(log_file, output_dir / "eval_on_testsuite_results_all.txt")
    # end if
    print(f"<<<<< Retrain: ALL+SST2")
    if logger is not None:
        logger.print(f"<<<<< Retrain: ALL+SST2")
    # end if
    return eval_result
    
def retrain(task,
            model_name,
            dataset_name,
            selection_method,
            label_vec_len,
            dataset_file,
            eval_dataset_file,
            train_by_lcs=True,
            testing_on_testsuite=True,
            log_file=None):
    tags = os.path.basename(str(dataset_file)).split('_testcase.json')[0]
    dataset_name = tags.split(f"{task}_")[-1].split(f"_{selection_method}")[0]
    model_dir_name = tags+"_"+model_name.replace("/", "-")
    output_dir = Macros.retrain_model_dir / task / model_dir_name
    if train_by_lcs:
        eval_result = _retrain_by_lc_types(task,
                                           model_name,
                                           dataset_name,
                                           selection_method,
                                           label_vec_len,
                                           dataset_file,
                                           output_dir,
                                           eval_dataset_file,
                                           testing_on_testsuite,
                                           log_file=log_file)
    else:
        eval_result = _retrain_all(task,
                                   model_name,
                                   dataset_name,
                                   selection_method,
                                   label_vec_len,
                                   dataset_file,
                                   output_dir,
                                   eval_dataset_file,
                                   testing_on_testsuite,
                                   log_file=log_file)
    # end if
    return eval_result

def main_retrain(nlp_task, 
                 search_dataset_name,
                 selection_method, 
                 model_name,
                 label_vec_len, 
                 retrain_by_lcs,
                 testing_on_testsuite,
                 log_file):
    testcase_file, eval_testcase_file = None, None
    if search_dataset_name==Macros.datasets[nlp_task][0]:
        testcase_file = Macros.retrain_dataset_dir / f"{nlp_task}_{search_dataset_name}_{selection_method}_testcase.json"
        eval_testcase_file = Macros.checklist_sa_testcase_file
        if not os.path.exists(str(testcase_file)):
            Retrain.get_sst_testcase_for_retrain(nlp_task, selection_method)
        # end if
        if not os.path.exists(str(eval_testcase_file)):
            Retrain.get_checklist_testcase_for_retrain(nlp_task)
        # end if
    elif search_dataset_name==Macros.datasets[nlp_task][1]:
        testcase_file = Macros.checklist_sa_testcase_file
        eval_testcase_file = Macros.retrain_dataset_dir / f"{nlp_task}_sst_{selection_method}_testcase.json"
        if not os.path.exists(str(testcase_file)):
            Retrain.get_checklist_testcase_for_retrain(nlp_task)
        # end if
        if not os.path.exists(str(eval_testcase_file)):
            Retrain.get_sst_testcase_for_retrain(nlp_task, selection_method)
        # end if
    # end if

    _ = retrain(
        task=nlp_task,
        dataset_name=search_dataset_name,
        model_name=model_name,
        selection_method=selection_method,
        label_vec_len=label_vec_len,
        dataset_file=testcase_file,
        eval_dataset_file=eval_testcase_file,
        train_by_lcs=retrain_by_lcs,
        testing_on_testsuite=testing_on_testsuite,
        log_file=log_file
    )
    return
