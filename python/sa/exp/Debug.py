import os
import re
import csv
import json
import tqdm
import torch
import random
import pickle
import numpy as np
import pandas as pd
import multiprocessing

from typing import *
from pathlib import Path
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


THIS_DIR: Path = Path(os.path.dirname(os.path.realpath(__file__)))
STORAGE_DIR: Path = Path('/nas1-nfs1/data/jxl115330/s2lct')
RESULT_DIR: Path = STORAGE_DIR / '_results'
DOWNLOAD_DIR: Path = STORAGE_DIR / '_downloads'
DATASET_DIR: Path = DOWNLOAD_DIR / 'datasets'
DEBUG_SAMPLES = [
    ("maybe it is good, but even if you insist, I will not say yes, it is good.", 0.),
    ("maybe you have good eyes on the movie, but even if you ask me if it is good, i won't say yes.", 0.),
    ("maybe, but it is even hard to say yes." , 0.),
    ("maybe you are right, but i do not even want to be the yes man for this.", 0.),
    ("maybe, but even with my all efforts, I couldn't say yes.", 0.),
    ("maybe I am wrong, but with my all knowledge, it is bad. yes it is.", 0.),
    ("maybe it is lightly bad. but I even think it should be worse than that. yes it is really bad.", 0.),
    ("maybe you should watch, but fyi when I ask sombody if i should watch the film, even nobody won't say yes.", 0.),
    ("maybe we should destroy it, but even with your disagreement, everyone will agree and yes with me.", 0.),
    ("maybe many people will see this film, but it is not even halfway to safisfy me. yes for sure.", 0.)
]
TARGET_SAMPLES = [
    ("Do I think that \"Maybe it is asking too much, but if a movie is truly going to inspire me, I want a little more than this.\"? yes", 0.),
    ("Do I think that \"Maybe it is asking too much, but if a movie is truly going to inspire me, I want even a little more than this.\"? yes", 0.)
]
NUM_SEEDS = 10

def compute_metrics(p):
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

# Create torch dataset
class Dataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class Debug:

    @classmethod
    def find_last_checkpoints(cls, 
                              model_dir, 
                              checkpoint_pat=r"checkpoint\-(\d+)"):
        checkpoints = list()
        for f in os.listdir(model_dir):
            pat_match = re.search(checkpoint_pat, f)
            if pat_match is not None:
                step = int(pat_match.group(1))
                checkpoints.append((model_dir / f, step))
            # end if
        # end for
        return sorted(
            checkpoints, 
            reverse=True,
            key=lambda x: x[1]
        )
       
    @classmethod
    def load_model(cls, model_name, local_path=None):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if local_path is None:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(local_path)
        # end if
        return model, tokenizer

    @classmethod
    def load_sst2_dataset(cls, 
                          split, 
                          tokenizer, 
                          debug_samples=None, 
                          num_ds_copy=1):
        split_fnames = {
            'train': 'train.tsv',
            'val':   'dev.tsv',
            'test':  'test.tsv',
        }
        if split is not None:
            assert split in split_fnames.keys()
            root_dir: Path = DATASET_DIR / 'sst2'
            df = pd.read_csv(str(root_dir / split_fnames[split]), header=None, sep='\t')
            labels = df[0].values[1:].astype(int)
            data = df[1].values[1:]
            num_orig_data = len(data)
            if debug_samples is not None:
                for s,l in debug_samples:
                    for _ in range(num_ds_copy):
                        data = np.append(data, s)
                        labels = np.append(labels, int(l))
                    # end for
                # end for
            # end if
            data_tokenized = tokenizer(data.tolist(), padding=True, truncation=True, max_length=512)
            return data_tokenized, labels.tolist()
        else:
            data = list()
            labels = list()
            if debug_samples is not None:
                for s,l in debug_samples:
                    for _ in range(num_ds_copy):
                        data.append(s)
                        labels.append(int(l))
                    # end for
                # end for
            # end if
            data_tokenized = tokenizer(data, padding=True, truncation=True, max_length=512)
            return data_tokenized, labels
        # end if

    @classmethod
    def get_training_args(cls, **kwargs):
        out_dir = kwargs.get('out_dir', None)
        out_dir.mkdir(parents=True, exist_ok=True)
        per_device_train_batch_size = kwargs.get('per_device_train_batch_size', 32)
        per_device_eval_batch_size = kwargs.get('per_device_eval_batch_size', 32)
        eval_steps = kwargs.get('eval_steps', 500)
        num_train_epochs = kwargs.get('num_train_epochs', 50)
        seed_num = 0
        training_args = TrainingArguments(
            output_dir=str(out_dir),
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            seed=seed_num,
            save_total_limit=2,
            load_best_model_at_end=True
        )
        return training_args

    @classmethod
    def get_trainer(cls, 
                    model, 
                    train_dataset=None,
                    eval_dataset=None,
                    training_args=None,
                    early_stopping_patience=7):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience
                )
            ]
        )
        return trainer

    @classmethod
    def main(cls,
             model_name="textattack/bert-base-uncased-SST-2",
             seed=0,
             num_ds_copy_train=1,
             num_ds_copy_val=1):
        # load pre-trained model
        model, tokenizer = cls.load_model(model_name)

        # Read data and preprocessing the data
        debug_samples = DEBUG_SAMPLES
        train_data_tokenized, train_labels = cls.load_sst2_dataset(
            'train',
            tokenizer,
            debug_samples=debug_samples, 
            num_ds_copy=num_ds_copy_train
        )
        val_data_tokenized, val_labels = cls.load_sst2_dataset(
            'val',
            tokenizer,
            debug_samples=debug_samples, 
            num_ds_copy=num_ds_copy_val
        )

        train_dataset = Dataset(train_data_tokenized, labels=train_labels)
        val_dataset = Dataset(val_data_tokenized, labels=val_labels)
        out_dir = RESULT_DIR / 'application_debug'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"SEED_{seed}...")
        # initiate seed for python and pytorch
        model_dir = out_dir / f"seed{seed}"
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        
        # ----- 2. Fine-tune pretrained model -----#
        # Define Trainer parameters
        per_device_train_batch_size = 64
        per_device_eval_batch_size = 64
        eval_steps = 500
        num_train_epochs = 100
        seed_num = 0
        training_args = cls.get_training_args(
            out_dir=model_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_steps=eval_steps,
            num_train_epochs=num_train_epochs
        )
        trainer = cls.get_trainer(
            model, 
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            training_args=training_args
        )
        
        trainer.train()
        
        # testing the trained model
        target_samples = TARGET_SAMPLES
        checkpoints = cls.find_last_checkpoints(
            model_dir,
            checkpoint_pat=r"checkpoint\-(\d+)"
        )
        local_model_path = checkpoints[0][0]
        model, tokenizer = cls.load_model(model_name, local_path=str(local_model_path))
        test_data_tokenized, test_labels = cls.load_sst2_dataset(
            None,
            tokenizer,
            debug_samples=target_samples, 
            num_ds_copy=1
        )
        print(f"Model for testing is loaded from {str(local_model_path)}")
        test_dataset = Dataset(test_data_tokenized, labels=test_labels)
        test_trainer = cls.get_trainer(model)
        
        # Make prediction
        raw_pred, _, _ = test_trainer.predict(test_dataset)
        
        # Preprocess raw predictions
        y_pred = np.argmax(raw_pred, axis=1)
        print(y_pred)
        pred_dict = {
            'input': target_samples,
            'predictions': y_pred.tolist()
        }
        test_file = model_dir / 'predictions.json'
        with open(test_file, "w") as out_f:
            json.dump(pred_dict, out_f, indent=4)
        # end with
        return

if __name__=='__main__':
    Debug.main(seed=9)
