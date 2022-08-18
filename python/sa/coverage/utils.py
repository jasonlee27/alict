
from typing import *
from pathlib import Path

import os
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertModel

from src import *


def load_model(task_id):
    key_list = []
    if task_id == 0:
        model_name = "textattack/bert-base-uncased-SST-2"
        for i in range(11):
            key = 'bert.encoder.layer.' + str(i) + '.output'
            key_list.append(key)
            key = 'bert.encoder.layer.' + str(i) + '.output.dense'
            key_list.append(key)
            key = 'bert.encoder.layer.' + str(i) + '.output.LayerNorm'
            key_list.append(key)
            key = 'bert.encoder.layer.' + str(i) + '.output.dropout'
            key_list.append(key)

    elif task_id == 1:
        model_name = 'textattack/roberta-base-SST-2'
        for i in range(11):
            key = 'roberta.encoder.layer.' + str(i) + '.attention.output.dense'
            key_list.append(key)
            key = 'roberta.encoder.layer.' + str(i) + '.intermediate.dense'
            key_list.append(key)
            key = 'roberta.encoder.layer.' + str(i) + '.attention.output'
            key_list.append(key)
            key = 'roberta.encoder.layer.' + str(i) + '.output.dropout'
            key_list.append(key)
            key = 'roberta.encoder.layer.' + str(i) + '.attention.self.dropout'
            key_list.append(key)
            key = 'roberta.encoder.layer.' + str(i) + '.output.dense'
            key_list.append(key)
            key = 'roberta.encoder.layer.' + str(i) + '.attention.self.value'
            key_list.append(key)
            key = 'roberta.encoder.layer.' + str(i) + '.attention.output.LayerNorm'
            key_list.append(key)
            key = 'roberta.encoder.layer.' + str(i) + '.output'
            key_list.append(key)
            key = 'roberta.encoder.layer.' + str(i) + '.intermediate'
            key_list.append(key)
            key = 'roberta.encoder.layer.' + str(i) + '.output.LayerNorm'
            key_list.append(key)
            key = 'roberta.encoder.layer.' + str(i) + '.attention.self.key'
            key_list.append(key)
            key = 'roberta.encoder.layer.' + str(i) + '.attention.self.query'
            key_list.append(key)
    elif task_id == 2:
        model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
        for i in range(6):
            key = 'distilbert.transformer.layer.' + str(i) + '.attention.q_lin'
            key_list.append(key)
            key = 'distilbert.transformer.layer.' + str(i) + '.attention.k_lin'
            key_list.append(key)
            key = 'distilbert.transformer.layer.' + str(i) + '.attention.v_lin'
            key_list.append(key)
            key = 'distilbert.transformer.layer.' + str(i) + '.attention.out_lin'
            key_list.append(key)
            key = 'distilbert.transformer.layer.' + str(i) + '.sa_layer_norm'
            key_list.append(key)
            key = 'distilbert.transformer.layer.' + str(i) + '.ffn'
            key_list.append(key)
            key = 'distilbert.transformer.layer.' + str(i) + '.ffn.dropout'
            key_list.append(key)
            key = 'distilbert.transformer.layer.' + str(i) + '.ffn.lin2'
            key_list.append(key)
            key = 'distilbert.transformer.layer.' + str(i) + '.output_layer_norm'
            key_list.append(key)
            key = 'distilbert.transformer.layer.' + str(i) + '.ffn.lin1'
            key_list.append(key)
    else:
        raise NotImplemented
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return model, tokenizer, key_list


def load_test_suite(task_id,
                    our_sents_file,
                    checklist_sent_file):
    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    data_dir = this_dir / 'data'
    sst2_dir = data_dir / 'archive/SST2-Data/SST2-Data/stanfordSentimentTreebank/stanfordSentimentTreebank'
    train_data_file = os.path.join(str(sst2_dir), 'datasetSentences.txt')
    with open(train_data_file, 'r') as f:
        train_dataset = f.readlines()
        train_dataset = [d.replace('\n', '').split('\t')[1] for d in train_dataset[1:]]
    # with open('data/our_sents.txt', 'r') as f:
    with open(str(data_dir / our_sents_file), 'r') as f:
        our_dataset = f.readlines()
        our_dataset = [d.replace('\n', '') for d in our_dataset]
    with open(str(data_dir / checklist_sent_file), 'r') as f:
        checklist_data = f.readlines()
        checklist_data = [d.replace('\n', '') for d in checklist_data]

    return train_dataset, our_dataset, checklist_data
