import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from random import randrange
from utils import *
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import json
import json
import time
import csv
import tqdm
import numpy as np
import wandb
import pandas as pd
import math
from datetime import datetime
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nlgeval import NLGEval
from utils import *
from torch import cuda

def my_collate(batch):
    distractor_set = []
    question = [item[0] for item in batch]
    distractor = [item[1][0] for item in batch]
    for item in batch:
        d = item[1]
        random_index = randrange(len(d))
        distractor_set.append(d[random_index])
    # for item in batch:
    #    print(item[1])
    # target = torch.LongTensor(target)

    return question, distractor_set

class Race_Dataset(Dataset):

    def __init__(self, data_root, tokenizer, source_len=512, target_len=150):
        self.instances = []
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len

        self.prompt = "distract: "

        with open(data_root, "r") as f:
            data = json.load(f)
        for inst in tqdm(data, desc="Loading test-data"):
            distractors = []
            answer = inst['answer'].lower()
            question = inst['question'].lower()
            distractors_ = inst['distractors']
            for dist in distractors_:
                distractors.append(dist.lower())
            question_answer = self.prompt + question + " [SEP] " + answer
            self.instances.append((question_answer, distractors))


    def __len__(self):
        return len(self.instances)
    def __getitem__(self, idx):
        ctext, all_distractors = self.instances[idx]
        ctext = exceed_512_(self.tokenizer, ctext)

        random_index = randrange(len(all_distractors))
        text = all_distractors[random_index]

        ctext = ' '.join(ctext.split())
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length=self.source_len, pad_to_max_length=True,
                                                  return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length=self.target_len, pad_to_max_length=True,
                                                  return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

class Race_Dataset_test(Dataset):

    def __init__(self, data_root, tokenizer, source_len=512, target_len=150):
        self.max_candidate = 6
        self.instances = []
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len

        self.prompt = "distract: "

        with open(data_root, "r") as f:
            data = json.load(f)
        for inst in tqdm(data, desc="Loading test-data"):
            distractors = []
            answer = inst['answer'].lower()
            question = inst['question'].lower()
            distractors_ = inst['distractors']
            for dist in distractors_:
                distractors.append(dist.lower())
            question_answer = self.prompt + question + " [SEP] " + answer
            self.instances.append((question_answer, distractors))

    def __len__(self):
        return len(self.instances)
    def __getitem__(self, idx):
        ctext, all_distractors = self.instances[idx]
        ctext = exceed_512_(self.tokenizer, ctext)

        target_distractors = all_distractors   #should be returned in [distractor1, distractor2,..] format
        target_distractors = [normalize_text(t.strip()) for t in target_distractors]

        random_index = randrange(len(all_distractors))
        text = all_distractors[random_index]

        ctext = ' '.join(ctext.split())
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length=self.source_len, pad_to_max_length=True,
                                                  return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length=self.target_len, pad_to_max_length=True,
                                                  return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        if self.max_candidate - len(target_distractors) > 0 :
            target_distractors.extend([""] * (self.max_candidate - len(target_distractors)))

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'target_distractors' : target_distractors
        }

