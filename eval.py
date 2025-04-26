import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

from DNA_metrics import *
from load_data import *
from utils import *
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
from accelerate import Accelerator
from rna_seq_information_trans import eval_rna
from tqdm import tqdm
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


PAD_IDX = 0
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def reconstruct(arr):
    arr = arr.split()
    s = arr[0]
    for i in range(1, len(arr)):
        s += arr[i][-1]

    return s

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load tokenizer
    tokenizer = load_tokenizer(model_args, data_args, training_args)
    k = int(data_args.kmer)

    # define datasets and data collator
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                         data_path=os.path.join(data_args.data_path, f'k{k}', data_path('dev', k)), 
                         kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                         data_path=os.path.join(data_args.data_path, f'k{k}', data_path('test', k)), 
                         kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # load model
    model = load_model(model_args, data_args, training_args)
    model.to(DEVICE)
    f_out = open('results/' + 'test_' + str(k) + 'mer.tsv', 'w')
    for x in tqdm(test_dataset):
        
        output = model.generate(
            input_ids = torch.unsqueeze(x['input_ids'].to(DEVICE), 0),
            attention_mask = torch.unsqueeze(x['attention_mask'].to(DEVICE), 0),
            decoder_input_ids = torch.unsqueeze(x['decoder_input_ids'][:1].to(DEVICE), 0),
            max_new_tokens=512,
        )
        
        actual = tokenizer.batch_decode(output, skip_special_tokens=True)
        expected = tokenizer.batch_decode([x['labels'].to(DEVICE)], skip_special_tokens=True)
        
        r_act = reconstruct(actual[0])
        r_exp = reconstruct(expected[0])
        print(r_act + '\t' + r_exp, file=f_out)
    f_out.close()

def data_path(type, kmer):
    return type + "_token.csv"


if __name__ == "__main__":
    main()
