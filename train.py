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
from distribute import *
from accelerate import Accelerator

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


PAD_IDX = 0
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load tokenizer
    tokenizer = load_tokenizer(model_args, data_args, training_args)
    k = int(data_args.kmer)

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                         data_path=os.path.join(data_args.data_path, f'k{k}', data_path('train', k)), 
                         kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                         data_path=os.path.join(data_args.data_path, f'k{k}', data_path('dev', k)), 
                         kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                         data_path=os.path.join(data_args.data_path, f'k{k}', data_path('test', k)), 
                         kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # load model
    model = load_model(model_args, data_args, training_args)

    # define trainer
    trainer = transformers.Trainer(model=model,
                tokenizer=tokenizer,
                args=training_args,
                compute_loss_func=loss_func,
                compute_metrics=compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator)
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)


def data_path(type, kmer):
    return type + "_token.csv"


if __name__ == "__main__":
    train()
