import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

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

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

PAD_IDX = 0

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def generate_kmer(texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    return [generate_kmer_str(text, k) for text in texts] 

"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        print("Starting to load data: ", data_path)

        sp1 = []
        sp2 = []
	
        i = 0
        # load data from the disk
        with open(data_path, "r") as f:
            for line in tqdm(f.readlines()):
                i += 1
                if i % 5 != 0:
                    continue
                arrs = [x.split(',') for x in line.split('\t')]
                sp1.append(torch.tensor([[int(x) for x in arrs[0][0:512]]]))
                sp2.append(torch.tensor([[int(x) for x in arrs[1][0:512]]]))
        
        print("Loaded all data.")
        """
        species1 = [d[0] for d in data]
        species2 = [d[1] for d in data]

        output1 = [tokenizer(
            x,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for x in species1]

        output2 = [tokenizer(
            x,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for x in species2]

        print("Tokenized inputs.")
        """
        self.input_ids = sp1
        self.attention_mask = [torch.ones_like(x) for x in sp1]
        self.decoder_input_ids = sp2
        self.labels = [torch.cat((x[:,1:], torch.tensor([[PAD_IDX]])), dim=1) for x in sp2]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i][0], 
            attention_mask=self.attention_mask[i][0],
            decoder_input_ids = self.decoder_input_ids[i][0],
            labels=self.labels[i][0]
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, decoder_input_ids, labels = tuple([instance[key] for instance in instances] for key in ["input_ids", "attention_mask", "decoder_input_ids", "labels"])
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=PAD_IDX
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=PAD_IDX
        )
        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            decoder_input_ids, batch_first=True, padding_value=PAD_IDX
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=PAD_IDX
        )
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
