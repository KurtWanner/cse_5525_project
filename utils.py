import numpy as np
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, T5ForConditionalGeneration
from transformers.models.bert.configuration_bert import BertConfig

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_model(model_args, data_args, training_args):

    if 'google-t5' in model_args.model_name_or_path:
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, config=config)
    
    tokenizer = load_tokenizer(model_args, data_args, training_args)
    model.resize_token_embeddings(len(tokenizer))
    
    return model.to(DEVICE)
    
def load_tokenizer(model_args, data_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(
        "./kmer_tokenizers/" + str(data_args.kmer) + "mer.model",
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    return tokenizer

def set_random_seeds(seed_value=42):
    '''
    Set random seeds for better reproducibility
    '''
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
