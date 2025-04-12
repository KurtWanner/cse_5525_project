import numpy as np
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel


def load_model():
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)


    return model
    
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
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
