import numpy as np
import random
from tqdm import tqdm

import torch


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
