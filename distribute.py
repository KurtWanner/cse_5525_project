import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):

    # initialize the process group
    dist.init_process_group("ncll", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()