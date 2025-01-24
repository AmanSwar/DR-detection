import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader , DistributedSampler
from torch.cuda.amp import autocast , GradScaler

from model import DRijepa ,DRijepa_scl

from pathlib import Path

class Trainer:

    def __init__(
            self,
            model,
            loss_fn,
            rank,
            world_size,
            train_loader,
            optim,
            scheduler=None,
            max_ep=500,
            save_dir = "checkpoints",
            log_interval=100,
            save_interval=10,
            use_amp=True
    ):
        
        
        self.model = DDP(model.to(rank) , device_ids=[rank])
        self.loss_fn = loss_fn.to(rank)
        self.rank = rank
        self.world_size = world_size
        self.train_loader = train_loader
        self.optim = optim
        self.scheduler = scheduler
        self.max_ep = max_ep
        self.save_dir = 'checkpoints'
        self.log_interval = 100
        self.save_interval = save_interval
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None


        if self.rank == 0:
            self.

