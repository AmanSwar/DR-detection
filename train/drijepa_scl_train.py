import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader , DistributedSampler
from torch.cuda.amp import autocast , GradScaler
import wandb
from model import DRijepa ,DRijepa_scl

from pathlib import Path
from tqdm import tqdm
import time
import os

from model.DRijepa_scl import train_step
from data_pipeline import data_set , data_aug

def cleanup():
    dist.destroy_process_group()


class AdaptiveLossWeightScheduler:
    
    def __init__(
            self,
            initial_con_weight=0.7,
            min_con_weight=0.3,
            decay_epochs=50
    ):
        
        self.current_con_weight = initial_con_weight
        self.min_con_weight = min_con_weight
        self.decay_epochs = decay_epochs

    def step(self , epoch , metric=None):
        """
        gradually decrese contrastive weight over time
        """
        
        decay = min(1.0 , epoch / self.decay_epochs)

        self.current_con_weight = max(
            self.min_con_weight,
            self.current_con_weight * (1 - decay * 0.1)
        )

        return self.current_con_weight
    
    def train_step(model , image , label , optim , sup_con_loss_fn , ce_loss_fn  ,con_weight):
        feautes , projection , logits = model(image , return_features=True)

        sup_con_loss = sup_con_loss_fn(projection , label)
        ce_loss = ce_loss_fn(logits , label)

        total_loss = con_weight * sup_con_loss + (1 - con_weight) * ce_loss

        return total_loss , sup_con_loss , ce_loss
    


class Trainer:

    def __init__(
            self,
            model,
            sup_con_loss,
            ce_loss,
            con_weight,
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
        self.sup_con_loss = sup_con_loss.to(rank)
        self.ce_loss = ce_loss.to(rank)
        self.con_weight = con_weight
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
            self.save_dir.mkdir(exist_ok=True)
            wandb.init(project="drijepa_scl_training" , name="training..(1)")

        
    def save_checkpoint(self , epoch , loss) -> None:
        """
        save model checkpoint
        args:
            epoch (int) -> number of epoch
            loss = loss function
        return 
            None

        """

        if self.rank == 0:
            checkpoint = {
                'epoch' : epoch,
                ' model_state' : self.model.module.state_dict(),
                'optim_state'  : self.optim.state_dict(),
                'loss' : loss,
            }

            if self.scheduler is not None:
                checkpoint['scheduler_state'] =  self.scheduler.state_dict()

            torch.save(checkpoint , self.save_dir/f"checkpoint_ep_{epoch}.pt")

    def train_epoch(self , epoch):

        self.model.train()
        total_loss = 0
        total_sup_con_loss = 0
        total_ce_loss = 0
        n_batch = len(self.train_loader)

        if self.rank == 0:
            pbar = tqdm(total=n_batch , desc=f"Epoch : {epoch}")

        for batch_idx , (img , label ) in enumerate(self.train_loader):

            img = img.to(self.rank)
            label = label.to(self.rank)
            self.optim.zero_grad()


            with autocast(enabled=self.use_amp):
                t_loss , sup_con_loss , ce_loss = train_step(
                    model=self.model , 
                    image=img , 
                    label=label , 
                    optim=self.optim , 
                    sup_con_loss_fn=self.sup_con_loss,
                    ce_loss_fn=self.ce_loss,
                    con_weight=self.con_weight
                    )
                
            if self.use_amp:
                self.scaler.scale(t_loss).backward()
                self.scaler.step(self.optim)

                self.scaler.update()


            else:
                t_loss.backward()
                self.optim.step()

            total_loss += t_loss.item()
            total_sup_con_loss += sup_con_loss.item()
            total_ce_loss += ce_loss.item()

            if batch_idx % self.log_interval == 0 and self.rank == 0:
                
                wandb.log(
                    {
                        "batch_loss" : t_loss.item(),
                        'epoch' : epoch,
                        'batch' : batch_idx

                    }
                )

                print(f'Batch [{batch_idx}/{len(self.train_loader)}], '
                      f'Loss: {t_loss.item():.4f}, '
                      f'SupCon Loss: {sup_con_loss.item():.4f}, '
                      f'CE Loss: {ce_loss.item():.4f}')

            if self.rank == 0:
                pbar.update(1)


        if self.rank == 0:
            pbar.close()


        avg_loss = total_loss / n_batch

        dist.all_reduce(torch.tensor(avg_loss).to(self.rank))

        avg_loss = avg_loss / self.world_size

        return avg_loss
    
    
    def train(self):

        best_loss = float("inf")

        for ep in range(self.max_ep):
            self.train_loader.sampler.set_epoch(ep)

            epoch_start_time = time.time()
            loss = self.train_epoch(ep)
            epoch_duration = time.time() - epoch_start_time


            if self.scheduler is not None:
                self.scheduler.step()


            if self.rank == 0:
                wandb.log(
                    {
                        'epoch_loss' : loss,
                        'epoch' : ep,
                        'learning_rate': self.optim.param_groups[0]['lr'],
                        'epoch_duration' : epoch_duration
                    }
                )

                print(f'Epoch {ep}: Loss = {loss:.4f}, Time = {epoch_duration:.2f}s')

            if ep % self.save_interval == 0 or loss < best_loss:
                self.save_checkpoint(ep , loss)
                if loss < best_loss:
                    best_loss = loss


def setup(rank , world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl" , rank=rank , world_size=world_size)


def main(rank , world_size):
    setup(rank , world_size)

    model = DRijepa_scl.DRSupConModel(
        ijepa_model=None,
        num_classes=5,
        proj_dim=128,
        dropout=True
    )

    sup_con_loss = DRijepa_scl.SubConLoss().to(rank)
    ce_loss = nn.CrossEntropyLoss().to(rank)

    batch_size = 64

    train_dataset = data_set.UnitedTrainingDataset("eyepacs" , "aptos" , "ddr" ,  "idrid" ,transformation=data_aug.IJEPAAugmentation())
    sampler = DistributedSampler(train_dataset , num_replicas=world_size , rank=rank)
    train_loader = DataLoader(train_dataset , batch_size=batch_size , sampler=sampler , pin_memory=True , num_workers=8)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-4,
        betas=(0.9 , 0.95),
        weight_decay=0.05
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=300,
        eta_min=1e-6
    )

    trainer = Trainer(
        model=model,
        sup_con_loss=sup_con_loss,
        ce_loss=ce_loss,
        con_weight=0.5,
        rank=rank,
        world_size=world_size,
        train_loader=train_loader,
        optim=optim,
        scheduler=scheduler,
        max_ep=300,
        save_dir="drijepa_scl_checkpoints",
        log_interval=50,
        save_interval=5,
        use_amp=True
    )


    trainer.train()

    cleanup()