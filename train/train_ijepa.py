from data_pipeline import data_set , data_aug
from model import DRijepa , ijepa
from metrics import all_metrics

#torch
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader , DistributedSampler
from torchvision import datasets , transforms
from torch.cuda.amp import autocast , GradScaler
import wandb

#functional
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import os
import traceback


def cleanup():
    dist.destroy_process_group()


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
            max_ep=300,
            save_dir='checkpoints',
            log_interval=100,
            save_interval=10,
            use_amp=True
    ):
        
        #distributed model
        self.model = DDP(model.to(rank) , device_ids=[rank])
        self.loss_fn = loss_fn.to(rank)
        self.rank = rank
        self.world_size = world_size
        self.train_loader = train_loader
        self.optim = optim
        self.scheduler = scheduler
        self.max_ep = max_ep
        self.save_dir = Path(save_dir)
        self.log_intervel = log_interval
        self.save_intervel = save_interval
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

        #if GPU node => primary
        if self.rank == 0:
            self.save_dir.mkdir(exist_ok=True)
            wandb.init(project="ijepa-training" , name="training1")

    def save_checkpnt(self , epoch , loss) -> None:
        """
        save the model checkpoint
        arg:
            epoch = number of epochs
            loss = loss function
        return:
            None
        """

        if self.rank == 0:
            checkpoint = {
                'epoch' : epoch,
                'model_state_dict' : self.model.module.state_dict(),
                'optim_state_dict' : self.optim.state_dict(),
                'loss' : loss,
            }

            if self.scheduler is not None:

                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

            torch.save(
                checkpoint , 
                self.save_dir / f"checkpoint_epoch_{epoch}.pt"
            )


    def train_epoch(self , epoch):

        self.model.train()
        total_loss = 0
        n_batch = len(self.train_loader)

        if self.rank == 0:
            pbar = tqdm(total=n_batch , desc=f"Epoch :{epoch}")

        for batch_idx , img in enumerate(self.train_loader):

            img = img.to(self.rank)
            self.optim.zero_grad()


            with autocast(enabled=self.use_amp):
                pred , target = self.model(img)
                loss = self.loss_fn(pred , target)

            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)

                self.scaler.update()

            else:
                loss.backward()
                self.optim.step()


            self.model.momentum_update(
                self.model.target_encoder,
                self.model.context_encoder
            )


            total_loss += loss.item()
            if batch_idx % self.log_interval == 0 and self.rank == 0:

                wandb.log(
                    {
                    'batch_loss': loss.item(),
                    'epoch': epoch,
                    'batch': batch_idx
                    }   
                )
            
            if self.rank == 0:
                pbar.update(1)


        if self.rank == 0:
            pbar.close()


        
        avg_loss = total_loss / n_batch
        dist.all_reduce(torch.tensor(avg_loss).to(self.rank))

        avg_loss = avg_loss / self.world_size


        return avg_loss



    def train(self):

        best_loss = float('inf')

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


            if ep % self.save_intervel == 0 or loss < best_loss:
                self.save_checkpnt(ep , loss)
                if loss < best_loss:
                    best_loss = loss


def setup(rank , world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl" , rank=rank , world_size=world_size)


def main(rank , world_size):  
    try:  
        setup(rank, world_size)

        model , loss_fn = DRijepa.create_DRijepa(
            # img_size=224,
            # patch_size=16,
            # in_chans=3,
            # embed_dim=1024,  # Increased for A100s
            # encoder_depth=12,
            # predictor_depth=4,
            # num_heads=16,
            # mlp_ratio=4,
            # dropout=0.1
        )
                    
        batch_size = 128
        # train_dataset = data_set.UnitedTrainingDataset("eyepacs" , "aptos" , "ddr" ,  "idrid" ,transformation=data_aug.IJEPAAugmentation())
        # sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        # train_loader = DataLoader(train_dataset , batch_size=batch_size , sampler=sampler, pin_memory=True , num_workers=8)
        dataset_names = ["eyepacs" , "aptos" , "ddr" , "idrid"]
        uniform_data_ld = data_set.UniformTrainDataloader(
            dataset_names=dataset_names,
            transformation=data_aug.IJEPAAugmentation(),
            batch_size=batch_size,
            num_workers=4,
            sampler=True
        )

        data_ld = uniform_data_ld.get_loader()
        
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=1.5e-4,
            betas=(0.9 , 0.95),
            weight_decay=0.05
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=300,  # number of epochs
            eta_min=1e-6
        )
        

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            rank=rank,
            world_size=world_size,
            train_loader=data_ld,
            optim= optim,
            scheduler=scheduler,
            max_ep=300,
            save_dir='ijepa_checkpoints',
            log_interval=50,
            save_interval=5,
            use_amp=True
        )


        trainer.train()

        cleanup()

    except Exception as e:
        print(e)
        traceback.print_exc()
        



if __name__ == "__main__":
    world_size = 2

    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size
    )
    


            

