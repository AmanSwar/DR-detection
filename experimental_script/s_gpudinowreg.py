import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
import wandb
import os
from pathlib import Path

from train.dinowreg import ViTRegs , RetAug , DINOwithReg , DINOLoss
from data_pipeline.data_set import UniformTrainDataloader


def train_dino(
    output_dir='./checkpoints',
    img_size=512,
    batch_size=32,
    num_epochs=100,
    warmup_epochs=10,
    embed_dim=768,
    patch_size=16,
    num_registers=4,
    lr=0.0005,
    weight_decay=0.04,
    momentum=0.996,
    save_freq=10,
    use_wandb=True
):

    if use_wandb:
        wandb.init(project="dino-registers", config={
            "img_size": img_size,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "warmup_epochs": warmup_epochs,
            "embed_dim": embed_dim,
            "patch_size": patch_size,
            "num_registers": num_registers,
            "lr": lr,
            "weight_decay": weight_decay
        })

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    
    augmentor = RetAug(img_size=img_size)

    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    dataset_names = ["eyepacs" , "aptos" , "ddr" , "idrid"]

    uniform_data_ld = UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=augmentor,
        batch_size=batch_size,
        num_workers=4,
        sampler=True
    )

    data_ld = uniform_data_ld.get_loader()

    
    student = ViTRegs(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_regs=num_registers
    ).to(device)

    teacher = ViTRegs(

        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_regs=num_registers

    ).to(device)

    
    dino_model = DINOwithReg(

        student=student,
        teacher=teacher,
        embed_dim=embed_dim,
        momentum=momentum,
        num_reg=num_registers

    ).to(device)


    decay_params = []

    no_decay_params = []

    for name, param in dino_model.student.named_parameters():

        if 'bias' in name or 'norm' in name or 'register' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr * (batch_size / 256), betas=(0.9, 0.95))


    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=1e-6,

    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(

        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,

    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(

        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]

    )


    best_loss = float('inf')
    
    for epoch in range(num_epochs):

        dino_model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(data_ld, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for images, _ in progress_bar:
            images = images.to(device)
            
            
            view1, view2 = view1.to(device), view2.to(device)
            
            loss = dino_model(view1, view2)
            
            
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            
            
            total_loss += loss.item()
            num_batches += 1
            current_lr = optimizer.param_groups[0]['lr']
            
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            if use_wandb:
                wandb.log(
                    {
                    'batch_loss': loss.item(),
                    'learning_rate': current_lr
                    }
                )
        
        
        epoch_loss = total_loss / num_batches
        
        
        print(f'Epoch {epoch + 1} - Average Loss: {epoch_loss:.4f}')
        if use_wandb:
            wandb.log(
                {
                'epoch': epoch + 1,
                'epoch_loss': epoch_loss,
                }
            )
        
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch + 1,
                'student_state_dict': dino_model.student.state_dict(),
                'teacher_state_dict': dino_model.teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, output_dir / 'best_model.pth')
        
        
        if (epoch + 1) % save_freq == 0:
            torch.save({
                'epoch': epoch + 1,
                'student_state_dict': dino_model.student.state_dict(),
                'teacher_state_dict': dino_model.teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        
        scheduler.step()
    
    if use_wandb:
        wandb.finish()

    print("Training completed!")
    return dino_model


if __name__ == "__main__":
    train_config = {
        "output_dir": "./dino_checkpoints",
        "img_size": 512,
        "batch_size": 32,
        "num_epochs": 100,
        "warmup_epochs": 10,
        "embed_dim": 768,
        "patch_size": 16,
        "num_registers": 4,
        "lr": 0.0005,
        "weight_decay": 0.04,
        "momentum": 0.996,
        "save_freq": 10,
        "use_wandb": True
    }
    
    model = train_dino(**train_config)