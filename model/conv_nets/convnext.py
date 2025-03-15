import torch
import torch.nn as nn
import timm
from data_pipeline.data_set import UniformTrainDataloader, UniformValidDataloader
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import wandb
import os
import signal
import sys
import time
from pathlib import Path
from data_pipeline.data_aug import MoCoSingleAug

dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]

class Convnext(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=6,
        )
        if pretrained_path:
            self.load_custom_checkpoint(pretrained_path)
            
    def load_custom_checkpoint(self, checkpoint_path):
        print(f"Loading custom pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Filter and adapt keys if needed (for ijepa/moco checkpoints)
        model_state_dict = self.model.state_dict()
        filtered_state_dict = {}
        
        for k, v in state_dict.items():
            # Remove module. prefix if present
            if k.startswith('module.'):
                k = k[7:]
                
            # Skip classifier weights as we have a different number of classes
            if 'head' in k or 'fc' in k or 'classifier' in k:
                continue
                
            # Check if key exists in model
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_state_dict[k].shape}")
            else:
                print(f"Key {k} not found in model state dict")
        
        # Load the filtered state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

    def forward(self, x):
        out = self.model(x)
        return out


train_augmentation = MoCoSingleAug(img_size=256)
valid_augmentation = MoCoSingleAug(img_size=256)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if config["use_wandb"]:
            wandb.init(
                project=config["wandb_project_name"],
                name=config["run_name"],
                config=config
            )
        
        # Setup data loaders
        self.setup_dataloaders()
        
        # Initialize model
        self.setup_model()
        
        # Setup optimizer and loss function
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=config["learning_rate"])
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Tracking variables
        self.start_epoch = 1
        self.best_val_f1 = 0.0
        self.global_step = 0
        
        # Checkpoint directory
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        
        if config["resume_checkpoint"]:
            self.load_checkpoint(config["resume_checkpoint"])
        
        signal.signal(signal.SIGINT, self.graceful_exit)
        signal.signal(signal.SIGTERM, self.graceful_exit)
        
    def setup_dataloaders(self):
        train_aug = MoCoSingleAug(img_size=256)
        val_aug = MoCoSingleAug(img_size=256)
        
        self.trainloader = UniformTrainDataloader(
            dataset_names=dataset_names,
            transformation=train_aug,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            sampler=True
        ).get_loader()
        
        self.validloader = UniformValidDataloader(
            dataset_names=dataset_names,
            transformation=val_aug,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            sampler=True
        ).get_loader()
        
        print(f"Training samples: {len(self.trainloader.dataset)}")
        print(f"Validation samples: {len(self.validloader.dataset)}")
    
    def setup_model(self):
        self.model = Convnext(pretrained_path=self.config["pretrained_path"])
        self.model.to(device=self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: ConvNeXt Tiny")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        # Progress bar
        progress_bar = tqdm(self.trainloader, desc=f"Epoch {epoch}/{self.config['num_epochs']} [Train]")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            running_loss += loss.item() * images.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log batch metrics to WandB
            if self.config["use_wandb"] and batch_idx % self.config["log_interval"] == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "global_step": self.global_step
                })
            
            self.global_step += 1
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.trainloader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Log epoch metrics
        if self.config["use_wandb"]:
            wandb.log({
                "train/epoch": epoch,
                "train/loss": epoch_loss,
                "train/accuracy": epoch_acc,
                "train/f1_score": epoch_f1
            })
        
        return epoch_loss, epoch_acc, epoch_f1
    
    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        # Progress bar
        progress_bar = tqdm(self.validloader, desc=f"Epoch {epoch}/{self.config['num_epochs']} [Valid]")
        
        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                
                # Update metrics
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                running_loss += loss.item() * images.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.validloader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Log epoch metrics
        if self.config["use_wandb"]:
            wandb.log({
                "valid/epoch": epoch,
                "valid/loss": epoch_loss,
                "valid/accuracy": epoch_acc,
                "valid/f1_score": epoch_f1
            })
        
        return epoch_loss, epoch_acc, epoch_f1
    
    def save_checkpoint(self, epoch, val_f1, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'global_step': self.global_step,
            'config': self.config
        }
        
        # Save periodic checkpoint
        checkpoint_path = os.path.join(self.config["checkpoint_dir"], f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save latest checkpoint (for resuming)
        latest_path = os.path.join(self.config["checkpoint_dir"], "checkpoint_latest.pt")
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if this is the best model so far
        if is_best:
            best_path = os.path.join(self.config["checkpoint_dir"], "checkpoint_best.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model saved with validation F1: {val_f1:.4f}")
        
        # Log checkpoint to WandB
        if self.config["use_wandb"]:
            wandb.save(checkpoint_path)
            
        # Clean up old checkpoints (keep only last 3 periodic checkpoints)
        checkpoint_files = sorted([
            f for f in os.listdir(self.config["checkpoint_dir"]) 
            if f.startswith("checkpoint_epoch_")
        ])
        
        if len(checkpoint_files) > 3:
            for old_file in checkpoint_files[:-3]:
                os.remove(os.path.join(self.config["checkpoint_dir"], old_file))
                print(f"Removed old checkpoint: {old_file}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training"""
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_f1 = checkpoint['best_val_f1']
        self.global_step = checkpoint['global_step']
        
        print(f"Resumed from epoch {self.start_epoch - 1}")
        print(f"Best validation F1 so far: {self.best_val_f1:.4f}")
    
    def graceful_exit(self, signum, frame):
        """Handle graceful exit by saving a checkpoint"""
        print("\nReceived signal to terminate. Saving checkpoint...")
        self.save_checkpoint(self.start_epoch, self.best_val_f1)
        print("Checkpoint saved. Exiting gracefully.")
        sys.exit(0)
    
    def train(self):
        print("\n" + "="*50)
        print(f"Starting training from epoch {self.start_epoch}")
        print("="*50 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config["num_epochs"] + 1):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc, train_f1 = self.train_one_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc, val_f1 = self.validate(epoch)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{self.config['num_epochs']} - Time: {epoch_time:.2f}s - Total: {total_time/60:.2f}m")
            print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
            print(f"Valid - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            print(f"{'='*80}\n")
            
            # Check if this is the best model
            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
            
            # Save checkpoints
            if epoch % self.config["save_interval"] == 0 or epoch == self.config["num_epochs"] or is_best:
                self.save_checkpoint(epoch, val_f1, is_best)
        
        # Final message
        total_time_mins = (time.time() - start_time) / 60
        print(f"\nTraining completed in {total_time_mins:.2f} minutes")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        
        # Close WandB
        if self.config["use_wandb"]:
            wandb.finish()


if __name__ == "__main__":
    config = {
        # Model parameters
        "pretrained_path": "checkpoint/trial_0/checkpoint_ep_40.pt",  # Set to None if not using pretrained
        
        # Training parameters
        "num_epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "num_workers": 4,
        
        # Checkpoint parameters         
        "checkpoint_dir": "checkpoints/convnext_dr_classification/ijepa",
        "save_interval": 20,  # Save checkpoint every N epochs  
        "resume_checkpoint": None,  # Set to checkpoint path if resuming
        
        # Logging parameters
        "use_wandb": True,
        "wandb_project_name": "supervised_convnext",
        "run_name": f"convnext_tiny_{time.strftime('%Y%m%d_%H%M%S')}",
        "log_interval": 10,  # Log every N batches
    }
    
    # Initialize and start training
    trainer = Trainer(config)
    trainer.train() 