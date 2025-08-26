import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchssl.framework.ssl import SSL
from tqdm import tqdm
from torchssl.loss.python.dinoloss import DINOLoss


class DinoModel(nn.Module):
    def __init__(
            self,
            backbone_model,
            projection_dim,
            hidden_dim,
            bottleneck_dim,
            teacher_temp,
            student_temp,
            center_mom=0.9
    ):
        super().__init__()

        self.backbone_model = backbone_model
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.center_momentum = center_mom
        
        # Initialize center for teacher
        self.register_buffer("center", torch.zeros(1, projection_dim))

        # Create student and teacher separately (not sharing projection head)
        self.student = self._create_encoder()
        self.teacher = self._create_encoder()

        # Initialize teacher weights from student + no gradients for teacher
        for s_param, t_param in zip(self.teacher.parameters(), self.student.parameters()):
            s_param.data.copy_(t_param.data)
            s_param.requires_grad_(False)

    def _create_encoder(self):
        feature_dim = self.backbone_model.get_features()

        projection_head = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim, eps=1e-6),
            nn.Linear(self.bottleneck_dim, self.projection_dim)
        )

        # Create a new instance of backbone for student/teacher to avoid weight sharing
        backbone_copy = type(self.backbone_model)(*self.backbone_model.__init__args__)
        return nn.Sequential(backbone_copy, projection_head)

    def forward(self, x):
        student_out = self.student(x)
        # Normalize along last dimension
        student_out = F.normalize(student_out, dim=-1)

        with torch.no_grad():
            teacher_out = self.teacher(x)
            teacher_out = F.normalize(teacher_out, dim=-1)

        return student_out, teacher_out

    @torch.no_grad()
    def update_teacher(self, m):
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = param_t.data * m + param_s.data * (1. - m)

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class Dino(SSL):
    def __init__(
            self,
            backbone_model,
            out_dim,
            projection_dim,
            hidden_dim,
            bottleneck_dim,
            teacher_temp,
            student_temp,
            center_mom,
            ncrops,
            warmup_teacher_temp,
            device, 
            wandb_run
            ):
        super().__init__(device, wandb_run)

        self.model = DinoModel(
            backbone_model=backbone_model,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            teacher_temp=teacher_temp,
            student_temp=student_temp,
            center_mom=center_mom
        ).to(device)

        self.loss_fn = None
        self.ncrops = ncrops

    def train_one_epoch(
            self,
            dataloader,
            optimizer,
            scheduler,
            epoch,
            m_teacher_momentum_schedule,
    ):
        self.model.train()
        running_loss = 0

        for i, images in enumerate(dataloader):
            # Move image crops to device
            images = [im.to(self.device) for im in images]
            
            # Get teacher momentum for current step
            momentum_val = m_teacher_momentum_schedule[epoch * len(dataloader) + i]
            
            # Process all images through the model
            student_outputs = []
            teacher_outputs = []
            
            # Process global views (first two crops are global views)
            for idx, img in enumerate(images):
                student_out, teacher_out = self.model(img)
                student_outputs.append(student_out)
                if idx < 2:  # Only use teacher outputs from global views
                    teacher_outputs.append(teacher_out)
            
            # Calculate loss using the DINOLoss
            loss = self.loss_fn(student_outputs, teacher_outputs, epoch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update teacher with momentum
            self.model.update_teacher(momentum_val)
            
            # Update center for loss normalization
            self.model.update_center(torch.cat(teacher_outputs))
            
            running_loss += loss.item()
        
            if i % 10 == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                logging.info(f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] "
                           f"Loss: {loss.item():.4f}, LR: {learning_rate:.6f}")
                if self.wandb_run:
                    self.wandb_run.log({
                        "train_loss": loss.item(), 
                        "learning_rate": learning_rate,
                        "epoch": epoch + i/len(dataloader)
                    })
    
        avg_loss = running_loss / len(dataloader)
        return avg_loss

    def validate(
            self,
            dataloader,
            temperature,
            epoch
    ):
        self.model.eval()
        running_loss = 0

        with torch.no_grad():
            for i, views in tqdm(enumerate(dataloader)):
                views = [v.to(self.device) for v in views]

                student_outputs = []
                teacher_outputs = []

                for idx, view in enumerate(views):
                    student_out, teacher_out = self.model(view)
                    student_outputs.append(student_out)
                    if idx < 2:  # Only use teacher outputs from global views
                        teacher_outputs.append(teacher_out)

                # Calculate validation loss
                loss = self.loss_fn(student_outputs, teacher_outputs, epoch)
                running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        logging.info(f"Epoch [{epoch+1}] Validation Loss: {avg_loss:.4f}")
        if self.wandb_run:
            self.wandb_run.log({"val_loss": avg_loss, "epoch": epoch+1})
        return avg_loss

    def fit(
            self,
            train_loader,
            valid_loader,
            num_epochs,
            optimizer,
            lr,
            schedulers,
            out_dim,
            ncrops,
            warmup_teacher_temp,
            teacher_temp,
            warmup_teacher_temp_epochs,
            student_temp=0.1,
            center_momentum=0.9,
            evaluation_epoch=5,
            save_checkpoint=0,
            checkpoint_dir=None,
            mixed_precision=False,
            warmup_scheduler_epoch=0,
            lr_min=0,
            ):
        
        # Initialize DINO loss
        self.loss_fn = DINOLoss(
            out_dim=out_dim,
            ncrops=ncrops,
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            nepochs=num_epochs,
            student_temp=student_temp,
            center_momentum=center_momentum
        )

        if warmup_scheduler_epoch > 0:
            initial_lr = lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_min if lr_min != 0 else lr / 10
                
        # Create teacher momentum schedule
        teacher_momentum = np.linspace(0.996, 0.9995, num_epochs * len(train_loader))
        
        best_val_loss = float('inf')
        scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        
        for epoch in range(num_epochs):
            # Apply warmup scheduler if needed
            if epoch < warmup_scheduler_epoch:
                lr_factor = (epoch + 1) / warmup_scheduler_epoch
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_min + (initial_lr - lr_min) * lr_factor
            
            # Train one epoch
            train_loss = self.train_one_epoch(
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=schedulers,
                epoch=epoch,
                m_teacher_momentum_schedule=teacher_momentum,
            )
            
            # Step scheduler
            if schedulers is not None:
                if isinstance(schedulers, list):
                    for scheduler in schedulers:
                        scheduler.step()
                else:
                    schedulers.step()
            
            # Validation
            if epoch % evaluation_epoch == 0 or epoch == num_epochs - 1:
                val_loss = self.validate(
                    dataloader=valid_loader,
                    temperature=teacher_temp,
                    epoch=epoch
                )
                
                # Save checkpoint if needed
                if save_checkpoint > 0 and (epoch % save_checkpoint == 0 or epoch == num_epochs - 1):
                    if checkpoint_dir is not None:
                        if not os.path.exists(checkpoint_dir):
                            os.makedirs(checkpoint_dir)
                        
                        checkpoint_path = os.path.join(checkpoint_dir, f"dino_checkpoint_epoch_{epoch}.pt")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                        }, checkpoint_path)
                        logging.info(f"Checkpoint saved at {checkpoint_path}")
                
                # Track best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if checkpoint_dir is not None:
                        best_model_path = os.path.join(checkpoint_dir, "dino_best_model.pt")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                        }, best_model_path)
                        logging.info(f"Best model saved with validation loss: {val_loss:.4f}")
        
        return best_val_loss
