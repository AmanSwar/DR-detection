import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import time
import timm
import os

wandb.init()

class IJEPALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted_features, target_features):
        predicted_features = F.normalize(predicted_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)
        sim = torch.einsum('bnc, bnc -> bn', predicted_features, target_features)
        loss = -sim.mean()
        return loss

class DRIjepa(nn.Module):
    def __init__(
        self,
        model_name='convnext_base',  
        img_size=256,
        n_box=6,
        dropout=0.1
    ):
        super().__init__()

        # Create two ConvNeXt backbones
        self.context_encoder = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool=''  
        )
        self.target_encoder = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool=''
        )

        # Get the output channel dimension from ConvNeXt
        self.out_dim = self.context_encoder.feature_info[-1]['num_chs'] \
            if hasattr(self.context_encoder, 'feature_info') else self.context_encoder.num_features

        # Predictor head
        self.predictor = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1)
        )

        self.n_box = n_box
        self.img_size = img_size

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def get_random_boxes(self, batch_size, H, n_box=None):
        """Generate random box coordinates for each item in the batch."""
        if n_box is None:
            n_box = self.n_box
        boxes = []
        for b in range(batch_size):
            batch_boxes = []
            for _ in range(n_box):
                # Generate a center with a small random offset
                center_bias = torch.randn(2) * 0.5
                x_center = H // 2 + int(center_bias[0] * H // 4)
                y_center = H // 2 + int(center_bias[1] * H // 4)
                # Random box size (ensure at least size 1)
                w = torch.randint(max(1, H // 8), max(2, H // 4), (1,)).item()
                h = torch.randint(max(1, H // 8), max(2, H // 4), (1,)).item()
                x1 = max(0, x_center - w // 2)
                y1 = max(0, y_center - h // 2)
                x2 = min(H, x1 + w)
                y2 = min(H, y1 + h)
                batch_boxes.append([x1, y1, x2, y2])
            boxes.append(batch_boxes)
        return torch.tensor(boxes, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def extract_target(self, feature, boxes):
        """Extract features corresponding to each box from the feature map."""
        B, C, H, W = feature.shape
        target_features = []
        for b in range(B):
            batch_targets = []
            for box in boxes[b]:
                x1, y1, x2, y2 = box
                target = feature[b:b+1, :, y1:y2, x1:x2]
                target = F.adaptive_avg_pool2d(target, (1, 1))
                batch_targets.append(target)
            batch_targets = torch.cat(batch_targets, dim=0)
            batch_targets = batch_targets.view(-1, C)
            target_features.append(batch_targets)
        target_features = torch.stack(target_features)
        return target_features

    @torch.no_grad()
    def momentum_update(self, momentum=0.999):
        """Update the target encoder with momentum."""
        for target_param, context_param in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            target_param.data.mul_(momentum).add_((1 - momentum) * context_param.data)

    def forward(self, images, boxes=None):
        B = images.shape[0]
        
        # Get context features (ConvNeXt outputs differently from Swin)
        context_feats = self.context_encoder.forward_features(images)  # (B, C, H, W)
        H = context_feats.shape[2]

        if boxes is None:
            boxes = self.get_random_boxes(B, H)

        # Apply predictor head to context features
        pred_feats = self.predictor(context_feats)
        pred_feats = self.extract_target(pred_feats, boxes)

        # Compute target features without gradient
        with torch.no_grad():
            target_feats = self.target_encoder.forward_features(images)
            target_feats = self.extract_target(target_feats, boxes)

        return pred_feats, target_feats

def create_DRijepa(
    img_size=256,
    model_name='convnext_base',  # Changed default to ConvNeXt
    n_box=6,
    dropout=0.1
):
    model = DRIjepa(
        model_name=model_name,
        img_size=img_size,
        n_box=n_box,
        dropout=dropout
    )
    loss = IJEPALoss()
    return model, loss

class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        train_loader,
        val_loader=None,
        max_ep=300,
        save_dir="checkpoint",
        log_interval=100,
        save_interval=10,
        device="cuda"
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Modified optimizer settings for ConvNeXt
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,  # Slightly lower learning rate for ConvNeXt
            betas=(0.9, 0.95),
            weight_decay=0.05
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max=300,
            eta_min=1e-6
        )
        self.max_ep = max_ep
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.device = device

    def save_checkpoint(self, epoch, loss):
        model_state = self.model.state_dict() if not hasattr(self.model, 'module') else self.model.module.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optim_state_dict': self.optim.state_dict(),
            'loss': loss,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(self.save_dir, f"checkpoint_ep_{epoch}.pt"))
        print(f"Checkpoint saved for epoch {epoch} with loss {loss:.4f}")

    def log_attention_map(self, epoch):
        """Log feature visualization from ConvNeXt."""
        self.model.eval()
        sample = next(iter(self.train_loader))
        sample = sample.to(self.device)

        with torch.no_grad():
            features = self.model.context_encoder.forward_features(sample)
            feature_map = features.mean(dim=1, keepdim=True)
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            feature_map = F.interpolate(
                feature_map, 
                size=(self.model.img_size, self.model.img_size), 
                mode='bilinear', 
                align_corners=False
            )
            feature_img = feature_map[0].squeeze().cpu().numpy()

        wandb.log({
            f"Feature_Map_Epoch_{epoch}": wandb.Image(
                feature_img, 
                caption=f"ConvNeXt Feature Map Epoch {epoch}"
            )
        })
        self.model.train()

    def train_epoch(self, epoch):
        self.model.train()
        n_batch = len(self.train_loader)
        total_loss = 0
        pbar = tqdm(total=n_batch, desc=f"Epoch: {epoch}")

        for batch_idx, batch in enumerate(self.train_loader):
            img = batch.to(self.device)
            self.optim.zero_grad()

            pred_feat, target_feat = self.model(img)
            loss = self.loss_fn(pred_feat, target_feat)
            loss.backward()
            self.optim.step()

            self.model.momentum_update()

            total_loss += loss.item()

            wandb.log({
                'batch_loss': loss.item(),
                'epoch': epoch,
                'batch': batch_idx
            })

            print(f"BATCH LOSS : {loss}")
            pbar.update()

        pbar.close()
        avg_loss = total_loss / n_batch
        return avg_loss

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        n_batch = len(self.val_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                img = batch.to(self.device)
                pred_feat, target_feat = self.model(img)
                loss = self.loss_fn(pred_feat, target_feat)
                total_loss += loss.item()
        avg_val_loss = total_loss / n_batch
        wandb.log({'val_loss': avg_val_loss, 'epoch': epoch})
        print(f"Validation Epoch {epoch}: Loss = {avg_val_loss:.4f}")
        return avg_val_loss

    def train(self):
        best_loss = float("inf")
        best_val_loss = float("inf") if self.val_loader is not None else None
        self.model.to(self.device)
        try:
            for ep in range(self.max_ep):
                epoch_start_time = time.time()
                loss = self.train_epoch(ep)
                ep_dur = time.time() - epoch_start_time

                if self.scheduler is not None:
                    self.scheduler.step()

                wandb.log({
                    'epoch_loss': loss,
                    'epoch': ep,
                    'learning_rate': self.optim.param_groups[0]['lr'],
                    'epoch_duration': ep_dur
                })

                print(f"Epoch {ep}: Loss = {loss:.4f}, Time = {ep_dur:.2f}s")

                if self.val_loader is not None:
                    val_loss = self.validate_epoch(ep)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint(ep, val_loss)

                if ep % self.save_interval == 0 or loss < best_loss:
                    self.save_checkpoint(ep, loss)
                    if loss < best_loss:
                        best_loss = loss

                if ep % 15 == 0:
                    self.log_attention_map(ep)

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Saving checkpoint and exiting...")
            self.save_checkpoint(ep, best_loss)

if __name__ == "__main__":
    # Your data loading code remains the same
    from data_pipeline.data_set import SSLTrainLoader, SSLValidLoader
    from data_pipeline.data_aug import IJEPAAugmentation

    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    transforms_ = IJEPAAugmentation(img_size=256)
    
    train_loader = SSLTrainLoader(
        dataset_names=dataset_names,
        transformation=transforms_,
        batch_size=16,
        num_work=4,
    ).get_loader()

    valid_loader = SSLValidLoader(
        dataset_names=dataset_names,
        transformation=transforms_,
        batch_size=16,
        num_work=4,
    ).get_loader()

    model, loss_fn = create_DRijepa(
        img_size=256,
        model_name='convnext_base',
        n_box=6,
        dropout=0.1
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=valid_loader,
        val_loader=valid_loader,
    )

    trainer.train()