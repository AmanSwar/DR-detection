import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import time
import timm  # make sure to install timm: pip install timm
from model.utils import RearrangeAndLayerNorm, vit_config
import os

from model.utils import swin_config

wandb.init()

class IJEPALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted_features, target_features):
        # predicted_features & target_features: (B, n_box, C)
        predicted_features = F.normalize(predicted_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)
        # Compute similarity per box per batch item
        sim = torch.einsum('bnc, bnc -> bn', predicted_features, target_features)
        loss = -sim.mean()
        return loss

class DRIjepa(nn.Module):
    def __init__(
        self,
        model_name='swin_base_patch4_window7_224',  # you can choose 'swin_t', 'swin_s', etc.
        img_size=256,  # updated img_size from 224 to 256
        n_box=6,
        dropout=0.1
    ):
        super().__init__()

        # Create two Swin Transformer backbones (from timm) in feature-only mode.
        self.context_encoder = timm.create_model(
            model_name,
            pretrained=False,
            #remove the classifier layer
            num_classes=0,
            features_only=True,
            out_indices=[-1]
        )
        self.target_encoder = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            features_only=True,
            out_indices=[-1]
        )

        # Get the output channel dimension from the backbone (used in predictor)
        self.out_dim = self.context_encoder.feature_info[-1]['num_chs']

        # Predictor head applied to context branch’s feature map.
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
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)

    def get_random_boxes(self, batch_size, H, n_box=None):
        """Generate random box coordinates for each item in the batch.
           Boxes are defined on the feature map grid (of size H x H)."""
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
        
        B, C, H, W = feature.shape
        target_features = []
        for b in range(B):
            batch_targets = []
            for box in boxes[b]:
                x1, y1, x2, y2 = box
             
                target = feature[b:b+1, :, y1:y2, x1:x2]  # (1, C, box_h, box_w)
                target = F.adaptive_avg_pool2d(target, (1, 1))  # (1, C, 1, 1)
                batch_targets.append(target)
            # Concatenate targets for all boxes for this image
            batch_targets = torch.cat(batch_targets, dim=0)  # (n_box, C, 1, 1)
            batch_targets = batch_targets.view(-1, C)  # (n_box, C)
            target_features.append(batch_targets)
        target_features = torch.stack(target_features)  # (B, n_box, C)
        return target_features

    @torch.no_grad()
    def momentum_update(self, momentum=0.999):
        """
        Update the target encoder with the context encoder parameters using momentum.
        """
        for target_param, context_param in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            target_param.data.mul_(momentum).add_((1 - momentum) * context_param.data)

    # forward method for deepseek
    def forward(self, images, boxes=None):
        B = images.shape[0]
        context_feats = self.context_encoder(images)[-1]
        context_feats = context_feats.permute(0, 3, 1, 2)  # (B, C, H, W)
        H = context_feats.shape[2]  # assuming square feature map

        if boxes is None:
            boxes = self.get_random_boxes(B, H)

        # Apply predictor head to context features
        pred_feats = self.predictor(context_feats)  # (B, C, H, W)
        pred_feats = self.extract_target(pred_feats, boxes)  # (B, n_box, C)

        # Compute target features without gradient
        with torch.no_grad():
            target_feats = self.target_encoder(images)[-1]  # (B, H, W, C)
            target_feats = target_feats.permute(0, 3, 1, 2)  # (B, C, H, W)
            target_feats = self.extract_target(target_feats, boxes)  # (B, n_box, C)

        return pred_feats, target_feats

def create_DRijepa(
    img_size=256,  # updated to 256
    model_name='swin_base_patch4_window7_224',
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
        val_loader=None,  # optional validation loader
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
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=1.5e-4,
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

        # Save checkpoint to file (uncomment and update the path as needed)
        # torch.save(checkpoint, os.path.join(self.save_dir, f"checkpoint_ep_{epoch}.pt"))
        print(f"Checkpoint saved for epoch {epoch} with loss {loss:.4f}")

    def log_attention_map(self, epoch):
        
        self.model.eval()  # Set model to evaluation mode
        # Grab one batch from the training loader
        sample = next(iter(self.train_loader))
        sample = sample.to(self.device)

        with torch.no_grad():
            features = self.model.context_encoder(sample)[-1]  # shape: (B, C, H, W)
            attn_map = features.mean(dim=1, keepdim=True)  # shape: (B, 1, H, W)
            # Normalize to [0, 1]
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            # Upsample to match the original image size
            attn_map = F.interpolate(attn_map, size=(self.model.img_size, self.model.img_size), mode='bilinear', align_corners=False)
            # Take the first sample for visualization
            attn_img = attn_map[0].squeeze().cpu().numpy()

        # Log the attention map as an image with wandb
        wandb.log({f"Attention_Map_Epoch_{epoch}": wandb.Image(attn_img, caption=f"Epoch {epoch}")})
        self.model.train()  # Set back to training mode

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

            self.model.momentum_update()  # update target encoder

            total_loss += loss.item()

            if batch_idx % self.log_interval == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'epoch': epoch,
                    'batch': batch_idx
                })

            pbar.update()

        pbar.close()
        avg_loss = total_loss / n_batch
        return avg_loss

    def validate_epoch(self, epoch):
        """Run one epoch over the validation set."""
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
            for ep in tqdm(range(self.max_ep)):
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

                # Run validation if a validation loader is provided
                if self.val_loader is not None:
                    val_loss = self.validate_epoch(ep)
                #     if val_loss < best_val_loss:
                #         best_val_loss = val_loss
                #         # Optionally, save checkpoint for best validation loss
                #         self.save_checkpoint(ep, loss)

                # if ep % self.save_interval == 0 or loss < best_loss:
                #     self.save_checkpoint(ep, loss)
                #     if loss < best_loss:
                #         best_loss = loss

                # Log an attention map every 15 epochs
                if ep % 15 == 0:
                    self.log_attention_map(ep)

            self.save_checkpoint(self.max_ep , best_loss)
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Saving checkpoint and exiting...")
            self.save_checkpoint(ep, loss)

if __name__ == "__main__":
    print("Hii I am here")
    BATCH_SIZE = 32
    from data_pipeline.data_set import SSLTrainLoader , SSLValidLoader
    from data_pipeline.data_aug import IJEPAAugmentation
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    transforms_ = IJEPAAugmentation(img_size=256)
    train_loader = SSLTrainLoader(
        dataset_names=dataset_names,
        transformation=transforms_,
        batch_size=48,
        num_work=4,
    ).get_loader()

    valid_loader = SSLValidLoader(
        dataset_names=dataset_names,
        transformation=transforms_,
        batch_size=8,
        num_work=4,
    ).get_loader()

    model, loss_fn = create_DRijepa(
        img_size=256,  # updated img_size from swin_config["img_size"] to 256
        model_name='swin_base_patch4_window7_224',
        n_box=6,
        dropout=0.1
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=valid_loader,  # pass the validation loader here
    )

    trainer.train()
