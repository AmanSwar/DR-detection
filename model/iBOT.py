import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import RandomApply

class RetAug:
    def __init__(self , img_size=512):

        self.transform1 = A.Compose(
            [
                A.Resize(img_size , img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2()
            ]
        )

        self.transform2 = A.Compose(
            [
                A.Resize(img_size , img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3,7)),
                        A.GaussNoise(),
                    ],
                    p=0.5
                ),

                A.ColorJitter(brightness=0.2 , contrast=0.2 , saturation=0.2 , p=0.8),
               A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                           fill_value=0, mask_fill_value=0, p=0.5),
                ToTensorV2()
            ]
        )

        self.lesion_sim = A.Compose([
            A.OneOf([
                A.ElasticTransform(alpha=50, sigma=7, alpha_affine=10, p=0.3),
                A.RandomSizedCrop(min_max_height=(16, 32), height=img_size, 
                                 width=img_size, p=0.3),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.4)
            ], p=0.5)
        ])


    def __call__(self , image):

        view1 = self.transform1(image)['image']
        view2 = self.transform2(image)['image']
        view2 = self.lesion_sim(image=view2.permute(1,2,0).numpy())['image'].permute(2,0,1)

        return view1 , view2



class MaskedViT(nn.Module):

    def __init__(
        self , 
        img_size = 512,
        patch_size = 16,
        in_chan = 3 , 
        embed_dim = 768,
        depth=12 , 
        num_heads=12,
        mlp_ratio = 4,
        mask_ratio=0.4,
        lesion_mask_prob=0.7
    ):

        super().__init__()
        self.path_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.lesion_mask_prob = lesion_mask_prob

        self.patch_embed = nn.Conv2d(in_chan , embed_dim , kernel_size=patch_size , stride=patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1 , self.num_patches + 1 , embed_dim))

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=mlp_ratio * embed_dim
                )
                for _ in range(depth)
            ]
        )


        self.mask_token = nn.Parameter(torch.zeros(1 , 1 , embed_dim))

        nn.init.normal_(self.mask_token , std=0.02)

        self.global_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , embed_dim)
        )

        self.local_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , embed_dim)
        )


    def lesion_aware_masking(self , x , attention_maps=None):
        B,N,D = x.shape
        num_mask = int(N * self.mask_ratio)
    
        if attention_maps is None:
            rand_mask = torch.rand(B , N , device=x.device)

        else:
            rand_mask = 0.3 * torch.rand(B , N , device=x.device) + 0.7 * attention_maps

        mask_idx = rand_mask.topk(num_mask , dim=1).indices

        mask = torch.zeros(B , N , device=x.device).scatter_(1 , mask_idx , 1)

        return mask.bool()
    
    def forward(self , x , mask_ratio=0.4):
        
        B, N, D = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1,2)
        x = x + self.pos_embed[: , 1: , :]

        mask = self.lesion_aware_masking(x)
        mask_token = self.mask_token.expand(B , N ,-1)

        x = x * (1 - mask.unsqueeze(-1)) + mask_token * mask.unsqueeze(-1)

        cls_token = self.cls_token + self.pos_embed[: , :1 , :]
        x = torch.cat((cls_token.expand(B , -1 , -1) , x) , dim=1)


        for blk in self.blocks:
            x = blk(x)


        global_feat = self.global_proj(x[: , 0])
        local_feat = self.local_proj(x[: , 1:])

        return global_feat , local_feat , mask
    
class CustomiBOT(nn.Module):

    def __init__(
            self,  
            student,
            teacher,
            embed_dim=768,
            temp=0.1,
            mask_ratio=0.4,
            momentum=0.996
    ):
        super().__init__()

        self.student = student
        self.teacher = teacher
        self.momentum = momentum
        self.temp = temp
        self.mask_ratio = mask_ratio

        for t_param , s_param in zip(self.teacher.parameters() , self.student.parameters()):
            t_param.data.copy_(s_param.data)
            t_param.requires_grad = False

        self.global_head = nn.Sequential(
            nn.Linear(embed_dim , 512),
            nn.GELU(),
            nn.Linear(512 , embed_dim)
        )

        self.local_head = nn.Sequential(
            nn.Linear(embed_dim , 512),
            nn.GELU(),
            nn.Linear(512 , embed_dim)
        )

    @torch.no_grad()
    def momentum_update(self):

        for t_param , s_param in zip(self.teacher.parameters() , self.student.parameters()):
            t_param.data = t_param.data * self.momentum + s_param.data * (1 - self.momentum)

    def compute_loss(self , student_global , student_local , teacher_global , teacher_local , mask):
        
        global_loss = F.cosine_embedding_loss(
            self.global_head(student_global),
            teacher_global.detach(),
            torch.ones(student_global.size(0).to(student_global.device))
        )

        masked_student = student_local[mask].view(-1 , student_local.size(-1))
        masked_teacher = teacher_local[mask].view(-1 , teacher_local.size(-1))
        local_loss = F.cosine_embedding_loss(
            self.local_head(masked_student),
            masked_teacher.detach(),
            torch.ones(masked_student.size(0).to(student_local.device))
        )

        return global_loss + 0.5 * local_loss
    
    def forward(self , x1 , x2):

        s_global1, s_local1, mask1 = self.student(x1)
        s_global2, s_local2, mask2 = self.student(x2)
        
        with torch.no_grad():
            t_global1, t_local1, _ = self.teacher(x1)
            t_global2, t_local2, _ = self.teacher(x2)
        
        loss = 0.5 * (self.compute_loss(s_global1, s_local1, t_global2, t_local2, mask1) +
                      self.compute_loss(s_global2, s_local2, t_global1, t_local1, mask2))
        
        # Update teacher
        self.momentum_update()
        return loss

    