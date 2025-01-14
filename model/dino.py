import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset , DataLoader
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

from data_pipeline.data_set import UnitedTrainingDataset

class Augmentation:

    def __init__(self ,
        img_size: int = 1024,
        global_crop: int = 900 ,
        local_crop: int = 400 ,
        n_crops_local : int = 7
        ):
        self.n_crops_local = n_crops_local
        self.base_transform = A.Compose([
            A.Resize(img_size , img_size),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
        ])
        self.global_transform = A.Compose(
            [
                A.RandomResizedCrop(size=(global_crop , global_crop)),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2,
                            contrast_limit=0.2,
                            p=1
                            ),
                        A.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.2,
                            hue=0.1,
                            p=1
                        ),
                    ] , p=0.8),
                
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3,7) , p=1),
                        A.GaussNoise(p=1)
                    ] , p=0.5
                ),
                ToTensorV2(),
            ]
        )


        self.local_transform = A.Compose(
            [
                A.RandomResizedCrop(size=(local_crop , local_crop)),
                A.HorizontalFlip(p=0.5),

                A.RandomBrightnessContrast(brightness_limit=0.1 , contrast_limit=0.1, p=0.7),
                ToTensorV2(),
            ]
        )


    def __call__(self , image):
        views = []

        if isinstance(image , Image.Image):
            image = np.array(image)


        # base transforms -> apply clahe
        prep_basic = self.base_transform(image=image)['image']

        # global transformation
        for _ in range(2):
            aug = self.global_transform(prep_basic)
            views.append(aug['image'])

        # local transformation
        for _ in range(self.n_crops_local):
            aug = self.local_transform(prep_basic)
            views.append(aug['image'])


        return views


class DINODataset(Dataset):

    def __init__(self , img_path: str) -> None:
        super().__init__()
        self.img_path = img_path
        self.transform = Augmentation()

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, index):
        img_path = self.img_path[index]
        img = Image.open(img_path).convert('RGB')
        view = self.transform(image=img)

        return view
    

class Student(nn.Module):
    def __init__(self , encoder: nn.Module , head: nn.Sequential):
        super().__init__()

        self.encoder = encoder
        self.head = head


    def forward(self ,x):
        out_encoder = self.encoder(x)
        out_head = self.head(out_encoder)
        return out_head


class Teacher(nn.Module):
    def __init__(self , encoder: nn.Module , head: nn.Sequential , centering):
        super().__init__()
        self.encoder = encoder
        self.head = head

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.head.parameters():
            param.requires_grad = False
    
    def forward(self , x):
        out_enc = self.encoder(x)
        out_head = self.head(out_enc)
        return out_head


class DINOHead(nn.Module):

    def __init__(self , in_dim , out_dim , hidden_dim=2048 , bottleneck_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim , hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim , hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim , bottleneck_dim),
        )

        self.last_layer = nn.Linear(bottleneck_dim , out_dim)
        self.apply(self._init_weights)
    def _init_weights(self , m):
        if isinstance(m , nn.Linear):
            nn.init.trunc_normal_(m.weight , std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias , 0)

    def forward(self ,x):

        x = self.mlp(x)
        x = F.normalize(x , dim=-1)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):

    def __init__(self , out_dim , teach_temp=0.04 , student_temp=0.1 , center_mom=0.9):

        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teach_temp
        self.center_mom = center_mom

        self.register_buffer("center" , torch.zeros(1, out_dim).cuda())

    def forward(self , student_out , teacher_out):

        student_out = student_out / self.student_temp
        teacher_out = teacher_out / self.teacher_temp

        student_soft = F.softmax(student_out , dim=-1)
        teacher_soft = F.softmax((teacher_out - self.center) , dim=-1)

        loss = torch.sum(-teacher_soft * torch.log_softmax(student_out , dim=-1) , dim=-1)
        self.update_center(teacher_out)

        return loss.mean()
    @torch.no_grad()
    def update_center(self , teacher_out):

        batch_center = torch.sum(teacher_out , dim=0 , keepdim=True)
        batch_center = batch_center / len(teacher_out)

        self.center = self.center * self.center_mom + batch_center * (1 - self.center_mom)



class DINO:

    def __init__(self, encoder: nn.Module , head: nn.Module , optimizer: torch.optim, loss_fn , lr_schedule , wd_schedule , momentum_schedule):
        self.encoder = encoder 
        self.head = head
        self.optim = optimizer
        self.loss_fn = loss_fn
        self.Teacher = Teacher(encoder=self.encoder , head=self.head).cuda()
        self.Student = Student(encoder=self.encoder , head=self.head).cuda()
        

    def update_teacher(self , m=0.996):
        for param_s , param_t in zip(self.Student.parameters() , self.Teacher.parameters()):
            param_t.data.mul_(m).add_((1-m) * param_s.detach().data)

    
    def train(self ,img_path, num_epoch , batch_size):
        """
        training loop
        """
        dataset = DINODataset(img_path=img_path)
        dataloader = DataLoader(
            dataset , 
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True
        )

        for epoch in tqdm(range(num_epoch)):
            total_loss = 0
            for views in tqdm(dataloader):

                views = [view.cuda() for view in views]

                global_views = views[:2]
                local_views = views[2:]

                all_views = global_views + local_views

                with torch.no_grad():
                    t_out = [self.Teacher(view) for view in global_views]

                s_out = [self.Student(view) for view in all_views]

                n_loss_term = 0
                loss = 0

                for i , teacher_out in enumerate(t_out):
                    for j , student_out in enumerate(s_out):
                        if j == i:
                            continue

                        loss += self.loss_fn(student_out , teacher_out.detach())
                        n_loss_term += 1
                # average of all loss
                loss /= n_loss_term
                total_loss += loss.item()

                #student
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                #teacher
                self.update_teacher()

            avg_loss = total_loss / len(dataloader)

            print(f"Epoch {epoch + 1}/{num_epoch} , avg loss = {avg_loss:.3f}")
                   
        















        
        











        



        

        

        





    





       


    

