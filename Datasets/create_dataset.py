'''
Split dataset as train, test, val  6:2:2
use function dataset_wrap, return {train:, val:, test:} torch dataset

datasets names: isic2018, PH2, DMF, SKD
'''

import os
import json
import torch
import random
import numpy as np
from torchvision import transforms
import albumentations as A
import pandas as pd
from Datasets.transform import *
from Datasets.unimatch_utils import obtain_cutmix_box

dataset_indices = {
    'isic2018': 0,
    'PH2': 1,
    'DMF': 2,
    'SKD': 3,
}

def norm01(x):
    return np.clip(x, 0, 255) / 255


class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        
        super(SkinDataset, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug

        self.num_samples = len(self.dataset)

        p = 0.5
        self.aug_transf = A.Compose([
            A.Resize(img_size, img_size),
            A.GaussNoise(p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.ShiftScaleRotate(p=p),
            A.RandomBrightnessContrast(p=p),
        ])
        self.transf = A.Compose([
            A.Resize(img_size, img_size),
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

        img_data = np.load(img_path)
        label_data = np.load(label_path) > 0.5

        if self.use_aug:
            tsf = self.aug_transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        else:
            tsf = self.transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        img_data, label_data = tsf['image'], tsf['mask']
        
        img_data = norm01(img_data)
        label_data = np.expand_dims(label_data, 0)

        img_data = torch.from_numpy(img_data).float()
        label_data = torch.from_numpy(label_data).float()

        img_data = img_data.permute(2, 0, 1)
        org_image =  torch.from_numpy(tsf['image']).float().permute(2, 0, 1)
        img_data = self.normalize(img_data)


        return{
            'org_image': org_image,
            'image': img_data,
            'label': label_data,
            'name': sample_name.replace('.npy', ''),
        }


    def __len__(self):
        return self.num_samples


class StrongWeakAugment(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        
        super(StrongWeakAugment, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug

        self.num_samples = len(self.dataset)

        w_p = 0.5
        s_p = 1.0
        self.weak_augment = A.Compose([
            A.Resize(img_size, img_size),
            A.GaussNoise(p=w_p),
            A.HorizontalFlip(p=w_p),
            A.VerticalFlip(p=w_p),
            A.ShiftScaleRotate(p=w_p),
            A.RandomBrightnessContrast(p=w_p),
        ])
        self.strong_augment = A.Compose([
            A.GaussNoise(p=s_p),
            A.ColorJitter(p=s_p),
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

        img_data = np.load(img_path)

        img_w = self.weak_augment(image=img_data.astype('uint8'))['image']
        img_s = self.strong_augment(image=img_w.astype('uint8'))['image']
        
        
        img_w = norm01(img_w)
        img_s = norm01(img_s)

        img_s = torch.from_numpy(img_s).float()
        img_w = torch.from_numpy(img_w).float()
        
        img_s = img_s.permute(2, 0, 1)
        img_w = img_w.permute(2, 0, 1)
        
        img_s = self.normalize(img_s)
        img_w = self.normalize(img_w)

        return{
            'img_w': img_w,
            'img_s': img_s,
        }


    def __len__(self):
        return self.num_samples
    
class StrongWeakAugment2(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super(StrongWeakAugment2, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug

        self.num_samples = len(self.dataset)

        w_p = 0.5
        s_p = 1.0
        self.weak_augment = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=w_p),
            A.VerticalFlip(p=w_p)
        ])
        self.strong_augment = A.Compose([
            A.GaussNoise(p=s_p),
            A.RandomBrightnessContrast(p=s_p),
            A.ColorJitter(p=s_p)
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

        img_data = np.load(img_path)
        org_img = torch.from_numpy(img_data).float().permute(2, 0, 1)

        img_w = self.weak_augment(image=img_data.astype('uint8'))['image']
        img_s = self.strong_augment(image=img_w.astype('uint8'))['image']
        
        img_w = norm01(img_w)
        img_s = norm01(img_s)
       
        img_w = torch.from_numpy(img_w).float()
        img_s = torch.from_numpy(img_s).float()

        img_w = img_w.permute(2, 0, 1)
        img_s = img_s.permute(2, 0, 1)
        
        img_w = self.normalize(img_w)
        img_s = self.normalize(img_s)

        return{
            'id': index,
            'img_w': img_w,
            'img_s': img_s,
            'org_image': org_img,
        }


    def __len__(self):
        return self.num_samples

class SkinDataset2(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super(SkinDataset2, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug

        self.num_samples = len(self.dataset)

        p = 0.5
        self.aug_transf = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
        ])
        self.transf = A.Compose([
            A.Resize(img_size, img_size),
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

        img_data = np.load(img_path)
        org_img = torch.from_numpy(img_data).float().permute(2, 0, 1)
        label_data = np.load(label_path) > 0.5

        if self.use_aug:
            tsf = self.aug_transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        else:
            tsf = self.transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        img_data, label_data = tsf['image'], tsf['mask']
        
        img_data = norm01(img_data)
        
        label_data = np.expand_dims(label_data, 0)

        img_data = torch.from_numpy(img_data).float()
        label_data = torch.from_numpy(label_data).float()

        img_data = img_data.permute(2, 0, 1)
        img_data = self.normalize(img_data)


        return{
            'image': img_data,
            'label': label_data,
            'org_image': org_img,
        }


    def __len__(self):
        return self.num_samples
    
class StrongWeakAugment3(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super(StrongWeakAugment3, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug

        self.num_samples = len(self.dataset)

        w_p = 0.5
        s_p = 1.0
        self.weak_augment = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=w_p),
            A.VerticalFlip(p=w_p)
        ])
        self.strong_augment = A.Compose([
            A.GaussNoise(p=s_p),
            A.RandomBrightnessContrast(p=s_p),
            A.ColorJitter(p=s_p)
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

        img_data = np.load(img_path)

        img_w = self.weak_augment(image=img_data.astype('uint8'))['image']
        img_s = self.strong_augment(image=img_w.astype('uint8'))['image']
        
        img_w = norm01(img_w)
        img_s = norm01(img_s)
       
        img_w = torch.from_numpy(img_w).float()
        img_s = torch.from_numpy(img_s).float()

        img_w = img_w.permute(2, 0, 1)
        img_s = img_s.permute(2, 0, 1)
        
        img_w = self.normalize(img_w)
        img_s = self.normalize(img_s)

        mask = torch.zeros((3, 224, 224))
        cutmix_w = obtain_cutmix_box(img_w.shape[1], p=0.5)
        cutmix_s = obtain_cutmix_box(img_s.shape[1], p=0.5)
        img_w[:, cutmix_w == 1] = mask[:, cutmix_w == 1]
        img_s[:, cutmix_s == 1] = mask[:, cutmix_s == 1]
        
        return{
            'id': index,
            'img_w': img_w,
            'img_s': img_s,
        }


    def __len__(self):
        return self.num_samples
    
class SkinDataset3(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super(SkinDataset3, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug

        self.num_samples = len(self.dataset)

        p = 0.5
        self.aug_transf = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
        ])
        self.transf = A.Compose([
            A.Resize(img_size, img_size),
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

        img_data = np.load(img_path)
        label_data = np.load(label_path) > 0.5
        W, H = 512, 512
        patch1 = img_data[:W//2, :H//2]
        patch2 = img_data[W//2:, :H//2]
        patch3 = img_data[:W//2, H//2:]
        patch4 = img_data[W//2:, H//2:]
        label_patch1 = label_data[:W//2, :H//2]
        label_patch2 = label_data[W//2:, :H//2]
        label_patch3 = label_data[:W//2, H//2:]
        label_patch4 = label_data[W//2:, H//2:]

        if self.use_aug:
            transf = self.aug_transf
        else:
            transf = self.transf
            
        tsf = transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        tsf1 = transf(image=patch1.astype('uint8'), mask=label_patch1.astype('uint8'))
        tsf2 = transf(image=patch2.astype('uint8'), mask=label_patch2.astype('uint8'))
        tsf3 = transf(image=patch3.astype('uint8'), mask=label_patch3.astype('uint8'))
        tsf4 = transf(image=patch4.astype('uint8'), mask=label_patch4.astype('uint8'))
        img_data, label_data = tsf['image'], tsf['mask']
        patch1, label_patch1 = tsf1['image'], tsf1['mask']
        patch2, label_patch2 = tsf2['image'], tsf2['mask']
        patch3, label_patch3 = tsf3['image'], tsf3['mask']
        patch4, label_patch4 = tsf4['image'], tsf4['mask']
        
        img_data = norm01(img_data)
        patch1 = norm01(patch1)
        patch2 = norm01(patch2)
        patch3 = norm01(patch3)
        patch4 = norm01(patch4)
        
        label_data = np.expand_dims(label_data, 0)
        label_patch1 = np.expand_dims(label_patch1, 0)
        label_patch2 = np.expand_dims(label_patch2, 0)
        label_patch3 = np.expand_dims(label_patch3, 0)
        label_patch4 = np.expand_dims(label_patch4, 0)

        img_data = torch.from_numpy(img_data).float()
        label_data = torch.from_numpy(label_data).float()
        patch1 = torch.from_numpy(patch1).float()
        label_patch1 = torch.from_numpy(label_patch1).float()
        patch2 = torch.from_numpy(patch2).float()
        label_patch2 = torch.from_numpy(label_patch2).float()
        patch3 = torch.from_numpy(patch3).float()
        label_patch3 = torch.from_numpy(label_patch3).float()
        patch4 = torch.from_numpy(patch4).float()
        label_patch4 = torch.from_numpy(label_patch4).float()

        img_data = img_data.permute(2, 0, 1)
        patch1 = patch1.permute(2, 0, 1)
        patch2 = patch2.permute(2, 0, 1)
        patch3 = patch3.permute(2, 0, 1)
        patch4 = patch4.permute(2, 0, 1)
        
        img_data = self.normalize(img_data)
        patch1 = self.normalize(patch1)
        patch2 = self.normalize(patch2)
        patch3 = self.normalize(patch3)
        patch4 = self.normalize(patch4)


        return{
            'image': img_data,
            'label': label_data,
            'patch1': patch1,
            'patch2': patch2,
            'patch3': patch3,
            'patch4': patch4,
            'label_patch1': label_patch1,
            'label_patch2': label_patch2,
            'label_patch3': label_patch3,
            'label_patch4': label_patch4,
        }


    def __len__(self):
        return self.num_samples



def get_dataset(args, img_size=384, supervised_ratio=0.2, train_aug=False, k=6, lb_dataset=SkinDataset2, ulb_dataset=StrongWeakAugment2, v_dataset=SkinDataset):
    
    folds = []
    for idx in range(1, 6):
        fold = []
        with open(f'{args.data.train_folder}/fold{idx}.txt', 'r') as f:
            fold = [line.replace('\n', '') for line in f.readlines()]
        folds.append(fold)
        
    
    train_data = []
    for j in range(5):
        if j != k - 1:
            train_data = [*train_data, *folds[j]]
            
    train_data = sorted(train_data)
    l_data = sorted(random.sample(train_data, int(len(train_data) * supervised_ratio)))
    u_data = sorted([sample for sample in train_data if sample not in l_data])
    l_dataset = lb_dataset(dataset=l_data, img_size=img_size, use_aug=train_aug, data_path=args.data.train_folder)
    u_dataset = ulb_dataset(dataset=u_data, img_size=img_size, use_aug=train_aug, data_path=args.data.train_folder)
        
    val_data = sorted(folds[k - 1])
    val_dataset = v_dataset(dataset=val_data, img_size=img_size, use_aug=False, data_path=args.data.val_folder)
    
    print(f'Train Data: {train_data[0]} - {len(train_data)}')
    print(f'Labeled Data: {l_data[0]} - {len(l_data)}')
    print(f'Unlabeled Data: {u_data[0]} - {len(u_data)}')
    print(f'Val Data: {val_data[0]} - {len(val_data)}')
    
    dataset = {
        'lb_dataset': l_dataset,
        'ulb_dataset': u_dataset,
        'val_dataset': val_dataset
    }
             
    return dataset
