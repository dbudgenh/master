from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io
from enum import Enum
from PIL import Image
from numpy import asarray
from torchvision.io import read_image
import numpy as np
import pytorch_lightning as pl
from torchvision.transforms import transforms
from torchvision.transforms.v2 import AugMix
from torch.utils.data import DataLoader

class Split(Enum):
    TRAIN=1
    TEST=2
    VALID=3

class BirdDataset(Dataset):
    """ Dataset containing images of birds
        The data is stored in the dataframe
        On demand, the required data i.e. path to the image, the class-id etc. are returned
    """

    def __init__(self,root_dir,csv_file,transform=None,split=None) -> None:
        self.bird_frame = pd.read_csv(csv_file,delimiter=',')
        if split:
            if split == Split.TRAIN:
                self.bird_frame = self.bird_frame[self.bird_frame['data set'] == 'train']
            if split == Split.TEST:
                self.bird_frame = self.bird_frame[self.bird_frame['data set'] == 'test']
            if split == Split.VALID:
                self.bird_frame = self.bird_frame[self.bird_frame['data set'] == 'valid']
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.bird_frame)

    
    def __getitem__(self, idx):
        if idx % 5000 == 0:
            print(f'Loading {idx}-th example')
        class_id =  self.bird_frame.iloc[idx,0]
        base_path = self.bird_frame.iloc[idx,1]
        img_path = os.path.join(self.root_dir,base_path)
        label = self.bird_frame.iloc[idx,2]
        dataset = self.bird_frame.iloc[idx,3]
        scientific_name = self.bird_frame.iloc[idx,4]
        image = read_image(img_path) #output has shape (3,224,224)

        sample = {'image': image, 'class_id': int(class_id), 'label': label,
                  'dataset': dataset, 'scientific_name': scientific_name,'path':img_path, 'base_path':base_path}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample
    
class BirdDatasetNPZ(Dataset):
    """Dataset containing images of birds
        In contrast to the above dataset, the complete dataset is stored in memory!
        This has the additional benefit that no network operations take place when loading the image
        In exchange: the complete dataset (13GB) has to be stored in memory.
    """
    def __init__(self,data_dict,transform=None,split=None) -> None:
        self.images,self.labels,self.dataset = data_dict['images'],data_dict['labels'],data_dict['dataset']
        if split:
            if split == Split.TRAIN:
                filter = self.dataset == 0
                self.images = self.images[filter]
                self.labels = self.labels[filter]
                self.dataset = self.dataset[filter]
            if split == Split.TEST:
                filter = self.dataset == 2
                self.images = self.images[filter]
                self.labels = self.labels[filter]
                self.dataset = self.dataset[filter]
            if split == Split.VALID:
                filter = self.dataset == 1
                self.images = self.images[filter]
                self.labels = self.labels[filter]
                self.dataset = self.dataset[filter]
        self.transform = transform

    def __len__(self) -> int:
        assert len(self.images) == len(self.labels) == len(self.dataset)
        return len(self.images)
    
    def __getitem__(self, idx):
        if idx % 5000 == 0:
            print(f'Loading {idx}-th example')
        #swap channels to be last, np.arrays should be HxWxC format
        image = np.moveaxis(self.images[idx],0,-1)
        label = self.labels[idx]
        dataset = self.dataset[idx]
        sample = {'image': image, 'class_id': int(label),'dataset': dataset}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample
    

class BirdDataModule(pl.LightningDataModule):
    RESIZE_SIZE = (224,224)

    def __init__(self,npz_file_path,batch_size=32,num_workers=2):
        super().__init__()
        self.npz_file_path = npz_file_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        print('Loading .npz file, this will take a while...')
        self.data_dict = np.load(npz_file_path)
        print('Done loading .npz file')

        self.train_transform  = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(BirdDataModule.RESIZE_SIZE),
            AugMix(severity=4,mixture_width=4,alpha=0.65),
            transforms.CenterCrop(BirdDataModule.RESIZE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.Normalize(mean=(0.4742, 0.4694, 0.3954),std=(0.2394, 0.2332, 0.2547))
        ])
        self.valid_transform = transform_valid = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(BirdDataModule.RESIZE_SIZE),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.Normalize(mean=(0.4742, 0.4694, 0.3954),std=(0.2394, 0.2332, 0.2547))
        ])
    
    def setup(self,stage:str):
        if stage == 'fit':
            self.train_data = BirdDatasetNPZ(data_dict=self.data_dict,transform=self.train_transform,split=Split.TRAIN)
            self.valid_data = BirdDatasetNPZ(data_dict=self.data_dict,transform=self.valid_transform,split=Split.VALID)
        if stage == 'test':
            self.test_data = BirdDatasetNPZ(data_dict=self.data_dict,transform=self.valid_transform,split=Split.TEST)
        if stage == 'predict':
            self.predict_data = BirdDatasetNPZ(data_dict=self.data_dict,transform=self.valid_transform,split=Split.TEST)
    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(dataset=self.predict_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class BirdDataNPZModule(pl.LightningDataModule):
    RESIZE_SIZE = (224,224)

    def __init__(self,root_dir,csv_file,batch_size=32,num_workers=2):
        super().__init__()
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bird_frame = pd.read_csv(self.csv_file,delimiter=',')
        self.train_transform  = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(BirdDataModule.RESIZE_SIZE),
            AugMix(severity=4,mixture_width=4,alpha=0.65),
            transforms.CenterCrop(BirdDataModule.RESIZE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.Normalize(mean=(0.4742, 0.4694, 0.3954),std=(0.2394, 0.2332, 0.2547))
        ])
        self.valid_transform = transform_valid = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(BirdDataModule.RESIZE_SIZE),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.Normalize(mean=(0.4742, 0.4694, 0.3954),std=(0.2394, 0.2332, 0.2547))
        ])
    
    def setup(self,stage:str):
        if stage == 'fit':
            self.train_data = BirdDataset(root_dir=self.root_dir,csv_file=self.csv_file,transform=self.train_transform,split=Split.TRAIN)
            self.valid_data = BirdDataset(root_dir=self.root_dir,csv_file=self.csv_file,transform=self.valid_transform,split=Split.VALID)
        if stage == 'test':
            self.test_data = BirdDataset(root_dir=self.root_dir,csv_file=self.csv_file,transform=self.valid_transform,split=Split.TEST)
        if stage == 'predict':
            self.predict_data = BirdDataset(root_dir=self.root_dir,csv_file=self.csv_file,transform=self.valid_transform,split=Split.TEST)
    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(dataset=self.predict_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    

class BaseDataModule(pl.LightningDataModule):
    RESIZE_SIZE = (224,224)

    def __init__(self,batch_size=32,num_workers=2):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_transform  = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(BirdDataModule.RESIZE_SIZE),
            AugMix(severity=4,mixture_width=4,alpha=0.65),
            transforms.CenterCrop(BirdDataModule.RESIZE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.Normalize(mean=(0.4742, 0.4694, 0.3954),std=(0.2394, 0.2332, 0.2547))
        ])
        self.valid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(BirdDataModule.RESIZE_SIZE),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.Normalize(mean=(0.4742, 0.4694, 0.3954),std=(0.2394, 0.2332, 0.2547))
        ])
    
    def train_data(self):
        pass
    def valid_data(self):
        pass
    def test_data(self):
        pass
    def predict_data(self):
        pass
    
    def setup(self,stage:str):
        if stage == 'fit':
            self.train_data = self.train_data()
            self.valid_data = self.valid_data()
        if stage == 'test':
            self.test_data = self.test_data()
        if stage == 'predict':
            self.predict_data = self.predict_data()
    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(dataset=self.predict_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)