from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io
from enum import Enum
from PIL import Image
from numpy import asarray
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.io import read_image
import numpy as np
import pytorch_lightning as pl
from torchvision.transforms import transforms
from torchvision.transforms.v2 import AugMix
from torch.utils.data import DataLoader
from abc import ABC,abstractmethod
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import pickle

class Split(Enum):
    TRAIN=1
    TEST=2
    VALID=3

class BirdDataset(Dataset):
    """ Dataset containing images of birds
        The data is stored in the dataframe
        On demand, the required data i.e. path to the image, the class-id etc. are returned
    """

    DEFAULT_RESIZE_SIZE = (224,224)
    MEAN = (0.4742, 0.4694, 0.3954)
    STD = (0.2394, 0.2332, 0.2547)

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
        class_id =  int(self.bird_frame.iloc[idx,0])
        base_path = self.bird_frame.iloc[idx,1]
        img_path = os.path.join(self.root_dir,base_path)
        #label = self.bird_frame.iloc[idx,2]
        dataset = self.bird_frame.iloc[idx,3]
        #scientific_name = self.bird_frame.iloc[idx,4]
        image = default_loader(img_path) #read_image(img_path) #output has shape (3,224,224)
        if self.transform:
            image = self.transform(image)
        return image, class_id
    
class BirdDatasetNPZ(Dataset):
    """Dataset containing images of birds. The dataset gets stored in memory, by loading the .npz (numpy compressed file) file
        In contrast to the above dataset, the complete dataset is stored in memory!
        This has the additional benefit that no network operations take place when loading the image
        In exchange: the complete dataset (13GB) has to be stored in memory.
        Faster operations in exchange for memory 
    """
    def __init__(self,npz_file_path,transform=None) -> None:
        data_dict = np.load(npz_file_path)
        self.images, self.labels = torch.tensor(data_dict['images']), torch.tensor(data_dict['labels'])
        self.transform = transform
    

    def __len__(self) -> int:
        assert len(self.images) == len(self.labels)
        return len(self.images)
    
    def __getitem__(self, idx):
        #swap channels to be last, np.arrays should be HxWxC format
        image = self.images[idx]
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image,label
    
class BaseDataModule(ABC,pl.LightningDataModule):
    """Base class for all the data modules, this class should never be instantiate directly, but inherited from.
    Data modules contain the train/valid/test/predict datasets, these methods should be implemented by the inherited class.

    """
    def __init__(self,train_transform,valid_transform,batch_size=32,num_workers=2,collate_fn=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.collate_fn = collate_fn
    
    @abstractmethod
    def train_data(self):
        pass
    @abstractmethod
    def valid_data(self):
        pass
    @abstractmethod
    def test_data(self):
        pass
    @abstractmethod
    def predict_data(self):
        pass
    
    def setup(self,stage:str):
        if stage == 'fit':
            print(f"Calling stage {stage} (fit)")
            self.training_data = self.train_data()
            self.validation_data = self.valid_data()
        if stage == 'test':
            print(f"Calling stage {stage} (test)")
            self.testing_data = self.test_data()
        if stage == 'predict':
            print(f"Calling stage {stage} (predict)")
            self.predicting_data = self.predict_data()
    
    def train_dataloader(self):
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,collate_fn=self.collate_fn,pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.validation_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.testing_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(dataset=self.predicting_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,pin_memory=True)

class BirdDataNPZModule(BaseDataModule):
    def __init__(self,train_npz_path,valid_npz_path,test_npz_path,train_transform,valid_transform,batch_size=32,num_workers=2,collate_fn=None):
        super().__init__(train_transform=train_transform,valid_transform=valid_transform,batch_size=batch_size,num_workers=num_workers,collate_fn=collate_fn)
        self.train_npz_path = train_npz_path
        self.valid_npz_path = valid_npz_path
        self.test_npz_path = test_npz_path
        print('Loading train data...')
        self.train = BirdDatasetNPZ(npz_file_path=train_npz_path,transform=self.train_transform)
        print('Loading valid data...')
        self.valid = BirdDatasetNPZ(npz_file_path=valid_npz_path,transform=self.valid_transform)
        print('Loading test data...')
        self.test = BirdDatasetNPZ(npz_file_path=test_npz_path,transform=self.valid_transform)
        print('Loading predict data...')
        self.predict = BirdDatasetNPZ(npz_file_path=test_npz_path,transform=self.valid_transform)
    
    def train_data(self):
        return self.train
    def valid_data(self):
        return self.valid
    def test_data(self):
        return self.test
    def predict_data(self):
        return self.predict



class BirdDataModule(BaseDataModule):
    def __init__(self,root_dir,csv_file,train_transform,valid_transform,batch_size=32,num_workers=2,collate_fn=None):
        super().__init__(train_transform=train_transform,valid_transform=valid_transform,batch_size=batch_size,num_workers=num_workers,collate_fn=collate_fn)
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.bird_frame = pd.read_csv(self.csv_file,delimiter=',')

    def train_data(self):
        return BirdDataset(root_dir=self.root_dir,csv_file=self.csv_file,transform=self.train_transform,split=Split.TRAIN)
    def valid_data(self):
        return BirdDataset(root_dir=self.root_dir,csv_file=self.csv_file,transform=self.valid_transform,split=Split.VALID)
    def test_data(self):
        return BirdDataset(root_dir=self.root_dir,csv_file=self.csv_file,transform=self.valid_transform,split=Split.TEST)
    def predict_data(self):
        return BirdDataset(root_dir=self.root_dir,csv_file=self.csv_file,transform=self.valid_transform,split=Split.TEST)
    
class BirdDataModuleV2(BaseDataModule):
    def __init__(self,root_dir,train_transform,valid_transform,batch_size=32,num_workers=2,collate_fn=None):
        super().__init__(train_transform=train_transform,valid_transform=valid_transform,batch_size=batch_size,num_workers=num_workers,collate_fn=collate_fn)
        self.root_dir = root_dir
        self.mappings = self._load_dict('mappings.pkl')

    def _load_dict(self,file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    
        """The labels for the classes are sorted aphabetically by bird name. This is different from the bird.csv file
           To keep the labels consistent, we map the labels to that of the birds.csv file
        """
    def _convert_label(self,label):
        return self.mappings[label]


    def train_data(self):
        return ImageFolder(root=self.root_dir + "/train" ,transform=self.train_transform,target_transform=self._convert_label)
    def valid_data(self):
        return ImageFolder(root=self.root_dir + "/valid" ,transform=self.valid_transform,target_transform=self._convert_label)
    def test_data(self):
        return ImageFolder(root=self.root_dir + "/test" ,transform=self.valid_transform,target_transform=self._convert_label)
    def predict_data(self):
        return ImageFolder(root=self.root_dir + "/test" ,transform=self.valid_transform,target_transform=self._convert_label)