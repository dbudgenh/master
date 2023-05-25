from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io
from enum import Enum
from PIL import Image
from numpy import asarray
from torchvision.io import read_image
import numpy as np

class Split(Enum):
    TRAIN=1
    TEST=2
    VALID=3



class BirdDataset(Dataset):
    """ Dataset containing images of birds
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
    def __init__(self,npz_file,transform=None,split=None) -> None:
        print('Loading npz file')
        data_dict = np.load(npz_file,allow_pickle=True)
        self.images,self.labels,self.dataset = data_dict['images'],data_dict['labels'],data_dict['dataset']
        print('Finished loading npz file')
        if split:
            if split == Split.TRAIN:
                filter = self.dataset = 0
                self.images = self.images[filter]
                self.labels = self.labels[filter]
                self.dataset = self.dataset[filter]
            if split == Split.TEST:
                filter = self.dataset = 2
                self.images = self.images[filter]
                self.labels = self.labels[filter]
                self.dataset = self.dataset[filter]
            if split == Split.VALID:
                filter = self.dataset = 1
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
        image = np.moveaxis(self.images[idx],0,-1)
        label = self.labels[idx]
        dataset = self.dataset[idx]
        sample = {'image': image, 'class_id': label,'dataset': dataset}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample