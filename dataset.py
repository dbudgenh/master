from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
import pandas as pd
import os
from enum import Enum
from numpy import asarray
import torchvision
torchvision.disable_beta_transforms_warning()
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from abc import ABC,abstractmethod
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import KFold
from collections import defaultdict
from random import sample
from collections import Counter
import random

NUM_CLASSES = 524

class Split(Enum):
    TRAIN=1
    TEST=2
    VALID=3

class Subset(Dataset):
    """Subset of a dataset at specified indices

    Args:
        dataset (Dataset): The whole dataset
        indices (Sequence): Indices in the whole set selected for subset
        transform: transformations
    """

    def __init__(self,dataset,indices,transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        imgs, labels = self.dataset.__getitem__(self.indices[idx])
        if self.transform:
            return self.transform(imgs),labels
        return imgs,labels
    
    def __len__(self):
        return len(self.indices)

class BirdDataset(Dataset):
    """ Dataset containing images of birds
        The data is stored in the dataframe
        On demand, the required data i.e. path to the image, the class-id etc. are returned
    """

    DEFAULT_RESIZE_SIZE = (224,224)
    MEAN = (0.4742, 0.4694, 0.3954)
    STD = (0.2394, 0.2332, 0.2547)

    def __init__(self,root_dir,csv_file,transform=None,split=None) -> None:
        bird_frame = pd.read_csv(csv_file,delimiter=',')
        if split:
            if split == Split.TRAIN:
                bird_frame = bird_frame[bird_frame['data set'] == 'train']
            if split == Split.TEST:
                bird_frame = bird_frame[bird_frame['data set'] == 'test']
            if split == Split.VALID:
                bird_frame = bird_frame[bird_frame['data set'] == 'valid']
        self.class_id = torch.tensor(bird_frame['class id'].tolist())
        self.base_path =np.array(bird_frame['filepaths'].tolist()).astype(np.string_)
        self.root_dir = root_dir
        self.transform = transform
        del bird_frame

    def __len__(self) -> int:
        assert len(self.class_id) == len(self.base_path)
        return len(self.class_id)

    
    def __getitem__(self, idx):
        class_id =  int(self.class_id[idx])
        base_path = str(self.base_path[idx],encoding='utf-8')
        img_path = os.path.join(self.root_dir,base_path)
        #label = self.bird_frame.iloc[idx,2]
        #dataset = self.bird_frame.iloc[idx,3]
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
        return DataLoader(dataset=self.training_data, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          pin_memory=True,
                          #generator=torch.Generator(device='cuda'),
                          persistent_workers=False if self.num_workers == 0 else True
                          )
    
    def val_dataloader(self):
        return DataLoader(dataset=self.validation_data, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          pin_memory=True,
                          #generator=torch.Generator(device='cuda'),
                          persistent_workers=False if self.num_workers == 0 else True
                          )
    
    def test_dataloader(self):
        return DataLoader(dataset=self.testing_data, 
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          #generator=torch.Generator(device='cuda'),
                          persistent_workers=False if self.num_workers == 0 else True
                          )
    
    def predict_dataloader(self):
        return DataLoader(dataset=self.predicting_data, 
                          batch_size=self.batch_size,
                          shuffle=False, 
                          num_workers=self.num_workers,
                          pin_memory=True,
                          #generator=torch.Generator(device='cuda'),
                          persistent_workers=False if self.num_workers == 0 else True
                          )

class UndersampleMinimumDataset(Dataset):
    """
    A PyTorch Dataset class that performs undersampling on an input dataset.

    This class is designed to handle class imbalance in datasets.
    It takes an existing dataset,  and creates a new dataset 
    where each class has an equal number of samples.

    The number of samples per class in the new dataset = number of samples in the 
    least represented class in the original dataset.
    """
    
    def __init__(self,
                 dataset:Dataset,
                 transform = None):
        super(UndersampleMinimumDataset,self).__init__()
        self.dataset = dataset #original dataset
        self.transform = transform #transformations to apply to undersampled dataset
        self.indices = self._undersample_indices(dataset)
        self.undersampled_dataset = Subset(
            dataset=self.dataset,
            indices=self.indices,
            transform=self.transform
        )

    def _undersample_indices(self, dataset: Dataset) -> list[int]:
        # Count samples and store indices for each class
        class_indices = defaultdict(list)
        if dataset.targets:
            #for each class label, the amount of images
            sum_per_target = Counter(dataset.targets)
            min_count = min(sum_per_target.values())
            sorted_sum_per_target = dict(sorted(sum_per_target.items(), key=lambda item: item[0]))
            
            indices = []
            for class_id, amount_of_images in sorted_sum_per_target.items():
                indices_for_class = [i for i, target in enumerate(dataset.targets) if target == class_id]
                rand_indices = random.sample(indices_for_class, min_count)
                indices.extend(rand_indices)
            return indices
        else:
            for idx in range(len(dataset)):
                _, label = dataset[idx]
                class_indices[label].append(idx)

            # Find the minimum count across classes
            min_count = min(len(indices) for indices in class_indices.values())

            # Select indices to equalize class distribution
            undersampled_indices = []
            for indices in class_indices.values():
                undersampled_indices.extend(sample(indices, min_count))

            return undersampled_indices

    def __len__(self) -> int:
        return len(self.undersampled_dataset)
    
    def __getitem__(self, idx):
       return self.undersampled_dataset.__getitem__(idx) 

class UndersampleSplitDatamodule(BaseDataModule):
    def __init__(self,
                 train_transform,
                 valid_transform,
                 total_dataset:Dataset,
                 data_split_ratios=[0.8,0.15,0.05],
                 random_seed=42,
                 batch_size=32,
                 num_workers=4,
                 collate_fn=None):
        super().__init__(train_transform=train_transform,
                         valid_transform=valid_transform,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        assert sum(data_split_ratios) == 1.0, "Data split ratios must sum to 1.0"
        self.train_data_percentage = data_split_ratios[0]
        self.validation_data_percentage = data_split_ratios[1]
        self.test_data_percentage = data_split_ratios[2]

        self.random_seed = random_seed #For recreation purposes
        total_dataset = UndersampleMinimumDataset(dataset=total_dataset)
        self.data_train, self.data_validation,self.data_test = self._split_dataset(total_dataset,
                                                                    data_split_ratios)
    def _split_dataset(self,
                       total_dataset:Dataset,
                       data_split_ratios:float) -> tuple[Subset,Subset]:
        
        dataset_size = len(total_dataset)
        items_per_class = dataset_size // NUM_CLASSES

        train_size_per_class = int(data_split_ratios[0] * items_per_class)
        valid_size_per_class = int(data_split_ratios[1] * items_per_class)

        # Preallocate lists for indices
        train_indices = np.zeros(NUM_CLASSES * train_size_per_class, dtype=int)
        valid_indices = np.zeros(NUM_CLASSES * valid_size_per_class, dtype=int)
        test_indices = np.zeros(dataset_size - len(train_indices) - len(valid_indices), dtype=int)
        
        # Index offset for filling in the indices arrays
        train_offset = valid_offset = test_offset = 0
        
        for class_id in range(NUM_CLASSES):
            start_index = class_id * items_per_class
            end_index = start_index + items_per_class
            
            # Shuffle class indices before splitting
            np.random.seed(self.random_seed)
            class_indices = np.random.permutation(np.arange(start_index, end_index))
            
            # Split indices for this class into train, valid, test
            train_end = train_offset + train_size_per_class
            valid_end = valid_offset + valid_size_per_class
            
            train_indices[train_offset:train_end] = class_indices[:train_size_per_class]
            valid_indices[valid_offset:valid_end] = class_indices[train_size_per_class:train_size_per_class + valid_size_per_class]
            test_indices[test_offset:test_offset + (items_per_class - train_size_per_class - valid_size_per_class)] = class_indices[train_size_per_class + valid_size_per_class:]
            
            train_offset += train_size_per_class
            valid_offset += valid_size_per_class
            test_offset += items_per_class - train_size_per_class - valid_size_per_class
        
        # Convert numpy arrays to lists (necessary for Subset class)
        train_indices = train_indices.tolist()
        valid_indices = valid_indices.tolist()
        test_indices = test_indices.tolist()
        
        # Create subsets
        train_dataset = Subset(dataset=total_dataset, 
                               indices=train_indices, 
                               transform=self.train_transform)
        valid_dataset = Subset(dataset=total_dataset,
                                indices=valid_indices,
                                transform=self.valid_transform)
        test_dataset = Subset(dataset=total_dataset,
                              indices=test_indices,
                              transform=self.valid_transform)
        return train_dataset, valid_dataset, test_dataset

        
        
      

    def train_data(self):
        return self.data_train
    def valid_data(self):
        return self.data_validation
    def test_data(self):
        return self.data_test
    def predict_data(self):
        return ImageFolder(root="test" ,transform=self.valid_transform)
    
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
    def train_data(self):
        return ImageFolder(root=self.root_dir + "/train" ,transform=self.train_transform,)#target_transform=self._convert_label)
    def valid_data(self):
        return ImageFolder(root=self.root_dir + "/valid" ,transform=self.valid_transform,)#target_transform=self._convert_label)
    def test_data(self):
        return ImageFolder(root=self.root_dir + "/test" ,transform=self.valid_transform,)#target_transform=self._convert_label)
    def predict_data(self):
        return ImageFolder(root=self.root_dir + "/test" ,transform=self.valid_transform,)#target_transform=self._convert_label)

class KFoldDataModule(BaseDataModule):
    def __init__(self,
                 k,
                 total_dataset,
                 train_transform,
                 valid_transform,
                 random_seed=123456789,
                 batch_size=32,
                 num_workers=2,
                 collate_fn=None):
        super().__init__(train_transform=train_transform,
                         valid_transform=valid_transform,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.k = k #Amount of folds
        self.random_seed = random_seed #For recreation purposes
        self.total_dataset = total_dataset
        
        kfold = KFold(n_splits=k,
                      shuffle=True,
                      random_state=random_seed)
        self.all_splits = kfold.split(self.total_dataset)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            train_indices,validation_indices = next(self.all_splits)
            self.data_train = Subset(
                dataset=self.total_dataset,
                indices=train_indices,
                transform=self.train_transform
            )
            self.data_validation = Subset(
                dataset = self.total_dataset,
                indices = validation_indices,
                transform = self.valid_transform)
            return self
        except StopIteration:
            raise StopIteration("All K-Fold splits have been processed")

    def train_data(self):
        return self.data_train
    def valid_data(self):
        return self.data_validation
    def test_data(self):
        return self.data_validation
    def predict_data(self):
       return self.data_validation
    

