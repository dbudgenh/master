from models import EfficientNet_B0,NaiveClassifier,EfficientNet_V2_S,EfficientNet_V2_M,EfficientNet_V2_S_Pretrained,EfficientNet_B0_Pretrained
from dataset import BirdDataNPZModule,BirdDataModule,BirdDataset,BirdDataModuleV2
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import transforms
from torchvision.transforms.v2 import AugMix
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import numpy as np
from torchvision.models import EfficientNet_V2_S_Weights,EfficientNet_B0_Weights

NUM_WORKERS = 10
PRINT_EVERY = 2000
CHECKPOINT_PATH = 'C:/Users/david/Desktop/Python/master/statistics/EfficientNet_V2_S_Pretrained_Adam/TransferLearning/epoch=38_validation_loss=0.6554_validation_accuracy=0.83_validation_mcc=0.83.ckpt'
BATCH_SIZE = 64 #128 is optimal for TPUs, use multiples of 64 that fits into memory
LEARNING_RATE = 1e-3
WEIGHT_DECAY=1e-2
EPOCHS = 100
MOMENUTUM = 0.9


def main():
    torch.set_float32_matmul_precision('medium')
    mean = BirdDataset.MEAN
    std =  BirdDataset.STD
    train_transform = transforms.Compose([
            #transforms.ToPILImage(),
            AugMix(severity=4,mixture_width=4,alpha=0.65),
            transforms.CenterCrop(BirdDataset.DEFAULT_RESIZE_SIZE),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.Resize(BirdDataset.DEFAULT_RESIZE_SIZE),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.Normalize(mean=mean,std=std),
            transforms.RandomErasing()
    ])
    valid_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize(BirdDataset.DEFAULT_RESIZE_SIZE),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.Normalize(mean=mean,std=std)
    ])

    datamodule = BirdDataModule(root_dir='C:/Users/david/Desktop/Python/master/data',
                                csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',
                                train_transform=train_transform,
                                valid_transform=valid_transform,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS)
    model = EfficientNet_V2_S(lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(
                                       dirpath=f'C:/Users/david/Desktop/Python/master/statistics/{model.name}_Adam',
                                       filename="{epoch}_{validation_loss:.4f}_{validation_accuracy:.2f}_{validation_mcc:.2f}", 
                                       save_top_k=1,
                                       monitor="validation_loss",
                                       mode='min')
    trainer = pl.Trainer(max_epochs=EPOCHS,callbacks=[model_checkpoint,lr_monitor],precision='bf16-mixed') #
    trainer.fit(model=model,datamodule=datamodule)

    

    
if __name__ == '__main__':
    main()