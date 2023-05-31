from models import EfficientNet_B0,NaiveClassifier,EfficientNet_V2_S,EfficientNet_V2_M
from dataset import BirdDataNPZModule,BirdDataModule
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import transforms
from torchvision.transforms.v2 import AugMix
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

NUM_WORKERS = 16
PRINT_EVERY = 2000
CHECKPOINT_PATH = './statistics/epoch=17_validation_loss=0.2622_validation_accuracy=0.93_validation_mcc=0.91.ckpt'
BATCH_SIZE = 32 #128 is optimal for TPUs, use multiples of 64 that fits into memory
LEARNING_RATE = 1e-4
WEIGHT_DECAY=1e-2
EPOCHS = 300
MOMENUTUM = 0.9


def main():
    torch.set_float32_matmul_precision('medium')
    root_dir = '/content/data/'
    csv_file = '/content/data/birds.csv'
    train_npz_path = 'C:/Users/david/Desktop/dataset_train.npz'
    valid_npz_path = 'C:/Users/david/Desktop/dataset_valid.npz'
    test_npz_path = 'C:/Users/david/Desktop/dataset_test.npz'
    print(f'root_dir={root_dir}\tcsv_file={csv_file}\tnpz_file={train_npz_path}')

    # datamodule = BirdDataNPZModule(train_npz_path=train_npz_path,
    #                                valid_npz_path=valid_npz_path,
    #                                test_npz_path=test_npz_path,
    #                                batch_size=BATCH_SIZE,
    #                                num_workers=NUM_WORKERS)
    datamodule = BirdDataModule(root_dir='C:/Users/david/Desktop/Python/master/data',
                                csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS)

    model_checkpoint = ModelCheckpoint(
                                       dirpath='C:/Users/david/Desktop/Python/master/statistics/',
                                       filename="{epoch}_{validation_loss:.4f}_{validation_accuracy:.2f}_{validation_mcc:.2f}", 
                                       save_top_k=1,
                                       monitor="validation_loss",
                                       mode='min')
    model = EfficientNet_V2_S(lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY,batch_size=BATCH_SIZE)
    trainer = pl.Trainer(max_epochs=EPOCHS,callbacks=[model_checkpoint],precision='bf16-mixed') #,accelerator='tpu',devices=1
    trainer.fit(model=model,datamodule=datamodule,ckpt_path=CHECKPOINT_PATH)

    
if __name__ == '__main__':
    main()