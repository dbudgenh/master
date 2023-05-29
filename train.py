from models import EfficientNet_B0,NaiveClassifier,EfficientNet_V2_S,EfficientNet_V2_M
from dataset import BirdDataset,Split,BirdDatasetNPZ
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

NUM_WORKERS=2
PRINT_EVERY = 2000
CHECKPOINT_PATH = './statistics/EfficientNet-V2_M_Lion_200Epochs/epoch=119_validation_loss=0.2003_validation_accuracy=0.95_validation_mcc=0.93.ckpt'
BATCH_SIZE = 32 #128 is optimal for TPUs, use multiples of 64 that fits into memory
LEARNING_RATE = 1e-4
WEIGHT_DECAY=1e-2
EPOCHS = 300
MOMENUTUM = 0.9
RESIZE_SIZE = (224,224)


def main():
    torch.set_float32_matmul_precision('medium')
    root_dir = '/content/data/'
    csv_file = '/content/data/birds.csv'
    npz_file = 'data/dataset.npz' #C:/Users/david/Desktop/dataset.npz'
    print(f'root_dir={root_dir}\tcsv_file={csv_file}\tnpz_file={npz_file}')

    #Prepare data transformation pipeline
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(RESIZE_SIZE),
        AugMix(severity=4,mixture_width=4,alpha=0.65),
        transforms.CenterCrop(RESIZE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(), #0-255 -> 0-1
        transforms.Normalize(mean=(0.4742, 0.4694, 0.3954),std=(0.2394, 0.2332, 0.2547))
    ])

    transform_valid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(RESIZE_SIZE),
        transforms.ToTensor(), #0-255 -> 0-1
        transforms.Normalize(mean=(0.4742, 0.4694, 0.3954),std=(0.2394, 0.2332, 0.2547))
    ])

    #Load dataset
    #train_dataset = BirdDataset(root_dir=root_dir,csv_file=csv_file,transform=transform_train,split=Split.TRAIN)
    #valid_datasetset = BirdDataset(root_dir=root_dir,csv_file=csv_file,transform=transform_valid,split=Split.VALID)
    #test_dataset =  BirdDataset(root_dir=root_dir,csv_file=csv_file,transform=transform_valid,split=Split.TEST)

    print('Loading .npz file, this will take a while...')
    data_dict = np.load(npz_file)
    print('Done loading .npz file')
    print('Loading training split')
    train_dataset = BirdDatasetNPZ(data_dict=data_dict,transform=transform_train,split=Split.TRAIN)
    print('Loading valid split')
    valid_datasetset = BirdDatasetNPZ(data_dict=data_dict,transform=transform_valid,split=Split.VALID)
    print('Loading test split')
    test_dataset =  BirdDatasetNPZ(data_dict=data_dict,transform=transform_valid,split=Split.TEST)
    del data_dict

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset=valid_datasetset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS) 


    model_checkpoint = ModelCheckpoint(
                                       dirpath='statistics/',
                                       filename="{epoch}_{validation_loss:.4f}_{validation_accuracy:.2f}_{validation_mcc:.2f}", 
                                       save_top_k=3,
                                       monitor="validation_loss",
                                       mode='min')
    
    model = EfficientNet_V2_S(lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY,batch_size=BATCH_SIZE)
    trainer = pl.Trainer(max_epochs=EPOCHS,callbacks=[model_checkpoint],accelerator='tpu',devices=1,precision='bf16')
    trainer.fit(model=model,train_dataloaders=train_loader,val_dataloaders=valid_loader)

    
if __name__ == '__main__':
    main()