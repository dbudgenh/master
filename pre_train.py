import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from models import EfficientNet_V2_S_Pretrained,EfficientNet_V2_L_Pretrained
from torchvision.transforms import transforms
from torchvision.transforms.v2 import AugMix
from dataset import BirdDataset,BirdDataModule
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import pytorch_lightning as pl

LEARNING_RATE_PRE_TRAIN = 1e-3
WEIGHT_DECAY_PRE_TRAIN = 1e-2

IMAGENET_MEAN =(0.485, 0.456, 0.406)
IMAGENET_STD = std=(0.229, 0.224, 0.225)

EPOCHS_PRE_TRAIN = 100
BATCH_SIZE = 128
NUM_WORKERS = 16

def main():
    torch.set_float32_matmul_precision('medium')
    train_transform = transforms.Compose([
            #transforms.ToPILImage(),
            AugMix(severity=4,mixture_width=4,alpha=0.65),
            transforms.CenterCrop(BirdDataset.DEFAULT_RESIZE_SIZE),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.Resize(BirdDataset.DEFAULT_RESIZE_SIZE),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD),
            transforms.RandomErasing()
    ])
    valid_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize(BirdDataset.DEFAULT_RESIZE_SIZE),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
    ])

    datamodule = BirdDataModule(root_dir='C:/Users/david/Desktop/Python/master/data',
                                csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',
                                train_transform=train_transform,
                                valid_transform=valid_transform,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS)
    
    #START PRE-TRAINING
    model = EfficientNet_V2_L_Pretrained(lr=LEARNING_RATE_PRE_TRAIN,
                                         weight_decay=WEIGHT_DECAY_PRE_TRAIN, 
                                         batch_size=BATCH_SIZE,
                                         mode='pre_train')
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(
                                       dirpath=f'C:/Users/david/Desktop/Python/master/statistics/{model.name}_Pretrained_Adam',
                                       filename="{epoch}_{validation_loss:.4f}_{validation_accuracy:.2f}_{validation_mcc:.2f}", 
                                       save_top_k=1,
                                       monitor="validation_loss",
                                       mode='min')
    
    trainer = pl.Trainer(max_epochs=EPOCHS_PRE_TRAIN,callbacks=[model_checkpoint,lr_monitor],precision='bf16-mixed') #
    trainer.fit(model=model,datamodule=datamodule)


if __name__ == '__main__':
    main()