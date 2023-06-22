import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from torchvision.transforms import transforms
from torchvision.transforms.v2 import AugMix
from dataset import BirdDataModule,BirdDataset,BirdDataNPZModule
from models import EfficientNet_V2_S_Pretrained,EfficientNet_V2_L_Pretrained
import pytorch_lightning as pl

#CHECKPOINT_PATH = 'C:/Users/david/Desktop/Python/master/statistics/EfficientNet_V2_L_Pretrained_Pretrained_Adam/epoch=66_validation_loss=0.2575_validation_accuracy=0.94_validation_mcc=0.94.ckpt'
CHECKPOINT_PATH = 'C:/Users/david/Desktop/Python/master/statistics/EfficientNet_V2_L_Pretrained_Adam/epoch=0_validation_loss=0.0879_validation_accuracy=0.98_validation_mcc=0.96.ckpt'
LEARNING_RATE_FINE_TUNE = 1e-4 #Should be much lower when fine-tuning
WEIGHT_DECAY_FINE_TRAIN =1e-2
EPOCHS_FINE_TUNE = 150
BATCH_SIZE = 16
NUM_WORKERS = 3
IMAGENET_MEAN =(0.485, 0.456, 0.406)
IMAGENET_STD = std=(0.229, 0.224, 0.225)



def main():
    torch.set_float32_matmul_precision('medium')
    train_npz_path = 'C:/Users/david/Desktop/dataset_train.npz'
    valid_npz_path = 'C:/Users/david/Desktop/dataset_valid.npz'
    test_npz_path = 'C:/Users/david/Desktop/dataset_test.npz'
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
    
#     datamodule2 = BirdDataNPZModule(train_npz_path=train_npz_path,
#                                     valid_npz_path=valid_npz_path,
#                                     test_npz_path=test_npz_path,
#                                     train_transform=train_transform,
#                                     valid_transform=valid_transform,
#                                     batch_size=BATCH_SIZE,
#                                     num_workers=NUM_WORKERS)

    
    print(f'Using {CHECKPOINT_PATH} as our checkpoint for the fine-tuning')

    #START FINE-TUNING
    model = EfficientNet_V2_L_Pretrained.load_from_checkpoint(CHECKPOINT_PATH,
                                                              lr=LEARNING_RATE_FINE_TUNE,
                                                              weight_decay=WEIGHT_DECAY_FINE_TRAIN,
                                                              batch_size=BATCH_SIZE,
                                                              mode='fine_tune')
    #Unfreeze some layers for the fine-tuning
    model.unfreeze_layers()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(
                                       dirpath=f'C:/Users/david/Desktop/Python/master/statistics/{model.name}_Adam',
                                       filename="{epoch}_{validation_loss:.4f}_{validation_accuracy:.2f}_{validation_mcc:.2f}", 
                                       save_top_k=1,
                                       monitor="validation_loss",
                                       mode='min')
   
    
    trainer = pl.Trainer(max_epochs=EPOCHS_FINE_TUNE,
                         callbacks=[model_checkpoint,lr_monitor],
                         precision='bf16-mixed')
    trainer.fit(model=model,datamodule=datamodule)

if __name__ == '__main__':
    main()