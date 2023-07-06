import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from models import EfficientNet_V2_S_Pretrained,EfficientNet_V2_M_Pretrained,EfficientNet_V2_L_Pretrained
from torchvision.transforms import transforms
from torchvision.transforms.v2 import AugMix
from dataset import BirdDataset,BirdDataModule
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import pytorch_lightning as pl
from transforms import default_transforms,default_collate_fn

LEARNING_RATE_PRE_TRAIN = 0.01
WEIGHT_DECAY_PRE_TRAIN = 0.00002
EPOCHS_PRE_TRAIN = 100
BATCH_SIZE = 128
NUM_WORKERS = 16

def main():
    torch.set_float32_matmul_precision('medium')
    
    #Data augmentation pipeline
    train_transform, valid_transform = default_transforms()
    #Mixup-cutmix
    collate_fn = default_collate_fn()

    datamodule = BirdDataModule(root_dir='C:/Users/david/Desktop/Python/master/data',
                                csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',
                                train_transform=train_transform,
                                valid_transform=valid_transform,
                                batch_size=BATCH_SIZE,
                                collate_fn=collate_fn,
                                num_workers=NUM_WORKERS)
    
    #START PRE-TRAINING
    model = EfficientNet_V2_L_Pretrained(lr=LEARNING_RATE_PRE_TRAIN,
                                         weight_decay=WEIGHT_DECAY_PRE_TRAIN, 
                                         batch_size=BATCH_SIZE,
                                         epochs=EPOCHS_PRE_TRAIN,
                                         training_mode='pre_train')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(
                                       dirpath=f'C:/Users/david/Desktop/Python/master/statistics/{model.name}_Pretrained_V2_Adam',
                                       filename="{epoch}_{validation_loss:.4f}_{validation_accuracy:.2f}_{validation_mcc:.2f}", 
                                       save_top_k=1,
                                       monitor="validation_loss",
                                       mode='min')
    
    trainer = pl.Trainer(max_epochs=EPOCHS_PRE_TRAIN,callbacks=[model_checkpoint,lr_monitor],precision='bf16-mixed') #
    #ckpt_path='C:/Users/david/Desktop/Python/master/statistics/EfficientNet_V2_L_Pretrained_Pretrained_V2_Adam/epoch=11_validation_loss=1.6964_validation_accuracy=0.88_validation_mcc=0.87.ckpt'
    trainer.fit(model=model,datamodule=datamodule)


if __name__ == '__main__':
    main()