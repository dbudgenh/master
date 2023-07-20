import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from torchvision.transforms import transforms
from torchvision.transforms.v2 import AugMix
from dataset import BirdDataModule,BirdDataset,BirdDataNPZModule
from models import EfficientNet_V2_S_Pretrained,EfficientNet_V2_M_Pretrained,EfficientNet_V2_L_Pretrained
import pytorch_lightning as pl
from transforms import default_transforms,default_collate_fn

#CHECKPOINT_PATH = 'C:/Users/david/Desktop/Python/master/statistics/EfficientNet_V2_M_Pretrained_Adam/epoch=59_validation_loss=0.6902_validation_accuracy=0.83_validation_mcc=0.82.ckpt'
CHECKPOINT_PATH = 'C:/Users/david/Desktop/Python/master/statistics/EfficientNet_V2_L_Pretrained_Adam/epoch=66_validation_loss=0.2575_validation_accuracy=0.94_validation_mcc=0.94.ckpt'
LEARNING_RATE_FINE_TUNE = 1e-3 #Should be much lower when fine-tuning
WEIGHT_DECAY_FINE_TRAIN = 2e-5
EPOCHS_FINE_TUNE = 150
BATCH_SIZE = 16
NUM_WORKERS = 16



def main():
    torch.set_float32_matmul_precision('medium')
    train_npz_path = 'C:/Users/david/Desktop/dataset_train.npz'
    valid_npz_path = 'C:/Users/david/Desktop/dataset_valid.npz'
    test_npz_path = 'C:/Users/david/Desktop/dataset_test.npz'

    train_transform, valid_transform = default_transforms()
    collate_fn = default_collate_fn()

    datamodule = BirdDataModule(root_dir='C:/Users/david/Desktop/Python/master/data',
                                csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',
                                train_transform=train_transform,
                                valid_transform=valid_transform,
                                batch_size=BATCH_SIZE,
                                collate_fn=collate_fn,
                                num_workers=NUM_WORKERS)
    print(f'Using {CHECKPOINT_PATH} as our checkpoint for the fine-tuning')

    #START FINE-TUNING
    model = EfficientNet_V2_L_Pretrained.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH,
                                                              strict=False,
                                                              lr=LEARNING_RATE_FINE_TUNE,
                                                              weight_decay=WEIGHT_DECAY_FINE_TRAIN,
                                                              batch_size=BATCH_SIZE,
                                                              epochs=EPOCHS_FINE_TUNE,
                                                              mode='fine_tune')
    #Unfreeze some layers for the fine-tuning
    model.unfreeze_layers()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(
                                       dirpath=f'C:/Users/david/Desktop/Python/master/statistics/{model.name}_Finetuned_Adam',
                                       filename="{epoch}_{validation_loss:.4f}_{validation_accuracy:.2f}_{validation_mcc:.2f}", 
                                       save_top_k=1,
                                       monitor="validation_loss",
                                       mode='min')
    trainer = pl.Trainer(max_epochs=EPOCHS_FINE_TUNE,
                         callbacks=[model_checkpoint,lr_monitor],
                         precision='bf16-mixed')
    ckpt_path = 'C:/Users/david/Desktop/Python/master/statistics/EfficientNet_V2_L_Pretrained_Finetuned_Adam/epoch=6_validation_loss=0.0613_validation_accuracy=0.99_validation_mcc=0.98.ckpt'
    trainer.fit(model=model,datamodule=datamodule,ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()