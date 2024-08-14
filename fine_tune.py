import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.transforms.v2 import AugMix
from dataset import BirdDataModule,BirdDataset,BirdDataNPZModule,BirdDataModuleV2,UndersampleSplitDatamodule
from models import EfficientNet_V2_S_Pretrained,EfficientNet_V2_M_Pretrained,EfficientNet_V2_L_Pretrained,VisionTransformer_L_16_Pretrained,VisionTransformer_H_14_Pretrained
import pytorch_lightning as pl
from transformations import default_transforms,default_collate_fn

#CHECKPOINT_PATH = 'C:/Users/david/Desktop/Python/master/statistics/EfficientNet_V2_M_Pretrained_Adam/epoch=59_validation_loss=0.6902_validation_accuracy=0.83_validation_mcc=0.82.ckpt'
CHECKPOINT_PATH = r"C:\Users\david\Desktop\Python\master\lightning_logs\version_25\checkpoints\VisionTransformer_H_14_Pretrained_pre_train_V2_epoch=74_validation_loss=1.3038_validation_accuracy=0.958_validation_mcc=0.945.ckpt"
LEARNING_RATE_FINE_TUNE = 0.0001 #Should be much lower when fine-tuning
EPOCHS_FINE_TUNE = 100
BATCH_SIZE = 32
NUM_WORKERS = 8 


def main():
    torch.set_float32_matmul_precision('medium')
    train_transform, valid_transform, version = default_transforms(is_vision_transformer=True)
    collate_fn = default_collate_fn()

    total_dataset = ImageFolder(root='./data/train_valid_test/')
    datamodule = UndersampleSplitDatamodule(train_transform=train_transform,
                                            valid_transform=valid_transform,
                                            total_dataset=total_dataset,
                                            random_seed=43,
                                            batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS,
                                            collate_fn=collate_fn)
    
    print(f'Using {CHECKPOINT_PATH} as our checkpoint for the fine-tuning')
    model = VisionTransformer_H_14_Pretrained.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH,
                                                              strict=False,
                                                              lr=LEARNING_RATE_FINE_TUNE,
                                                              batch_size=BATCH_SIZE,
                                                              epochs=EPOCHS_FINE_TUNE,
                                                              num_workers=NUM_WORKERS,
                                                              training_mode='fine_tune',
                                                              norm_weight_decay=0.0,
                                                              label_smoothing=0.1,
                                                              lr_warmup_epochs=5,
                                                              lr_warmup_method='linear',
                                                              optimizer_algorithm='sgd',
                                                              lr_warmup_decay=0.01)


    #Unfreeze some layers for the fine-tuning
    model.unfreeze_layers()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(
                                       filename=f"{model.name}_{version}_"+ "{epoch}_{validation_loss:.4f}_{validation_accuracy:.3f}_{validation_mcc:.3f}", 
                                       save_top_k=1,
                                       monitor="validation_loss",
                                       mode='min')
    trainer = pl.Trainer(max_epochs=EPOCHS_FINE_TUNE,
                         callbacks=[model_checkpoint,lr_monitor],
                         precision='bf16-mixed')
    ckpt_path = r"C:\Users\david\Desktop\Python\master\lightning_logs\version_36\checkpoints\VisionTransformer_H_14_Pretrained_fine_tune_V2_epoch=52_validation_loss=1.0414_validation_accuracy=0.988_validation_mcc=0.977.ckpt"
    trainer.fit(model=model,datamodule=datamodule,ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()