import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.datasets import ImageFolder
from models import EfficientNet_V2_S_Pretrained,EfficientNet_V2_M_Pretrained,EfficientNet_V2_L_Pretrained,VisionTransformer_L_16_Pretrained,VisionTransformer_H_14_Pretrained
from torchvision.transforms import transforms
from torchvision.transforms.v2 import AugMix
from dataset import BirdDataset,BirdDataModule,BirdDataNPZModule,BirdDataModuleV2,UndersampleSplitDatamodule
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import pytorch_lightning as pl
from transformations import default_transforms,default_collate_fn

LEARNING_RATE_PRE_TRAIN = 0.1
WEIGHT_DECAY_PRE_TRAIN = 0.00002
MOMENTUM=0.9
EPOCHS_PRE_TRAIN = 75
BATCH_SIZE = 64
NUM_WORKERS = 12

def main():
    torch.set_float32_matmul_precision('medium')
    #Data augmentation pipeline
    train_transform, valid_transform, version = default_transforms(use_npz_dataset=False,is_vision_transformer=True)
    #Mixup-cutmix
    collate_fn = default_collate_fn()

    total_dataset = ImageFolder(root='./data/train_valid_test/')
    datamodule = UndersampleSplitDatamodule(train_transform=train_transform,
                                            valid_transform=valid_transform,
                                            total_dataset=total_dataset,
                                            random_seed=43,
                                            batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS,
                                            collate_fn=collate_fn)

    #START PRE-TRAINING
    model = VisionTransformer_H_14_Pretrained(lr=LEARNING_RATE_PRE_TRAIN,
                                         weight_decay=WEIGHT_DECAY_PRE_TRAIN, 
                                         momentum=MOMENTUM,
                                         batch_size=BATCH_SIZE,
                                         epochs=EPOCHS_PRE_TRAIN,
                                         num_workers=NUM_WORKERS,
                                         training_mode='pre_train')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(
                                       filename=f"{model.name}_{version}_"+ "{epoch}_{validation_loss:.4f}_{validation_accuracy:.3f}_{validation_mcc:.3f}",
                                       save_top_k=1,
                                       monitor="validation_loss",
                                       mode='min')
    trainer = pl.Trainer(max_epochs=EPOCHS_PRE_TRAIN,callbacks=[model_checkpoint,lr_monitor],precision='bf16-mixed')
    trainer.fit(model=model,datamodule=datamodule)


if __name__ == '__main__':
    main()