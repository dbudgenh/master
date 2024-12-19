from models import AlexNet, EfficientNet_B0,NaiveClassifier,EfficientNet_V2_S,EfficientNet_V2_M,EfficientNet_V2_S_Pretrained,EfficientNet_V2_L,VisionTransformer_L_16,Resnet_18
from dataset import BirdDataNPZModule,BirdDataModule,BirdDataset,BirdDataModuleV2, FullTestDatamodule,UndersampleSplitDatamodule
import torch
import torchvision

from utils import get_model
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import transforms
from torchvision.transforms.v2 import AugMix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor,EarlyStopping
import numpy as np
from torchvision.models import EfficientNet_V2_S_Weights,EfficientNet_B0_Weights
from transformations import default_transforms,default_collate_fn,old_transforms



#CHECKPOINT_PATH = r'D:\Users\david\Desktop\Python\master\lightning_logs\version_46\checkpoints\AlexNet_V1_epoch=66_validation_loss=1.3744_validation_accuracy=0.952_validation_mcc=0.916.ckpt'
CHECKPOINT_PATH = r'D:\Users\david\Desktop\Python\master\lightning_logs\version_86\checkpoints\VisionTransformer_H_14_Pretrained_fine_tune_V2_epoch=85_validation_loss=1.0218_validation_accuracy=0.992_validation_mcc=0.984.ckpt'
#CHECKPOINT_PATH = r'D:\Users\david\Desktop\Python\master\lightning_logs\version_36\checkpoints\VisionTransformer_H_14_Pretrained_fine_tune_V2_epoch=52_validation_loss=1.0414_validation_accuracy=0.988_validation_mcc=0.977.ckpt'
#CHECKPOINT_PATH = r'D:\Users\david\Desktop\Python\master\statistics\524_Classes\EfficientNet_V2_L_V2_SGD\version_12\checkpoints\EfficientNet_V2_L_V2_epoch=599_validation_loss=1.0233_validation_accuracy=0.99_validation_mcc=0.99.ckpt'
#CHECKPOINT_PATH = r'D:\Users\david\Desktop\Python\master\lightning_logs\version_48\checkpoints\Resnet_18_V1_epoch=299_validation_loss=1.2943_validation_accuracy=0.967_validation_mcc=0.934.ckpt'
BATCH_SIZE = 64 #128 is optimal for TPUs, use multiples of 64 that fit into memory
NUM_WORKERS = 12
EPOCHS = 300
LEARNING_RATE = 0.5
MOMENTUM=0.9
LR_SCHEDULER = 'cosineannealinglr'
LR_WARMUP_EPOCHS = 5
LR_WARMUP_METHOD = 'linear'
LR_WARMUP_DECAY = 0.01
WEIGHT_DECAY = 0.00002
NORM_WEIGHT_DECAY = 0.0
LABEL_SMOOTHING = 0.1

def main():
    torch.set_float32_matmul_precision('medium')
    train_transform, valid_transform, version = default_transforms()#default_transforms(is_vision_transformer=True)
    collate_fn = default_collate_fn()
    total_dataset = ImageFolder(root='./data/train_valid_test/')
    datamodule = UndersampleSplitDatamodule(train_transform=train_transform,
                                            valid_transform=valid_transform,
                                            total_dataset=total_dataset,
                                            random_seed=40,
                                            batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS,
                                            collate_fn=collate_fn)
    # datamodule = FullTestDatamodule(train_transform=train_transform,
    #                                 valid_transform=valid_transform,
    #                                 total_dataset=total_dataset,
    #                                 batch_size=BATCH_SIZE,
    #                                 num_workers=NUM_WORKERS,
    #                                 collate_fn=collate_fn)
    model = get_model(checkpoint_path=CHECKPOINT_PATH)
    model.log_config = {
            'testing':False,
            'confusion_matrix':False,
            'roc_curve':False,
            'auroc':False,
            'classification_report':False,
            'pytorch_cam':False,
            'captum_alg':False,
            'topk':False,
            'bottomk':False,
            'randomk':False,
            'all':False,
            'filter_type':None#'misclassified', #can be 'misclassified', 'correct', or None
            }
    trainer = pl.Trainer(inference_mode=False)
    trainer.test(model,datamodule=datamodule)

    #switching off inference for test
    #trainer.inference_mode = False
    #trainer.test_loop.inference_mode = False
    #trainer.test(model=model,datamodule=bird_data)
    #trainer.test(model=model,datamodule=datamodule,ckpt_path=)

    # log_config = {
    #         'confusion_matrix':False,
    #         'roc_curve':False,
    #         'auroc':False,
    #         'classification_report':False,
    #         'pytorch_cam':False,
    #         'captum_alg':False,
    #         'topk':False,
    #         'bottomk':False,
    #         'randomk':False
    # 

   




    

    
if __name__ == '__main__':
    main()