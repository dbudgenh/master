from models import AlexNet, EfficientNet_B0,NaiveClassifier,EfficientNet_V2_S,EfficientNet_V2_M,EfficientNet_V2_S_Pretrained,EfficientNet_V2_L, VisionTransformer_B_16, VisionTransformer_H_14,VisionTransformer_L_16,Resnet_18
from dataset import BirdDataNPZModule,BirdDataModule,BirdDataset,BirdDataModuleV2,UndersampleSplitDatamodule
import torch
import torchvision
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



CHECKPOINT_PATH = r'D:\Users\david\Desktop\Python\master\lightning_logs\version_44\checkpoints\AlexNet_V1_epoch=62_validation_loss=1.8134_validation_accuracy=0.869_validation_mcc=0.788.ckpt'
BATCH_SIZE = 64 #128 is optimal for TPUs, use multiples of 64 that fit into memory
NUM_WORKERS = 1
EPOCHS = 600
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
    train_transform, valid_transform, version = default_transforms(is_vision_transformer=False)
    collate_fn = default_collate_fn()

    # datamodule = BirdDataModuleV2(root_dir='C:/Users/david/Desktop/Python/master/data',
    #                             train_transform=train_transform,
    #                             valid_transform=valid_transform,
    #                             batch_size=BATCH_SIZE,
    #                             num_workers=NUM_WORKERS,
    #                             collate_fn=collate_fn)

    total_dataset = ImageFolder(root='./data/train_valid_test/')
    datamodule = UndersampleSplitDatamodule(train_transform=train_transform,
                                            valid_transform=valid_transform,
                                            total_dataset=total_dataset,
                                            random_seed=40,
                                            batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS,
                                            collate_fn=collate_fn)
    model = VisionTransformer_H_14(lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY,
                              momentum=MOMENTUM,
                              norm_weight_decay=NORM_WEIGHT_DECAY,
                              batch_size=BATCH_SIZE,
                              label_smoothing=LABEL_SMOOTHING,
                              lr_scheduler=LR_SCHEDULER,
                              lr_warmup_epochs=LR_WARMUP_EPOCHS,
                              lr_warmup_method=LR_WARMUP_METHOD,
                              lr_warmup_decay=LR_WARMUP_DECAY,
                              epochs=EPOCHS,
                              num_workers=NUM_WORKERS,
                              optimizer_algorithm='sgd')
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(
                                       filename=f"{model.name}_{version}_"+ "{epoch}_{validation_loss:.4f}_{validation_accuracy:.3f}_{validation_mcc:.3f}", 
                                       save_top_k=1,
                                       verbose=True,
                                       monitor="validation_loss",
                                       mode='min')
    
    trainer = pl.Trainer(max_epochs=EPOCHS,callbacks=[model_checkpoint,lr_monitor],precision='bf16-mixed') #
    trainer.fit(model=model,datamodule=datamodule,)#ckpt_path=CHECKPOINT_PATH)
    model.log_text_to_tensorboard('best_checkpoint_file_name',model_checkpoint.best_model_path)

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