from models import EfficientNet_V2_L,EfficientNet_V2_S, OfflineFeatureBasedDistillation,OfflineResponseBasedDistillation
import torch
from transformations import old_transforms
from dataset import BirdDataModule,BirdDataModuleV2,UndersampleMinimumDataset
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import pytorch_lightning as pl

TEACHER_CHECKPOINT = r'C:\Users\david\Desktop\Python\master\statistics\524_Classes\EfficientNet_V2_L_Finetuned_V2_SGD\version_2\checkpoints\EfficientNet_V2_L_Pretrained_fine_tune_V2_epoch=99_validation_loss=1.0096_validation_accuracy=0.99_validation_mcc=0.99.ckpt'
LEARNING_RATE = 0.5
MOMENTUM = 0.9
WEIGHT_DECAY = 2e-5
BATCH_SIZE = 32
LR_SCHEDULER = 'cosineannealinglr'
LR_WARMUP_EPOCHS = 5
LR_WARMUP_METHOD = 'linear'
LR_WARMUP_DECAY = 0.01
EPOCHS = 500
ALPHA = 0.95
TEMPERATURE = 3.5
NUM_WORKERS = 4

def main():
    torch.set_float32_matmul_precision('medium')
    train_transform, valid_transform,version = old_transforms()
    
    datamodule = BirdDataModuleV2(root_dir='C:/Users/david/Desktop/Python/master/data',
                                train_transform=train_transform,
                                valid_transform=valid_transform,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                collate_fn=None)
    
    teacher = EfficientNet_V2_L.load_from_checkpoint(checkpoint_path=TEACHER_CHECKPOINT)
    student = EfficientNet_V2_S()

    # kd_model= OfflineResponseBasedDistillation(student_model=student,
    #                                       teacher_model=teacher,
    #                                       lr=LEARNING_RATE,
    #                                       weight_decay=WEIGHT_DECAY,
    #                                       momentum=MOMENTUM,
    #                                       batch_size=BATCH_SIZE,
    #                                       lr_scheduler=LR_SCHEDULER,
    #                                       lr_warmup_epochs=LR_WARMUP_EPOCHS,
    #                                       lr_warmup_method=LR_WARMUP_METHOD,
    #                                       lr_warmup_decay=LR_WARMUP_DECAY,
    #                                       epochs=EPOCHS,
    #                                       alpha=ALPHA,
    #                                       T=TEMPERATURE,
    #                                       num_workers=NUM_WORKERS)
    
    kd_model= OfflineFeatureBasedDistillation(student_model=student,
                                          teacher_model=teacher,
                                          lr=LEARNING_RATE,
                                          weight_decay=WEIGHT_DECAY,
                                          momentum=MOMENTUM,
                                          batch_size=BATCH_SIZE,
                                          lr_scheduler=LR_SCHEDULER,
                                          lr_warmup_epochs=LR_WARMUP_EPOCHS,
                                          lr_warmup_method=LR_WARMUP_METHOD,
                                          lr_warmup_decay=LR_WARMUP_DECAY,
                                          epochs=EPOCHS,
                                          num_workers=NUM_WORKERS)
    
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model_checkpoint = ModelCheckpoint(
                                       filename=f"{kd_model.name}_{version}_"+ "{epoch}_{validation_loss:.4f}_{validation_accuracy:.2f}_{validation_mcc:.2f}",
                                       save_top_k=1,
                                       monitor="validation_loss",
                                       mode='min')
    
    trainer = pl.Trainer(max_epochs=EPOCHS,
                         callbacks=[model_checkpoint,lr_monitor],
                         precision='bf16-mixed',
                         num_sanity_val_steps=0)
    trainer.fit(model=kd_model,datamodule=datamodule)#,ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()