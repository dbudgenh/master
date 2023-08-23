from models import EfficientNet_V2_L,EfficientNet_V2_S,KnowledgeDistillationModule
import torch
from transforms import old_transforms
from dataset import BirdDataModule
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import pytorch_lightning as pl

TEACHER_CHECKPOINT = 'C:/Users/david/Desktop/Python/master/statistics/EfficientNet_V2_L_Finetuned_Adam/epoch=28_validation_loss=0.0459_validation_accuracy=0.99_validation_mcc=0.99.ckpt'
LEARNING_RATE = 0.5
MOMENTUM = 0.9
WEIGHT_DECAY = 2e-5
BATCH_SIZE = 128
LR_SCHEDULER = 'cosineannealinglr'
LR_WARMUP_EPOCHS = 5
LR_WARMUP_METHOD = 'linear'
LR_WARMUP_DECAY = 0.01
EPOCHS = 500
ALPHA = 0.95
TEMPERATURE = 3.5
NUM_WORKERS = 12

def main():
    torch.set_float32_matmul_precision('medium')
    train_transform, valid_transform = old_transforms()
    datamodule = BirdDataModule(root_dir='C:/Users/david/Desktop/Python/master/data',
                                csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',
                                train_transform=train_transform,
                                valid_transform=valid_transform,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                collate_fn=None)
    teacher = EfficientNet_V2_L.load_from_checkpoint(checkpoint_path=TEACHER_CHECKPOINT)
    student = EfficientNet_V2_S()
    kd_model= KnowledgeDistillationModule(student_model=student,
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
                                          alpha=ALPHA,
                                          T=TEMPERATURE)
    
    lr_monitor = LearningRateMonitor(logging_interval='step',
                                     log_momentum=False)
    model_checkpoint = ModelCheckpoint(
                                       #dirpath=rf'C:/Users/david/Desktop/test',
                                       dirpath=rf'C:/Users/david/Desktop/Python/master/statistics/{kd_model.name}',
                                       filename="{epoch}_{validation_loss:.4f}_{validation_accuracy:.2f}_{validation_mcc:.2f}", #+ f'version={kd_model.get_version_number()}', 
                                       save_top_k=1,
                                       save_weights_only=False,
                                       monitor="validation_loss",
                                       mode='min')
    trainer = pl.Trainer(max_epochs=EPOCHS,callbacks=[model_checkpoint,lr_monitor],precision='bf16-mixed')
    #ckpt_path = 'C:/Users/david/Desktop/test/epoch=303_validation_loss=0.4812_validation_accuracy=0.99_validation_mcc=0.97.ckpt'
    trainer.fit(model=kd_model,datamodule=datamodule)#,ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()