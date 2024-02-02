from dataset import KFoldDataModule
from models import NaiveClassifier
import pytorch_lightning as pl
import torch
from transformations import old_transforms
from torchvision.datasets import ImageFolder
from dataset import UndersampleMinimumDataset,BirdDataModuleV2


k = 10
EPOCHS = 3
BATCH_SIZE = 256
NUM_WORKERS = 14
LEARNING_RATE = 0.5

def main():
    torch.set_float32_matmul_precision('medium')
    train_transform, valid_trainsform,version = old_transforms()
    collate_fn = None
    model = NaiveClassifier(
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
    )
    image_folder = ImageFolder(
        root='./data/cross_validation',
        transform=None
    )
    total_dataset = UndersampleMinimumDataset(dataset=image_folder)
    datamodule = KFoldDataModule(k=k,
                                 total_dataset=total_dataset,
                                    collate_fn=collate_fn,
                                    num_workers=NUM_WORKERS,
                                    batch_size=BATCH_SIZE,
                                    train_transform=train_transform,
                                    valid_transform=valid_trainsform)
    for kfold_iteration_datamodule in datamodule:
        trainer = pl.Trainer(max_epochs=EPOCHS,
                             num_sanity_val_steps=0,
                            precision='bf16-mixed') #
        
        trainer.fit(model=model,
                    datamodule=kfold_iteration_datamodule)
    

if __name__ == "__main__":
    main()
