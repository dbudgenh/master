from dataset import KFoldDataModule
from models import NaiveClassifier
import pytorch_lightning as pl
import torch
from transformations import old_transforms


k = 10
EPOCHS = 3
BATCH_SIZE = 128
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
    for index in range(k):
        print(f"Cross validation index: {index}/{k}")
        datamodule = KFoldDataModule(k=k,
                                    index=index,
                                    root_dir='C:/Users/david/Desktop/Python/master/data',
                                    collate_fn=collate_fn,
                                    num_workers=NUM_WORKERS,
                                    batch_size=BATCH_SIZE,
                                    train_transform=train_transform,
                                    valid_transform=valid_trainsform)
        trainer = pl.Trainer(max_epochs=EPOCHS,
                             num_sanity_val_steps=0,
                            precision='bf16-mixed') #
        
        trainer.fit(model=model,
                    datamodule=datamodule)

if __name__ == "__main__":
    main()