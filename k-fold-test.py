from dataset import KFoldDataModule
from models import NaiveClassifier
import pytorch_lightning as pl
import torch
from transformations import old_transforms
from torchvision.datasets import ImageFolder
from dataset import UndersampleMinimumDataset,BirdDataModuleV2
from pytorch_lightning.callbacks import ModelCheckpoint
import re
import statistics

k = 10
EPOCHS = 3
BATCH_SIZE = 256
NUM_WORKERS = 14
LEARNING_RATE = 0.5

def main():
    torch.set_float32_matmul_precision('medium')
    train_transform, valid_trainsform,version = old_transforms()
    collate_fn = None
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
    
    validation_accuracies = []
    validation_loss = []

    for kfold_iteration_datamodule in datamodule:
        model = NaiveClassifier(
            lr=LEARNING_RATE,
            batch_size=BATCH_SIZE,
        )
        model_checkpoint = ModelCheckpoint(
                                       filename=f"{model.name}_{version}_"+ "{epoch}_{validation_loss:.4f}_{validation_accuracy:.2f}_{validation_mcc:.2f}", 
                                       save_top_k=1,
                                       verbose=True,
                                       monitor="validation_loss",
                                       mode='min')
        
        trainer = pl.Trainer(max_epochs=EPOCHS,
                            num_sanity_val_steps=0,
                            callbacks=[model_checkpoint],
                            precision='bf16-mixed') #
        
        trainer.fit(model=model,
                    datamodule=kfold_iteration_datamodule)
        

        validation_loss.append(round(float(model_checkpoint.best_model_score),4))
        validation_accuracies.append(float(re.search(r'validation_accuracy=([0-9.]+)', model_checkpoint.best_model_path).group(1)))

    avg_model_loss = round(statistics.fmean(validation_loss),4)
    avg_model_acc = round(statistics.fmean(validation_accuracies),4)
    stdev_model_loss = round(statistics.stdev(validation_loss),4)
    stdev_model_acc = round(statistics.stdev(validation_accuracies),4)

    #log average and standard deviation across models/folds losses and validation accuracies
    model.logger.experiment.add_text('CV_average_loss_with_stdev',str(avg_model_loss) + " +- " + str(stdev_model_loss))
    model.logger.experiment.add_text('CV_average_validation_accuracy_with_stdev',str(avg_model_acc) + " +- " + str(stdev_model_acc))
    

if __name__ == "__main__":
    main()
