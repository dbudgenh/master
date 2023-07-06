from torchvision.transforms import transforms
from dataset import BirdDataset,Split,BirdDataModule,BirdDataModuleV2
from torch.utils.data import DataLoader
from models import EfficientNet_V2_S,EfficientNet_V2_S_Pretrained,EfficientNet_V2_L
import pytorch_lightning as pl
import torch
from pytorch_grad_cam import GradCAM, ScoreCAM,HiResCAM,GradCAMElementWise
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
from image_utils import show_image_from_path
from bird_mapping import get_mapping
from browser import search_image_on_google_in_new_tab,close_tab
from torchvision.transforms.v2 import AugMix
from transforms import old_transforms


'Convert Labels from .csv to ImageFolder'
'mapping = .csv mapping'
'bird_to_label = ImageFolder Mapping'
def conversionV1_to_V2(mapping,bird_to_label,label):
    bird_name = mapping[label]
    return bird_to_label[bird_name]

checkpoint_path = 'C:/Users/david/Desktop/Python/master/statistics/EfficientNet_V2_L_Finetuned_Adam/epoch=28_validation_loss=0.0459_validation_accuracy=0.99_validation_mcc=0.99.ckpt'
#checkpoint_path = 'C:/Users/david/Desktop/Python/master/statistics/EfficientNet_V2_S_Pretrained_Adam/epoch=61_validation_loss=0.0592_validation_accuracy=0.99_validation_mcc=0.96.ckpt'
def main():
    torch.set_float32_matmul_precision('medium')
    train_transform, valid_transform = old_transforms()

    datamodule = BirdDataModule(root_dir='C:/Users/david/Desktop/Python/master/data/',
                                csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',
                                train_transform=train_transform,
                                valid_transform=valid_transform,
                                batch_size=16,num_workers=2)
    
    #MAPPINGS ARE DIFFERENT BETWEEN IMAGEFOLDER AND OUR DATALOADER!!!
    model = EfficientNet_V2_L.load_from_checkpoint(checkpoint_path=checkpoint_path)
    mapping = get_mapping(csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv')
    trainer = pl.Trainer()
    trainer.test(model,datamodule=datamodule)


    for path,label,prediction,confidence in model.wrong_classifications:
        search_image_on_google_in_new_tab(mapping[label])
        search_image_on_google_in_new_tab(mapping[prediction])
        show_image_from_path(path,title=f'Correct={mapping[label]}\nPredicted={mapping[prediction]}\nConfidence={confidence}')
        close_tab()
        close_tab()




if __name__ == '__main__':
    main()