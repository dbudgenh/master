from train import BATCH_SIZE,NUM_WORKERS,CHECKPOINT_PATH
from torchvision.transforms import transforms
from dataset import BirdDataset,Split,BirdDataModule,BirdDataModuleV2
from torch.utils.data import DataLoader
from models import EfficientNet_V2_S,EfficientNet_V2_S_Pretrained
import pytorch_lightning as pl
import torch
from pytorch_grad_cam import GradCAM, ScoreCAM,HiResCAM,GradCAMElementWise
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from image_utils import array_from_image_path,show_multiple_images
import numpy as np
from skimage.transform import resize
from image_utils import show_image_from_path
from bird_mapping import get_mapping
from browser import search_image_on_google_in_new_tab,close_tab
from train import IMAGENET_MEAN,IMAGENET_STD
from torchvision.transforms.v2 import AugMix
from image_utils import show_image_from_tensor


'Convert Labels from .csv to ImageFolder'
'mapping = .csv mapping'
'bird_to_label = ImageFolder Mapping'

def conversionV1_to_V2(mapping,bird_to_label,label):
    bird_name = mapping[label]
    return bird_to_label[bird_name]

checkpoint_path = 'C:/Users/david/Desktop/Python/master/statistics/EfficientNet_V2_S_Pretrained_Adam/epoch=61_validation_loss=0.0592_validation_accuracy=0.99_validation_mcc=0.96.ckpt'
def main():
    torch.set_float32_matmul_precision('medium')


    train_transform = transforms.Compose([
            #transforms.ToPILImage(),
            AugMix(severity=4,mixture_width=4,alpha=0.65),
            transforms.CenterCrop(BirdDataset.DEFAULT_RESIZE_SIZE),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.Resize(BirdDataset.DEFAULT_RESIZE_SIZE),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD),
            transforms.RandomErasing()
    ])
    valid_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize(BirdDataset.DEFAULT_RESIZE_SIZE),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
    ])

    
    datamodule = BirdDataModuleV2(root_dir='C:/Users/david/Desktop/Python/master/data/',
                                train_transform=train_transform,
                                valid_transform=valid_transform,
                                batch_size=32,num_workers=1)
    
    datamodule2 = BirdDataModule(root_dir='C:/Users/david/Desktop/Python/master/data/',
                                csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',
                                train_transform=train_transform,
                                valid_transform=valid_transform,
                                batch_size=32,num_workers=1)
    
    #MAPPINGS ARE DIFFERENT BETWEEN IMAGEFOLDER AND OUR DATALOADER!!!

    model = EfficientNet_V2_S.load_from_checkpoint(checkpoint_path=checkpoint_path)
    mapping = get_mapping(csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv')
    trainer = pl.Trainer()
    trainer.test(model,datamodule=datamodule2)


    for path,label,prediction,confidence in model.wrong_classifications:
        search_image_on_google_in_new_tab(mapping[label])
        search_image_on_google_in_new_tab(mapping[prediction])
        show_image_from_path(path,title=f'Correct={mapping[label]}\nPredicted={mapping[prediction]}\nConfidence={confidence}')
        close_tab()
        close_tab()




if __name__ == '__main__':
    main()