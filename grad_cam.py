from train import BATCH_SIZE,NUM_WORKERS,CHECKPOINT_PATH
from torchvision.transforms import transforms
from dataset import BirdDataset,Split
from torch.utils.data import DataLoader
from models import EfficientNet_V2_M,EfficientNet_V2_S,EfficientNet_V2_L,VisionTransformer_L_16_Pretrained,VisionTransformer_L_16,VisionTransformer_H_14
import pytorch_lightning as pl
from pytorch_grad_cam import GradCAM, ScoreCAM,HiResCAM,GradCAMElementWise
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from image_utils import array_from_image_path,show_multiple_images
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import utils
from transformations import default_transforms,default_collate_fn
from functools import partial

USE_PRETRAINED_NETWORK = True

def main():
    train_transform, valid_transform, version = default_transforms(is_vision_transformer=True)
    collate_fn = default_collate_fn()

    test_dataset =  BirdDataset(root_dir='C:/Users/david/Desktop/Python/master/data/',csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',transform=valid_transform,split=Split.TEST)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True, num_workers=2) 

    ckpt_path = r'C:\Users\david\Desktop\Python\master\lightning_logs\version_36\checkpoints\VisionTransformer_H_14_Pretrained_fine_tune_V2_epoch=52_validation_loss=1.0414_validation_accuracy=0.988_validation_mcc=0.977.ckpt'
    model = VisionTransformer_H_14.load_from_checkpoint(checkpoint_path=ckpt_path)
    model = model.eval()
    
    input_tensor,labels,paths = next(iter(test_loader))
    
    target_layers = [model.model.encoder.layers[-1].ln_1]
    reshape_transform = partial(utils.reshape_transform,height=16,width=16) #height=14, width=14
    cam = GradCAM(model=model.model,target_layers=target_layers,use_cuda=True,reshape_transform=reshape_transform)
    targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets,aug_smooth=False,eigen_smooth=False)

    for i in range(16):
        rgb_image = resize(array_from_image_path(paths[i]),(224,224))
        image_cam = grayscale_cam[i]
        visualization = show_cam_on_image(rgb_image, image_cam, use_rgb=True,image_weight=0.5,colormap=20)
        print(paths[i])
        show_multiple_images(images=[rgb_image,image_cam,visualization],titles=['Original','Heat map', 'Combined'])



if __name__ == '__main__':
    main()