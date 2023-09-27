from train import BATCH_SIZE,NUM_WORKERS,CHECKPOINT_PATH
from torchvision.transforms import transforms
from dataset import BirdDataset,Split
from torch.utils.data import DataLoader
from models import EfficientNet_V2_M,EfficientNet_V2_S,EfficientNet_V2_L
import pytorch_lightning as pl
import torch
from pytorch_grad_cam import GradCAM, ScoreCAM,HiResCAM,GradCAMElementWise
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from image_utils import array_from_image_path,show_multiple_images
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

USE_PRETRAINED_NETWORK = True

def main():
    mean = BirdDataset.MEAN if USE_PRETRAINED_NETWORK else (0.4742, 0.4694, 0.3954)
    std =  BirdDataset.STD if USE_PRETRAINED_NETWORK else (0.2394, 0.2332, 0.2547)
    torch.set_float32_matmul_precision('medium')
    transform_valid = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(), #0-255 -> 0-1
        transforms.Normalize(mean=mean,std=std)
    ])

    test_dataset =  BirdDataset(root_dir='C:/Users/david/Desktop/Python/master/data/',csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',transform=transform_valid,split=Split.TEST)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS) 

    ckpt_path = 'C:/Users/david/Desktop/Python/master/statistics/EfficientNet_V2_L_Finetuned_Adam/epoch=28_validation_loss=0.0459_validation_accuracy=0.99_validation_mcc=0.99.ckpt'
    input_tensor,labels,paths = next(iter(test_loader))
    model = EfficientNet_V2_L.load_from_checkpoint(checkpoint_path=ckpt_path)

    target_layers = [model.efficient_net.features[1]]
    cam = GradCAM(model=model.efficient_net,target_layers=target_layers,use_cuda=True)
    targets = [ClassifierOutputTarget(524)]*BATCH_SIZE
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets,aug_smooth=True,eigen_smooth=False)

    for i in range(BATCH_SIZE):
        rgb_image = resize(array_from_image_path(paths[i]),(224,224))
        image_cam = grayscale_cam[i]
        visualization = show_cam_on_image(rgb_image, image_cam, use_rgb=True,image_weight=0.5)
        print(paths[i])
        show_multiple_images(images=[rgb_image,image_cam,visualization],titles=['Original','Heat map', 'Combined'])


if __name__ == '__main__':
    main()