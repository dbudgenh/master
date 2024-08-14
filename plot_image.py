from image_utils import show_image_from_tensor
from dataset import BirdDataset
import torch
from torchvision.transforms import transforms
from torchvision.transforms.v2 import AugMix,RandAugment
import os
import numpy as np
from torchvision.models import EfficientNet_V2_S_Weights,EfficientNet_B0_Weights

def main():
      #Prepare data and augment data
        trans = transforms.Compose([
            #transforms.ToPILImage(),
            AugMix(severity=4,mixture_width=4,alpha=0.65),
            transforms.CenterCrop(BirdDataset.DEFAULT_RESIZE_SIZE),
            transforms.RandomHorizontalFlip(p=1.0),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomRotation(degrees=45),
            transforms.Resize(BirdDataset.DEFAULT_RESIZE_SIZE),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.RandomErasing(p=1.0)
        ])
        #trans = EfficientNet_V2_S_Weights.DEFAULT.transforms()
        bird_dataset = BirdDataset(root_dir='C:/Users/david/Desktop/Python/master/data/',csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',transform=trans)
    
        for i in range(len(bird_dataset)):
               image, class_id, name = bird_dataset[i]
               show_image_from_tensor(tensor=image,title=f'Name: {name} ClassID: {class_id}')
    


if __name__ == "__main__":
    main()