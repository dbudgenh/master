from image_utils import show_image_from_tensor
from dataset import BirdDataset
import torch
from torchvision.transforms import transforms
from torchvision.transforms.v2 import AugMix
import os
import numpy as np

def main():
      #Prepare data and augment data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        AugMix(severity=4,mixture_width=4,alpha=0.65),
        transforms.RandomHorizontalFlip(1),
        #transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(), #0-255 -> 0-1
        #transforms.Normalize() #(tensor([0.4742, 0.4694, 0.3954]), tensor([0.2394, 0.2332, 0.2547]))
    ])
    bird_dataset = BirdDataset(root_dir='data',csv_file='data/birds.csv',transform=transform)
    
    for i in range(len(bird_dataset)):
        sample = bird_dataset[i]
        image,label,class_id = sample['image'],sample['label'],sample['class_id']
        show_image_from_tensor(tensor=image,title=f'Label: {label} \n Class-ID: {class_id}')
        break
    


if __name__ == "__main__":
    main()