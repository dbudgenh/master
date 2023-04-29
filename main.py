from utils import show_image
from dataset import BirdDataset
import torch
from torchvision.transforms import transforms
import os

def main():
    #Check for device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Prepare data and augment data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        #transforms.RandomHorizontalFlip(), #stack data augmentation techniques
        transforms.ToTensor(), #0-255 -> 0-1
        transforms.Normalize(mean=(0.4742, 0.4694, 0.3954),std=(0.2394, 0.2332, 0.2547)) #(tensor([0.4742, 0.4694, 0.3954]), tensor([0.2394, 0.2332, 0.2547]))
    ])
    #Load bird data
    bird_dataset = BirdDataset(root_dir='data',csv_file='data/birds.csv',transform=None)
    #Load snail data
   
    print(f'Using device  {device}')
   

    

   

    
if __name__ == '__main__':
    main()