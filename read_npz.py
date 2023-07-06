from dataset import BirdDatasetNPZ,Split,BirdDataset
from torchvision import transforms
import numpy as np
from image_utils import show_image_from_tensor

mean = BirdDataset.MEAN
std =  BirdDataset.STD

def standardize(x):
        return x / 255.0

transform_train = transforms.Compose([transforms.Resize((224,224)),])
npz_file_path = 'C:/Users/david/Desktop/dataset_train.npz'
bird_dataset = BirdDatasetNPZ(npz_file_path=npz_file_path,transform=transform_train)

for i in range(len(bird_dataset)):
        image,label = bird_dataset[i]
        show_image_from_tensor(image,title=f"Class: {label}\n")
