from dataset import BirdDatasetNPZ,Split
from torchvision import transforms
import numpy as np
from image_utils import show_image_from_tensor

transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor()])
npz_file_path = 'C:/Users/david/Desktop/dataset_test.npz'
bird_dataset = BirdDatasetNPZ(npz_file_path=npz_file_path,transform=transform)

for i in range(len(bird_dataset)):
        sample = bird_dataset[i]
        image = sample['image']
        label = sample['class_id']
        show_image_from_tensor(image,title=f"Class: {label}\n")
