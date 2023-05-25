from dataset import BirdDatasetNPZ
from torchvision import transforms
import numpy as np
from image_utils import show_image_from_tensor

transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor()])
bird_dataset = BirdDatasetNPZ(npz_file='dataset.npz',transform=transform,split=None)

for i in range(len(bird_dataset)):
        sample = bird_dataset[i]
        image = sample['image']
        show_image_from_tensor(image,title="Test")
