from dataset import BirdDatasetNPZ,Split
from torchvision import transforms
import numpy as np
from image_utils import show_image_from_tensor

transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor()])
npz_file = 'C:/Users/david/Desktop/dataset.npz'
data_dict = np.load(npz_file,allow_pickle=True)
bird_dataset = BirdDatasetNPZ(data_dict,transform=transform,split=Split.VALID)

for i in range(len(bird_dataset)):
        sample = bird_dataset[i]
        image = sample['image']
        label = sample['class_id']
        print(label)
        dataset = sample['dataset']
        print(dataset)
        show_image_from_tensor(image,title=f"Class: {label}\nDataset: {dataset}")
