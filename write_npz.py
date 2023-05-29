from dataset import BirdDataset
from torchvision import transforms
import numpy as np

transform = transforms.Compose([transforms.Resize((224,224))])
bird_dataset = BirdDataset(root_dir='C:/Users/david/Desktop/Python/master/data',csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',transform=transform)

data = []
labels = []
dataset = []

for i in range(len(bird_dataset)):
        sample = bird_dataset[i]
        data.append(sample['image'].numpy())
        labels.append(sample['class_id'])
        split = sample['dataset']
        split_int = 0
        if split == 'train':
                split_int = 0
        if split == 'valid':
                split_int = 1
        if split == 'test':
                split_int = 2
        dataset.append(split_int)
data = np.array(data,dtype=np.uint8)
labels = np.array(labels,dtype=np.int16)
dataset = np.array(dataset,dtype=np.uint8)

print(data.shape)
print(labels.shape)
print(dataset.shape)

np.savez_compressed('C:/Users/david/Desktop/dataset.npz',images=data,labels=labels,dataset=dataset)
