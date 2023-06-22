from dataset import BirdDataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm

transform = transforms.Compose([transforms.Resize((224,224))])
bird_dataset = BirdDataset(root_dir='C:/Users/david/Desktop/Python/master/data',csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',transform=transform)

data = []
labels = []
dataset = []

for i in tqdm(range(len(bird_dataset))):
        image,label,split = bird_dataset[i]
        data.append(np.moveaxis(np.array(image),-1,0))
        labels.append(label)
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

train_filter = dataset == 0
train_data  = data[train_filter]
train_labels = labels[train_filter]

np.savez_compressed('C:/Users/david/Desktop/dataset_train.npz',images=train_data,labels=train_labels)

del train_data
del train_labels

valid_filter = dataset == 1
valid_data  = data[valid_filter]
valid_labels = labels[valid_filter]

np.savez_compressed('C:/Users/david/Desktop/dataset_valid.npz',images=valid_data,labels=valid_labels)

del valid_data
del valid_labels

test_filter = dataset == 2
test_data  = data[test_filter]
test_labels = labels[test_filter]

np.savez_compressed('C:/Users/david/Desktop/dataset_test.npz',images=test_data,labels=test_labels)



