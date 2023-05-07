from utils import show_image
from dataset import BirdDataset,Split
import torch
from torchvision.transforms import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader


NUM_WORKERS = 8

#Hyperparameters for training
BATCH_SIZE = 16
LEARNING_RATE = 0.1
RESIZE_SIZE = (244,244)



def main():
    #Check for device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    #Prepare data and augment data

    #show_image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(RESIZE_SIZE),
        #transforms.RandomHorizontalFlip(), #stack data augmentation techniques
        transforms.ToTensor(), #0-255 -> 0-1
        transforms.Normalize(mean=(0.4742, 0.4694, 0.3954),std=(0.2394, 0.2332, 0.2547))
    ])

    #Load dataset
    train_dataset = BirdDataset(root_dir='data',csv_file='data/birds.csv',transform=None,split=Split.TRAIN)
    valid_datasetset = BirdDataset(root_dir='data',csv_file='data/birds.csv',transform=None,split=Split.VALID)
    test_dataset =  BirdDataset(root_dir='data',csv_file='data/birds.csv',transform=None,split=Split.TEST)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset=valid_datasetset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS) 

    #Load model
    model = efficientnet_b0(weights=None)
    
    #Load pre-trained model (with imagenet mean and std)
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    #Load the preprocessing steps for IMAGENET1K_V1.
    preprocess = weights.transforms()

    #Training loop

    #train on training set
    #based on validation accuracy, change 





   
    
   

    



    
if __name__ == '__main__':
    main()