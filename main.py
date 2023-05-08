from image_utils import show_image
from model_utils import get_model
from dataset import BirdDataset,Split
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


NUM_WORKERS = 8
PRINT_EVERY = 1000

#Hyperparameters for training
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 100
MOMENUTUM = 0.9
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
    model = get_model('naive')
    
    #Load pre-trained model (with imagenet mean and std)
    #weights = EfficientNet_B0_Weights.DEFAULT
    #model = efficientnet_b0(weights=weights)
    #Load the preprocessing steps for IMAGENET1K_V1.
    #preprocess = weights.transforms()
    
    
    
    #Define loss function
    criterion = nn.CrossEntropyLoss()
    #Define optimizer
    optimizer = optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=MOMENUTUM)
    #Training loop
    print('Starting training')
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data['image'], data['label']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % PRINT_EVERY == (PRINT_EVERY-1):    # print every couple mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINT_EVERY:.3f}')
                running_loss = 0.0
    

    
if __name__ == '__main__':
    main()