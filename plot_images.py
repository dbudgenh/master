import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms.v2 import AugMix,RandAugment
from torch.utils.data import DataLoader
from dataset import BirdDataset  # Replace with your actual module

ROWS, COLS = 3,3
def main():
    # Define the transformation pipeline
    trans = transforms.Compose([
        # AugMix(severity=4, mixture_width=4, alpha=0.65),
        # transforms.CenterCrop(BirdDataset.DEFAULT_RESIZE_SIZE),
        # transforms.RandomHorizontalFlip(p=1.0),
        transforms.Resize(BirdDataset.DEFAULT_RESIZE_SIZE),
        transforms.ToTensor(),  # Converts to a tensor and normalizes values
        #transforms.RandomErasing(p=1.0)
    ])
    
    # Load the dataset
    bird_dataset = BirdDataset(
        root_dir='D:/Users/david/Desktop/Python/master/data/',
        csv_file='D:/Users/david/Desktop/Python/master/data/birds.csv',
        transform=trans
    )
    
    # Set up the plot
    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3, ROWS * 3))
    #fig.suptitle('Random Images from Bird Dataset', fontsize=16)
    
    # Randomly select 9 images from the dataset
    indices = random.sample(range(len(bird_dataset)), ROWS * COLS)
    images_and_labels = [bird_dataset[i] for i in indices]
    
    # Populate the grid
    for ax, (image, class_id, name) in zip(axes.flatten(), images_and_labels):
        # Convert tensor to numpy format for matplotlib
        img_np = image.permute(1, 2, 0).numpy()  # Rearrange dimensions to HxWxC
        ax.imshow(img_np)
        ax.set_title(f'{name}\nClass ID: {class_id}')
        ax.axis('off')  # Hide axes for better visualization
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()