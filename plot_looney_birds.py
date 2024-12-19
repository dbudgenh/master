import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

ROWS, COLS = 1, 4

def main(folder_path):
    # Define the transformation pipeline for PIL images
    trans = transforms.Compose([
        transforms.Resize((224, 224)),  # Using standard size instead of dataset constant
        transforms.ToTensor(),
    ])
    
    # Get all image files from the folder
    image_files = list(Path(folder_path).glob('*.[jJ][pP][gG]'))  # handles jpg/JPG
    image_files.extend(Path(folder_path).glob('*.[pP][nN][gG]'))  # handles png/PNG
    
    # Randomly sample images
    selected_files = random.sample(image_files, min(ROWS * COLS, len(image_files)))
    
    # Set up the plot
    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3, ROWS * 3))
    axes = axes.flatten()
    
    # Process and display images
    images = []
    for file_path in selected_files:
        img = Image.open(file_path)
        img_tensor = trans(img)
        images.append(img_tensor.permute(1, 2, 0).numpy())
    
    # Convert to numpy array in batch
    images = np.stack(images)
    
    # Populate the grid efficiently
    for idx, (img, file_path) in enumerate(zip(images, selected_files)):
        axes[idx].imshow(img)
        axes[idx].set_title(f'LOONEY BIRDS\nCLASS ID: 326')
        axes[idx].axis('off')
    
    # Clear remaining axes
    for ax in axes[len(images):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder_path = r'D:\Users\david\Desktop\Python\master\images\LOONEY BIRDS'
    main(folder_path)