from train import RESIZE_SIZE,BATCH_SIZE,NUM_WORKERS,CHECKPOINT_PATH
from torchvision.transforms import transforms
from dataset import BirdDataset,Split
from torch.utils.data import DataLoader
from models import EfficientNet_V2_M
import pytorch_lightning as pl
import torch
from pytorch_grad_cam import GradCAM, ScoreCAM,HiResCAM,GradCAMElementWise
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from image_utils import array_from_image_path,show_multiple_images
import numpy as np
from skimage.transform import resize

def main():
    torch.set_float32_matmul_precision('medium')
    transform_valid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(RESIZE_SIZE),
        transforms.ToTensor(), #0-255 -> 0-1
        transforms.Normalize(mean=(0.4742, 0.4694, 0.3954),std=(0.2394, 0.2332, 0.2547))
    ])

    test_dataset =  BirdDataset(root_dir='data',csv_file='data/birds.csv',transform=transform_valid,split=Split.TEST)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS) 
    model = EfficientNet_V2_M.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH)
    trainer = pl.Trainer()
    trainer.test(model,dataloaders=test_loader)




if __name__ == '__main__':
    main()