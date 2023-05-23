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

    # transform_reverse = transforms.Compose([ transforms.Normalize(mean=[-0.4742/0.2394,-0.4694/0.2332,-0.3954/0.2547],std=[1/0.2394,1/0.2332,1/0.2547])
    #                            ])
    test_dataset =  BirdDataset(root_dir='data',csv_file='data/birds.csv',transform=transform_valid,split=Split.TEST)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS) 

    sample = next(iter(test_loader))
    input_tensor = sample['image']
    labels = sample['class_id']
    paths = sample['path']

    # rgb_images = transform_reverse(input_tensor)
    # show_tensor_as_image(transform_reverse(rgb_images[1])*255,"Test")

    #show_image_from_path(paths[1],title="Original image")


    model = EfficientNet_V2_M.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH)

    target_layers = [model.efficient_net.features[-1]]
    cam = HiResCAM(model=model.efficient_net,target_layers=target_layers,use_cuda=True)
    targets = [ClassifierOutputTarget(524)]*BATCH_SIZE
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets,aug_smooth=True,eigen_smooth=False)

    for i in range(BATCH_SIZE):
        rgb_image = resize(array_from_image_path(paths[i]),RESIZE_SIZE)
        image_cam = 1-grayscale_cam[i]
        visualization = show_cam_on_image(rgb_image, image_cam, use_rgb=True)
        print(paths[i])
        show_multiple_images(images=[rgb_image,image_cam,visualization],titles=['Original','Heat map', 'Combined'])


if __name__ == '__main__':
    main()