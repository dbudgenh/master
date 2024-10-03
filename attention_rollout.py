from train import BATCH_SIZE, NUM_WORKERS, CHECKPOINT_PATH
from torchvision.transforms import transforms
from dataset import BirdDataset,Split
from torch.utils.data import DataLoader
from models import EfficientNet_V2_M,EfficientNet_V2_S,EfficientNet_V2_L,VisionTransformer_L_16_Pretrained,VisionTransformer_L_16,VisionTransformer_H_14
import pytorch_lightning as pl
from pytorch_grad_cam import GradCAM,ScoreCAM,HiResCAM,GradCAMElementWise
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from image_utils import array_from_image_path,show_multiple_images
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import utils
from transformations import default_transforms,default_collate_fn
from functools import partial
from vit_rollout import VITAttentionRollout,VITAttentionGradRollout
import cv2
import torch.nn.functional as F

USE_PRETRAINED_NETWORK = True
BATCH_SIZE = 8

def show_mask_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask),
                                cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main():
    train_transform, valid_transform, version = default_transforms(is_vision_transformer=True)
    collate_fn = default_collate_fn()
    test_dataset =  BirdDataset(root_dir='C:/Users/david/Desktop/Python/master/data/',
                                csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',
                                transform=valid_transform,split=Split.TEST)
    
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True, 
                            num_workers=2)
    #ckpt_path = r'C:\Users\david\Desktop\Python\master\lightning_logs\version_36\checkpoints\VisionTransformer_H_14_Pretrained_fine_tune_V2_epoch=52_validation_loss=1.0414_validation_accuracy=0.988_validation_mcc=0.977.ckpt'
    ckpt_path = r'C:\Users\david\Desktop\Python\master\lightning_logs\version_23\checkpoints\VisionTransformer_L_16_Pretrained_fine_tune_V2_epoch=94_validation_loss=1.0670_validation_accuracy=0.982_validation_mcc=0.976.ckpt'
    
    model = VisionTransformer_L_16.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()

    input_tensor,labels,paths = next(iter(test_loader))

    attention_rollout = VITAttentionRollout(model=model.model,
                                            attention_layer_name='self_attention$',
                                            discard_ratio=0.0,
                                            head_fusion='mean')
    # attention_rollout = VITAttentionGradRollout(model=model.model,
    #                                             attention_layer_name='self_attention$',
    #                                             discard_ratio=0.0)
    #mask = attention_rollout(input_tensor)

    for i in range(BATCH_SIZE):
        mask = attention_rollout(input_tensor[i].unsqueeze(0))
        #mask = attention_rollout(input_tensor[i].unsqueeze(0),0)
        mask = cv2.resize(mask, (224, 224))

  
        rgb_image = resize(array_from_image_path(paths[i]),(224,224))
        #mask = cv2.resize(mask, (224, 224))
        image_cam = mask

        visualization = show_cam_on_image(rgb_image, mask, use_rgb=True,image_weight=0.5,colormap=2)
        print(paths[i])
        show_multiple_images(images=[rgb_image,image_cam,visualization],titles=['Original','Heat map', 'Combined'])



if __name__ == '__main__':
    main()