from train import BATCH_SIZE,NUM_WORKERS,CHECKPOINT_PATH
from torchvision.transforms import transforms
from dataset import BirdDataset,Split
from torch.utils.data import DataLoader
from models import EfficientNet_V2_M,EfficientNet_V2_S,EfficientNet_V2_L,VisionTransformer_L_16_Pretrained,VisionTransformer_L_16,VisionTransformer_H_14
import pytorch_lightning as pl
from pytorch_grad_cam import GradCAM,HiResCAM,GradCAMElementWise,GradCAMPlusPlus,AblationCAM,ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from image_utils import array_from_image_path,show_multiple_images
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import utils
from transformations import default_transforms,default_collate_fn
from functools import partial
from attribution import denormalize_image
from captum.attr import visualization as viz
import cv2
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.metrics.road import ROADCombined
import image_utils
from attribution import denormalize_image
import torch

USE_PRETRAINED_NETWORK = True




def main():
    train_transform, valid_transform, version = default_transforms(is_vision_transformer=True)
    collate_fn = default_collate_fn()

    test_dataset =  BirdDataset(root_dir='D:/Users/david/Desktop/Python/master/test/',csv_file='D:/Users/david/Desktop/Python/master/test/birds.csv',transform=valid_transform,split=Split.TEST)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2) 

    ckpt_path = r'D:\Users\david\Desktop\Python\master\statistics\524_Classes\EfficientNet_V2_L_V2_SGD\version_12\checkpoints\EfficientNet_V2_L_V2_epoch=599_validation_loss=1.0233_validation_accuracy=0.99_validation_mcc=0.99.ckpt'
    model = EfficientNet_V2_L.load_from_checkpoint(checkpoint_path=ckpt_path)
    model = model.eval()
    
    input_tensor,labels,rgb_images = next(iter(test_loader))
    
    #target_layers = [model.model.encoder.layers[-1].ln_1] VIT
    target_layers = [model.model.features[-1]] #EfficientNet

    reshape_transform = partial(utils.reshape_transform,height=16,width=16) #height=14, width=14 for VIT

    cam_methods = [GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, AblationCAM, ScoreCAM]
    cam_instances = [cam_method(model=model, target_layers=target_layers, reshape_transform=None) for cam_method in cam_methods]

    input_tensor, labels, rgb_images = next(iter(test_loader))
    targets = None
    targets_metric = [ClassifierOutputSoftmaxTarget(label) for label in labels]

    for cam_instance in cam_instances:
        #grayscale_cam = cam_instance(input_tensor=input_tensor, targets=targets, aug_smooth=False, eigen_smooth=False)
        print(f"Using {cam_instance.__class__.__name__}")
        grayscale_cam = utils.process_in_batches(batch_size=4,
                                                 attribution_function=cam_instance,
                                                 input_data=input_tensor,
                                                 targets=targets,
                                                 aug_smooth=False,
                                                 eigen_smooth=False,)
        cam_metric = CamMultImageConfidenceChange()

        #scores, pertubations = cam_metric(input_tensor.cuda(),grayscale_cam,targets_metric,model.model,return_visualization=True)
        
        rgb_image = denormalize_image(rgb_images[0])
        rgb_image = np.transpose(rgb_image.squeeze().cpu().detach().numpy(), (1, 2, 0))
        image_cam = grayscale_cam[0]
        #score = scores
        image_cam_3ch = np.stack([image_cam, image_cam, image_cam], axis=-1)
        visualization = show_cam_on_image(rgb_image, image_cam, use_rgb=True, image_weight=0.5, colormap=20)
        isMostRelevant = False
        remove_percent = 40
        prob = model(input_tensor.cuda()).softmax(dim=1)
        prob_gt = prob[0,labels[0]].item()
        prob_pred = prob[0,prob.argmax()].item()
        print(f"Ground truth probability: {prob_gt:.2f}, Predicted probability: {prob_pred:.2f}")
        pred_label_idx = model(input_tensor.cuda()).argmax().item()
        score,pertubation = utils.perturb_image(rgb_images[0], pred_label_idx,image_cam, model, remove_percent=remove_percent,use_smoothing=False,isMostRelevant=isMostRelevant)
        print(score)
        arrow = '↓' if score > 0 else '↑'
        xai_metric = f"{"Most relevant" if isMostRelevant else "Least relevant"} ({remove_percent}%)"
        print(f"Probability {arrow} {score*100:.2f}%")
        #pertubation = pertubations[3].cpu()
        fig = image_utils.show_image_from_tensor(denormalize_image(pertubation.cpu()))
        #image_utils.show_image_from_tensor(denormalize_image(pertubation.squeeze().cpu()))
        
        #show_multiple_images(images=[rgb_image, image_cam, visualization], titles=['Original', 'Heat map', 'Combined'])
        fig, axes = viz.visualize_image_attr_multiple(image_cam_3ch,
                             rgb_image,
                             methods=["original_image", "heat_map",'masked_image','alpha_scaling','blended_heat_map'],
                             signs=['all', 'positive','positive','positive','positive'],
                             #titles=[f'Class: {pred_label_idx} ({test_label})\n (Probability: {prediction_score.squeeze().item()*100:.2f}%', 'Heatmap', 'Masked-image', 'Alpha-scaling', 'Blended heatmap'],
                             #cmap=cmap,
                             fig_size=(16,12),
                             show_colorbar=True,
                             )

if __name__ == '__main__':
    main()