from image_utils import show_multiple_images
import image_utils
from models import EfficientNet_V2_S,EfficientNet_V2_L
from transformations  import old_transforms,default_transforms, default_collate_fn
from dataset import BirdDataModule, BirdDataset, Split
from matplotlib.colors import LinearSegmentedColormap

from captum.attr import IntegratedGradients,Saliency
from captum.attr import GuidedGradCam,Occlusion
from captum.attr import GradientShap
from captum.attr import Saliency
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

import torch
import numpy as np
from torchvision import transforms
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils import perturb_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

BATCH_SIZE = 4
NUM_WORKERS = 4

CHECKPOINT_PATH = r'D:\Users\david\Desktop\Python\master\statistics\524_Classes\EfficientNet_V2_L_Finetuned_V2_SGD_old\version_13\checkpoints\EfficientNet_V2_L_Pretrained_fine_tune_V2_epoch=199_validation_loss=1.0633_validation_accuracy=0.98_validation_mcc=0.97.ckpt'

def denormalize_image(tensor):
    """
    Denormalize an image tensor that was normalized with ImageNet stats
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    return tensor * std + mean

def rescale_to_01(array):
    min_val = np.min(array)
    max_val = np.max(array)
    rescaled_array = (array - min_val) / (max_val - min_val)
    return rescaled_array

def normalize_attributions_to_01(attributions: torch.Tensor) -> torch.Tensor:
    """
    Normalize attribution values to the range [0, 1].
    
    Parameters:
        attributions (torch.Tensor): The tensor of attribution values.
    
    Returns:
        torch.Tensor: A tensor with normalized values in the range [0, 1].
    """
    min_val = attributions.min()
    max_val = attributions.max()
    
    # Avoid division by zero if all attributions are the same
    if max_val == min_val:
        return torch.zeros_like(attributions) if max_val <= 0 else torch.ones_like(attributions)
    
    normalized = (attributions - min_val) / (max_val - min_val)
    return normalized

def main():
    train_transform, valid_transform,version = default_transforms()
    collate_fn = default_collate_fn()

    model = EfficientNet_V2_L.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH)
    model = model.eval()
    # datamodule = BirdDataModule(root_dir='D:/Users/david/Desktop/Python/master/data',
    #                             csv_file='D:/Users/david/Desktop/Python/master/data/birds.csv',
    #                             train_transform=train_transform,
    #                             valid_transform=valid_transform,
    #                             batch_size=BATCH_SIZE,
    #                             num_workers=NUM_WORKERS,
    #                             collate_fn=collate_fn)
    test_dataset =  BirdDataset(root_dir='D:/Users/david/Desktop/Python/master/data/',csv_file='D:/Users/david/Desktop/Python/master/data/birds.csv',transform=valid_transform,split=Split.TEST)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=2) 
    
    attribution_method = IntegratedGradients(model)
    target_layer = model.model.features[-2]
    grad_cam = GuidedGradCam(model=model,layer=target_layer)


    for index,(test_image, test_label,rgb_image) in enumerate(test_loader):
        convert_to_tensor = transforms.ToTensor()
        input = test_image.to('cuda:0')#test_image.unsqueeze(0).to('cuda:0')
        output = model(input)
        output = torch.softmax(output,dim=1)
        prediction_score,pred_label_idx = torch.topk(output,1)
        pred_label_idx.squeeze_()
        gt_label = ''#test_label.squeeze()
        gt_prob = '' #output[0,gt_label].item()
        #attributions_ig = attribution_method.attribute(input,target=pred_label_idx,sliding_window_shapes=(3,15,15),strides=(3,6,6))
        #attributions_ig = attribution_method.attribute(input,target=pred_label_idx)
        #print("Performing attribution method")
        #attributions_ig = attribution_method.attribute(input,target=pred_label_idx,n_steps=200,internal_batch_size=1)
        #attributions_gc = grad_cam.attribute(input,target=pred_label_idx)

        noise_tunnel = NoiseTunnel(attribution_method)
        attributions_ig_nt = noise_tunnel.attribute(input, nt_samples=35, nt_type='smoothgrad_sq', target=pred_label_idx,internal_batch_size=1,n_steps=200)
        print("Performing noise tunnel")
        #attributions_gc_nt = noise_tunnel.attribute(input, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
        #attributions_ig_nt = noise_tunnel.attribute(input, nt_samples=36, nt_type='smoothgrad_sq', target=pred_label_idx)
        #attributions_ig_nt = noise_tunnel.attribute(input, nt_samples=35, nt_type='smoothgrad_sq', target=pred_label_idx,sliding_window_shapes=(3,15,15),strides=(3,6,6))

        default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)
        cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
        

        # Denormalize the RGB image before visualization
        denorm_rgb_image = denormalize_image(rgb_image.squeeze())
        denorm_rgb_image = np.transpose(denorm_rgb_image.squeeze().cpu().detach().numpy(), (1,2,0))


        normalized_attribution_with_noise_tunnel = attributions_ig_nt.squeeze() #normalize_attributions_to_01(attributions_ig_nt.squeeze())
        attribution_map_with_noise_tunnel = np.transpose(normalized_attribution_with_noise_tunnel.cpu().detach().numpy(), (1,2,0))

        #normalized_attribution = attributions_ig.squeeze()
        #attribution_map = np.transpose(normalized_attribution.cpu().detach().numpy(), (1,2,0))

        isMostRelevant = True
        useSmoothing = True
        remove_percent = 45
        
        print('Performing perturbation')
        score, pertubation = perturb_image(input.squeeze(),pred_label_idx,attribution_map_with_noise_tunnel,model,remove_percent=remove_percent,isMostRelevant=isMostRelevant,use_smoothing=useSmoothing)
        arrow = '↓' if score > 0 else '↑'
        print(f'Predicted: {pred_label_idx} score: {prediction_score.squeeze().item()} Ground truth: {gt_label} score: {gt_prob}')
        print(f"Probability dropped by {score*100:.2f}%")
        xai_metric = f"{"Most relevant" if isMostRelevant else "Least relevant"} ({remove_percent}%)"
        #image_utils.show_image_from_tensor(denormalize_image(pertubation.cpu()),title=f'{xai_metric}\nClass: {pred_label_idx} ({test_label})\n Probability: {(prediction_score.squeeze().item()*100 - score*100):.2f}% ({arrow} {score*100:.2f}%)')
        image_utils.show_image_from_tensor(denormalize_image(pertubation.cpu()))
        #vis = show_cam_on_image(denorm_rgb_image, rescale_to_01(attribution_map_with_noise_tunnel), use_rgb=True,image_weight=0.5)

        #show_multiple_images(images=[denorm_rgb_image,rescale_to_01(attribution_map_with_noise_tunnel),vis],titles=['Original','Heat map', 'Combined'])


        plt_fig, plt_axis = viz.visualize_image_attr_multiple(attribution_map_with_noise_tunnel,
                             denorm_rgb_image,
                             #methods=["original_image", "heat_map",'masked_image','alpha_scaling','blended_heat_map'],
                                methods=["original_image", "heat_map",'masked_image'],
                                signs=['all', 'positive','positive'],
                             #signs=['all', 'positive','positive','positive','positive'],
                             #titles=[f'Class: {pred_label_idx} ({test_label})\n Probability: {prediction_score.squeeze().item()*100:.2f}%', 'Heatmap', 'Masked-image', 'Alpha-scaling', 'Blended heatmap'],
                             #cmap=cmap,
                             fig_size=(24,16),
                             use_pyplot=True,
                             show_colorbar=True)
        


if __name__ =='__main__':
    main()