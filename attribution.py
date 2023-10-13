from models import EfficientNet_V2_S,EfficientNet_V2_L
from transforms import old_transforms,default_transforms, default_collate_fn
from dataset import BirdDataModule
from matplotlib.colors import LinearSegmentedColormap

from captum.attr import IntegratedGradients
from captum.attr import GuidedGradCam
from captum.attr import GradientShap
from captum.attr import Saliency
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

import torch
import numpy as np
from torchvision import transforms

BATCH_SIZE = 4
NUM_WORKERS = 4

CHECKPOINT_PATH = r'C:\Users\david\Desktop\Python\master\statistics\EfficientNet_V2_L_Finetuned_V2_SGD\version_162\checkpoints\EfficientNet_V2_L_fine_tune_V2_epoch=47_validation_loss=1.0633_validation_accuracy=0.99_validation_mcc=0.97.ckpt'

def main():
    train_transform, valid_transform,version = default_transforms()
    collate_fn = default_collate_fn()

    model = EfficientNet_V2_L.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH)
    model = model.eval()
    datamodule = BirdDataModule(root_dir='C:/Users/david/Desktop/Python/master/data',
                                csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv',
                                train_transform=train_transform,
                                valid_transform=valid_transform,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                collate_fn=collate_fn)
    
    integrated_gradients = IntegratedGradients(model)
    target_layer = model.efficient_net.features[1]
    grad_cam = GuidedGradCam(model=model,layer=target_layer)


    for test_image, test_label,rgb_image in datamodule.test_data(): 
        convert_to_tensor = transforms.ToTensor()
        input = test_image.unsqueeze(0).to('cuda:0')
        output = model(input)
        output = torch.softmax(output,dim=1)
        prediction_score,pred_label_idx = torch.topk(output,1)
        pred_label_idx.squeeze_()

        attributions_ig = integrated_gradients.attribute(input,target=pred_label_idx,n_steps=200,internal_batch_size=1)
        #attributions_gc = grad_cam.attribute(input,target=pred_label_idx)

        noise_tunnel = NoiseTunnel(integrated_gradients)
        attributions_ig_nt = noise_tunnel.attribute(input, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx,internal_batch_size=1)
        #attributions_gc_nt = noise_tunnel.attribute(input, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)

        default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)
        cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
        print(f'Predicted: {pred_label_idx} score: {prediction_score.squeeze().item()}')

        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(convert_to_tensor(rgb_image).squeeze().cpu().detach().numpy(), (1,2,0)),
                             methods=["original_image", "heat_map",'masked_image','alpha_scaling','blended_heat_map'],
                             signs=['all', 'positive','positive','positive','positive'],
                             titles=['Original','Heatmap','Masked-image','Alpha-scaling','Blended heatmap'],
                             #cmap=cmap,
                             fig_size=(12,8),
                             show_colorbar=True)


if __name__ =='__main__':
    main()