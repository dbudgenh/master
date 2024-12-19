import torch
from typing import List, Optional, Tuple
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import importlib
from tqdm import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar
from torchvision.models import VisionTransformer,ResNet,EfficientNet

from transformations import IMAGENET_STD

MODULE_NAME = 'models'
def get_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_name = checkpoint['hyper_parameters']['name']
    del checkpoint
    class_object = get_class(MODULE_NAME,model_name)
    return class_object.load_from_checkpoint(checkpoint_path)

def get_class(module_name, class_name):
    try:
        # Dynamically import the module
        module = importlib.import_module(module_name)

        # Get the class object from the module
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        # Handle import or attribute errors
        raise ValueError(f"Error loading class '{class_name}' from module '{module_name}': {e}")

#https://github.com/pytorch/vision/blob/main/references/classification/utils.py
def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups

def find_largest_version_number(folder_path):
    max_version = -1
    # Iterate through all subdirectories in the given folder
    for subdir in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, subdir)):
            try:
                version_number = int(subdir.split('_')[1])
                max_version = max(max_version, version_number)
            except IndexError:
                pass
            except ValueError:
                pass
            
    return max_version + 1

def get_roc_curve_figure(fpr, tpr, thresholds, step_size=10, font_size=10):
    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve', marker='o', markersize=3)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, label='No skill', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')  
    plt.title(f'Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    # Annotate ROC curve with thresholds, using step_size to control density
    for i, threshold in enumerate(thresholds[::step_size]):
        plt.scatter(fpr[::step_size][i], tpr[::step_size][i], color='black', marker=".", s=10, zorder=2)
        plt.annotate(f'{threshold:.2f}', 
                    (fpr[::step_size][i], tpr[::step_size][i]), 
                    textcoords="offset points", 
                    xytext=(0, 5), 
                    ha='center',
                    fontsize=font_size)
    result = plt.gcf()
    plt.close()
    return result

def is_vision_transformer(model):
    return isinstance(model,VisionTransformer)
def is_resnet(model):
    return isinstance(model,ResNet)
def is_efficientnet(model):
    return isinstance(model,EfficientNet)

def reshape_transform(tensor, height=14, width=14):
    # Exclude the class token and reshape the tensor
    # (batch_size, seq_length, hidden_dim) -> (batch_size, height, width, hidden_dim)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Transpose the dimensions to bring channels to the first dimension
    # (batch_size, height, width, hidden_dim) -> (batch_size, hidden_dim, height, width)
    result = result.transpose(2, 3).transpose(1, 2)
    
    return result


def get_confusion_matrix_figure(computed_confusion):
    df_cm = pd.DataFrame(computed_confusion)
    plt.figure(figsize=(20,20))

    #colors from white to blue
    cmap = sns.color_palette("light:#Ff4100", as_cmap=True)

    fig = sns.heatmap(df_cm,annot=True,cmap=cmap,annot_kws={'fontsize':3}).get_figure()
    #plt.show()
    #plt.close(fig)
    return fig

def get_translation_dict(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    """
    0: ABBOTTS BABBLER
    1: ABBOTTS BOOBY
    ...
    """
    translation_dict = {}
    for line in lines:
        index, label = line.strip().split(': ')
        translation_dict[int(index)] = label
    return translation_dict

def get_k_random_values(tensor, k,device=None):
    """
    Get k random values from a PyTorch tensor along with their indices.

    Args:
    - tensor (torch.Tensor): Input tensor.
    - k (int): Number of random values to select
    - device (str): A torch device 

    Returns:
    - values (torch.Tensor): Tensor containing the selected random values.
    - indices (torch.Tensor): Tensor containing the indices of the selected values.
    """

    # Check if k is greater than the number of elements in the tensor
    if k > tensor.numel():
        raise ValueError("k cannot be greater than the number of elements in the tensor.")

    # Flatten the tensor to 1D
    flattened_tensor = tensor.view(-1)

    # Generate random indices
    random_indices = torch.randperm(flattened_tensor.numel(),device=device)[:k]

    # Use topk to get the values and indices
    values, indices = torch.topk(flattened_tensor[random_indices], k)

    return values, random_indices[indices]

import torch
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

# Define weights of the surrounding pixels
neighbors_weights = [((1, 1), 1 / 12),
                     ((0, 1), 1 / 6),
                     ((-1, 1), 1 / 12),
                     ((1, -1), 1 / 12),
                     ((0, -1), 1 / 6),
                     ((-1, -1), 1 / 12),
                     ((1, 0), 1 / 6),
                     ((-1, 0), 1 / 6)]

class NoisyLinearImputer:
    def __init__(self, noise=0.01, weighting=neighbors_weights):
        """
        Noisy linear imputation.
        Args:
            noise (float): Magnitude of noise to add (set to 0 for no noise).
            weighting (List): List of tuples (offset, weight).
        """
        self.noise = noise
        self.weighting = weighting

    @staticmethod
    def add_offset_to_indices(indices, offset, mask_shape):
        """ Add the corresponding offset to the indices. """
        cord1 = indices % mask_shape[1]
        cord0 = indices // mask_shape[1]
        cord0 += offset[0]
        cord1 += offset[1]
        valid = ~((cord0 < 0) | (cord1 < 0) | 
                  (cord0 >= mask_shape[0]) | 
                  (cord1 >= mask_shape[1]))
        return valid, indices + offset[0] * mask_shape[1] + offset[1]

    @staticmethod
    def setup_sparse_system(mask, img, neighbors_weights):
        """ Vectorized setup for sparse linear system. """
        maskflt = mask.flatten()
        imgflat = img.reshape((img.shape[0], -1))
        indices = np.argwhere(maskflt == 0).flatten()
        coords_to_vidx = np.zeros(len(maskflt), dtype=int)
        coords_to_vidx[indices] = np.arange(len(indices))
        num_equations = len(indices)
        
        # System matrix and RHS
        A = lil_matrix((num_equations, num_equations))
        b = np.zeros((num_equations, img.shape[0]))
        
        sum_neighbors = np.ones(num_equations)
        for offset, weight in neighbors_weights:
            valid, new_coords = NoisyLinearImputer.add_offset_to_indices(
                indices, offset, mask.shape)
            valid_coords = new_coords[valid]
            valid_ids = np.argwhere(valid).flatten()
            
            has_values_coords = valid_coords[maskflt[valid_coords] > 0.5]
            has_values_ids = valid_ids[maskflt[valid_coords] > 0.5]
            b[has_values_ids, :] -= weight * imgflat[:, has_values_coords].T
            
            has_no_values = valid_coords[maskflt[valid_coords] < 0.5]
            variable_ids = coords_to_vidx[has_no_values]
            has_no_values_ids = valid_ids[maskflt[valid_coords] < 0.5]
            A[has_no_values_ids, variable_ids] = weight
            
            sum_neighbors[np.argwhere(~valid).flatten()] -= weight

        A[np.arange(num_equations), np.arange(num_equations)] = -sum_neighbors
        return A, b

    def __call__(self, img, mask):
        """ Linear imputation with added noise. """
        imgflt = img.reshape(img.shape[0], -1)
        maskflt = mask.reshape(-1)
        indices_linear = np.argwhere(maskflt == 0).flatten()
        
        A, b = NoisyLinearImputer.setup_sparse_system(
            mask.cpu().numpy(), img.cpu().numpy(), self.weighting)
        res = torch.tensor(spsolve(csc_matrix(A), b), dtype=torch.float)

        img_infill = imgflt.clone()
        noise = self.noise * torch.tensor(IMAGENET_STD,device=img.device).view(-1,1) * torch.randn_like(res.t())
        img_infill[:, indices_linear] = res.t() + noise

        return img_infill.reshape_as(img)

def perturb_image(rgb_image, label, heatmap, model, remove_percent=20, isMostRelevant=True, use_smoothing=True):
    """
    Perturb image by removing a percentage of pixels using Noisy Linear Imputation.
    Args:
        rgb_image (torch.Tensor): Input image tensor (C, H, W).
        label (int): Target label.
        heatmap (np.array): Heatmap indicating pixel relevance.
        model (torch.nn.Module): Model to evaluate.
        remove_percent (float): Percentage of pixels to remove.
        isMostRelevant (bool): Remove most or least relevant pixels.
        use_smoothing (bool): Use smoothing or binary mask.
    """
    if heatmap.ndim == 3 and heatmap.shape[-1] == 3:
        heatmap = np.mean(heatmap, axis=-1)

    threshold_value = np.percentile(heatmap, 
                                    100 - remove_percent if isMostRelevant else remove_percent)
    mask = heatmap >= threshold_value if isMostRelevant else heatmap <= threshold_value
    mask_tensor = torch.from_numpy(~mask).to('cuda')
    
    model.eval()
    with torch.no_grad():
        original_output = model(rgb_image.unsqueeze(0).to('cuda'))
        original_score = torch.softmax(original_output, dim=1)[0, label].item()
        print(original_score)
    
    if use_smoothing:
        imputer = NoisyLinearImputer(noise=0.1)
        perturbed_image = imputer(rgb_image.cpu(), mask_tensor.cpu())
    else:
        perturbed_image = rgb_image.cuda() * mask_tensor

    with torch.no_grad():
        perturbed_output = model(perturbed_image.unsqueeze(0).to('cuda'))
        perturbed_score = torch.softmax(perturbed_output, dim=1)[0, label].item()
        print(perturbed_score)

    prob_difference = original_score - perturbed_score
    return prob_difference, perturbed_image

def process_in_batches(batch_size, attribution_function, input_data, **kwargs):
    """
    Processes input data in batches using the specified attribution function.

    Parameters:
        batch_size (int): The size of each batch to process.
        attribution_function (callable): The attribution function to apply (e.g., noise_tunnel.attribute).
        input_data (torch.Tensor): The input data tensor to process.
        **kwargs: Additional parameters to pass to the attribution function.

    Returns:
        torch.Tensor or numpy.ndarray: Concatenated results of the attribution function applied to all batches.
    """
    results = []

    for i in range(0, len(input_data), batch_size):
        print(f"Processing batch {i // batch_size + 1} of {len(input_data) // batch_size + 1}")
        batch = input_data[i:i + batch_size]
        batch_result = attribution_function(batch, **kwargs)
        results.append(batch_result)

    # Check if results are numpy arrays
    if isinstance(results[0], np.ndarray):
        return np.concatenate(results, axis=0)
    
    # Default to torch.cat for torch tensors
    return torch.cat(results, dim=0)

def main():
    pass

if __name__ =='__main__':
    main()