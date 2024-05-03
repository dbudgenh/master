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

def get_roc_curve_figure(fpr, tpr, thresholds,step_size=10,font_size=10):
    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve', marker='o',markersize=3)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, label='No skill', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')  
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for class')
    plt.legend(loc='lower right')
    # Annotate ROC curve with thresholds
    for i, threshold in enumerate(thresholds[::step_size]):
        plt.scatter(fpr[::step_size][i], tpr[::step_size][i], label='Specific Point', color='black', marker=".", s=10,zorder=2)
        plt.annotate(f'{threshold:.2f}', (fpr[::step_size][i], tpr[::step_size][i]), textcoords="offset points", xytext=(0, 5), ha='center',fontsize=font_size)
    result = plt.gcf()
    plt.close()
    return result


def get_confusion_matrix_figure(computed_confusion):
    df_cm = pd.DataFrame(computed_confusion)
    plt.figure(figsize=(20,20))

    #colors from white to blue
    cmap = sns.color_palette("light:#Ff4100", as_cmap=True)

    fig = sns.heatmap(df_cm,annot=True,cmap=cmap,annot_kws={'fontsize':3}).get_figure()
    plt.show()
    plt.close(fig)
    return fig

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
def main():
    pass

if __name__ =='__main__':
    main()