import torch
from typing import List, Optional, Tuple
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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

def get_roc_curve_figure(fpr, tpr, thresholds):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve', marker='o',markersize=3)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, label='No skill', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    # Annotate ROC curve with thresholds
    for i, threshold in enumerate(thresholds):
        plt.annotate(f'{threshold:.2f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(0, 5), ha='center',fontsize=6)
    result = plt.gcf()
    plt.close()
    return result


def get_confusion_matrix_figure(computed_confusion):
    df_cm = pd.DataFrame(computed_confusion)
    plt.figure(figsize=(40,40))

    #colors from white to blue
    cmap = sns.color_palette("light:#0012ff", as_cmap=True)

    fig = sns.heatmap(df_cm,annot=True,cmap=cmap,annot_kws={'fontsize':3}).get_figure()
    plt.show()
    plt.close(fig)
    return fig

def main():

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate example data
    data = np.random.randint(0, 100, (10, 10))  # Random data for demonstration

    # Define a custom color map (colormap) #0012ff
    cmap = sns.color_palette("light:#Ff4100", as_cmap=True)

    # Create the heatmap with the custom colormap
    ax = sns.heatmap(data, cmap=cmap)

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Custom Colormap Heatmap')

    # Show the heatmap
    plt.show()

if __name__ =='__main__':
    main()