import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
import re

def contains_regex(regex,string):
    return re.search(regex, string) is not None

def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise ValueError("Attention head fusion type Not supported")

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]

    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask

def grad_rollout(attentions, gradients, discard_ratio):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):                
            weights = grad
            attention_heads_fused = (attention*weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask 

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='self_attention$', head_fusion="mean",
        discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if contains_regex(attention_layer_name, name):
                print(f"Registering hook for {name}")
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        _,weights = output
        self.attentions.append(weights.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor.cuda())

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)
    
class VITAttentionGradRollout:
    def __init__(self,model,attention_layer_name='self_attention$',discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            print(name)
            if contains_regex(attention_layer_name, name):
                print(f"Registering forward hook for {name}")
                module.register_forward_hook(
                    lambda *args, **kwargs: VITAttentionGradRollout.get_attention(self, *args, **kwargs))
            if contains_regex('ln_1', name):
                print(f"Registering backward hook for {name}")
                module.register_full_backward_hook(
                    lambda *args, **kwargs: VITAttentionGradRollout.get_attention_gradient(self, *args, **kwargs))

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        _,weights = output
        self.attentions.append(weights.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        attentions_gradient = grad_output[0].cpu()
        self.attention_gradients.append(attentions_gradient)

    def __call__(self,input_tensor,category_index):
        self.model.zero_grad()
        output = self.model(input_tensor.cuda())
        category_mask = torch.zeros(output.size(),device=output.device)
        category_mask[:,category_index] = 1
        loss = (output * category_mask).sum()
        loss.backward()

        return grad_rollout(self.attentions,self.attention_gradients,self.discard_ratio)