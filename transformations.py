from torchvision.transforms import transforms,autoaugment,AugMix
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data.dataloader import default_collate
from dataset import BirdDataset


IMAGENET_MEAN =(0.485, 0.456, 0.406)
IMAGENET_STD = std=(0.229, 0.224, 0.225)

MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
RANDOM_ERASE_PROB = 0.1

INTERPOLATION = InterpolationMode.BILINEAR
VAL_RESIZE_SIZE = 232
VAL_CROP_SIZE = 224
TRAIN_CROP_SIZE = 176

IMAGENET_MEAN =(0.485, 0.456, 0.406)
IMAGENET_STD =(0.229, 0.224, 0.225)
NUM_CLASSES = 524

def standardize(x):
    return x / 255.0

def denormalize(z,mu,std):
    denorm = transforms.Compose(
        [
        transforms.Normalize(
        mean=[-m / s for m, s in zip(mu, std)],
        std=[1.0 / s for s in std],
        ),
        transforms.ConvertImageDtype(dtype=torch.uint8)
        ]
    )
    return denorm(z)



#Recommended transformations for state-of-the-art performance
def default_transforms(use_npz_dataset=False,is_vision_transformer=False):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(TRAIN_CROP_SIZE if not is_vision_transformer else 224,interpolation=INTERPOLATION,antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        autoaugment.TrivialAugmentWide(interpolation = INTERPOLATION),
        transforms.ToTensor() if not use_npz_dataset else standardize, #Npz dataset is already a tensor, so standardize from 0-255 -> 0-1
        transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD),
        transforms.RandomErasing(p=RANDOM_ERASE_PROB)
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(VAL_RESIZE_SIZE,interpolation=INTERPOLATION,antialias=True),
        transforms.CenterCrop(VAL_CROP_SIZE if not is_vision_transformer else 224) ,
        transforms.ToTensor() if not use_npz_dataset else standardize, #Npz dataset is already a tensor, so standardize from 0-255 -> 0-1
        transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
    ])
    return train_transform,valid_transform,"V2"

#Callable object to bypass pickle error
class MixupCutMixCollator:
    def __init__(self,func):
        self.func=func
    def __call__(self,batch):
        return self.func(*default_collate(batch))
    
#Apply random choice of either mixup or cutmix
def default_collate_fn():
    result = None
    mixup_transforms = []
    if MIXUP_ALPHA > 0.0:
        mixup_transforms.append(RandomMixup(NUM_CLASSES, p=1.0, alpha=MIXUP_ALPHA))
    if CUTMIX_ALPHA > 0.0:
        mixup_transforms.append(RandomCutmix(NUM_CLASSES, p=1.0, alpha=CUTMIX_ALPHA))
    if mixup_transforms:
        mixupcutmix = transforms.RandomChoice(mixup_transforms)
        # def collate_fn(batch):
        #     return mixupcutmix(*default_collate(batch))

        #Annonymous functions dont work, because they can't be pickled...
        result = MixupCutMixCollator(func=mixupcutmix)
    return result


def old_transforms():
    train_transform = transforms.Compose([
            #transforms.ToPILImage(),
            AugMix(severity=4,mixture_width=4,alpha=0.65),
            transforms.CenterCrop(BirdDataset.DEFAULT_RESIZE_SIZE),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.Resize(BirdDataset.DEFAULT_RESIZE_SIZE),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD),
            transforms.RandomErasing()
    ])
    valid_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize(BirdDataset.DEFAULT_RESIZE_SIZE),
            transforms.ToTensor(), #0-255 -> 0-1
            transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
    ])
    return train_transform, valid_transform,'V1'


import math
from typing import Tuple

import torch
from torch import Tensor
from torchvision.transforms import functional as F


class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("Please provide a valid positive value for the num_classes.")
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        _, H, W = F.get_dimensions(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s