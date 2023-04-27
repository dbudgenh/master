from dataset import BirdDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os
from utils import show_tensor_image

def mean_std(data_loader):
  sample = next(iter(data_loader))
  images = sample['image']
  # shape of images = [b,c,w,h]
  mean, std = images.mean([0,2,3]), images.std([0,2,3])
  return mean, std

def batch_mean_std(data_loader):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for samples in data_loader:
      images = samples['image']
      b, c, h, w = images.shape
      nb_pixels = b * h * w
      sum_ = torch.sum(images, dim=[0, 2, 3])
      sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
      fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
      snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
      cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std

def main():
    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor()])
    bird_dataset = BirdDataset(root_dir='data',csv_file='data/birds.csv',transform=transform)
    image_data_loader = DataLoader(dataset=bird_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(batch_mean_std(image_data_loader)) #(tensor([0.4742, 0.4694, 0.3954]), tensor([0.2394, 0.2332, 0.2547]))

if __name__ == '__main__':
    main()