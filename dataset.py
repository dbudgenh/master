from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io

class BirdDataset(Dataset):
    """ Dataset containing images of birds
    """

    def __init__(self,root_dir,csv_file,transform=None) -> None:
        self.bird_frame = pd.read_csv(csv_file,delimiter=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.bird_frame)

    
    def __getitem__(self, idx):
        if idx % 5000 == 0:
            print(f'Loading {idx}-th example')
        class_id =  self.bird_frame.iloc[idx,0]
        img_path = os.path.join(self.root_dir,
                                self.bird_frame.iloc[idx,1])
        label = self.bird_frame.iloc[idx,2]
        dataset = self.bird_frame.iloc[idx,3]
        scientific_name = self.bird_frame.iloc[idx,4]
        image = io.imread(img_path)

        sample = {'image': image, 'class_id': class_id, 'label': label,
                  'dataset': dataset, 'scientific_name': scientific_name}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample
