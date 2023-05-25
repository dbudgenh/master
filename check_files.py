from dataset import BirdDataset
import os.path
import shutil

FROM_PATH = 'C:/Users/david/Desktop/Python/master/data'
bird_dataset = BirdDataset(root_dir='data',csv_file='data/birds.csv',transform=None)
doesnt_exist = []
for i in range(len(bird_dataset)):
        sample = bird_dataset[i]
        path = sample['path']
        base_path = sample['base_path']
        if not os.path.exists(path):
            src = os.path.join(FROM_PATH,base_path)
            dest = path
            shutil.copyfile(src,dest)