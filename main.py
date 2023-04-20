from utils import show_image
from dataset import BirdDataset


def main():
    bird_dataset = BirdDataset(root_dir='data',csv_file='data/birds.csv',transform=None)
    #Load snail data
    #Finally done with programming
   
    for i in range(len(bird_dataset)):
        sample = bird_dataset[i]
        image,label,class_id = sample['image'],sample['label'],sample['class_id']
        show_image(image_array=image,title=f'Label: {label} \n Class-id: {class_id}')
        break
    

   

    
if __name__ == '__main__':
    main()