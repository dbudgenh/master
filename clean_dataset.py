import torch
from cleanlab import Datalab
from transformations import old_transforms,default_transforms,default_collate_fn
from models import EfficientNet_V2_L
from dataset import BirdDataset,Split
from tqdm import tqdm
from skorch import NeuralNetClassifier
from cleanlab.filter import find_label_issues
from dataset import BirdDataModuleV2


BATCH_SIZE = 16
checkpoint_path = r"C:\Users\david\Desktop\Python\master\statistics\524_Classes\EfficientNet_V2_L_Finetuned_V2_SGD\version_2\checkpoints\EfficientNet_V2_L_Pretrained_fine_tune_V2_epoch=99_validation_loss=1.0096_validation_accuracy=0.99_validation_mcc=0.99.ckpt"
def main():
    torch.set_float32_matmul_precision('medium')
    train_transform, valid_transform,version = old_transforms()
    collate_fn = default_collate_fn()

    datamodule = BirdDataModuleV2(root_dir='C:/Users/david/Desktop/Python/master/data',
                                train_transform=train_transform,
                                valid_transform=valid_transform,
                                batch_size=BATCH_SIZE,
                                num_workers=8,
                                collate_fn=collate_fn)
    datamodule.setup("test")
    data_loader = datamodule.test_dataloader()

    model = EfficientNet_V2_L.load_from_checkpoint(checkpoint_path=checkpoint_path).to('cuda')
    model.eval()

    all_features = []
    all_labels = []
    all_outputs = []
    all_predictions = []

    dataset = {
        'image':[],
        'label':[]
    }
    with torch.no_grad():
        for batch_data in tqdm(data_loader):
            images,labels = batch_data

            dataset['image'].append(images)
            dataset['label'].append(labels)

            batch_output = model(images.to('cuda'))

            all_features.append(batch_output)
            all_labels.append(labels)
            all_outputs.append(torch.softmax(batch_output,dim=1))
            all_predictions.append(torch.argmax(batch_output,dim=1))

    dataset['image'] = torch.cat(dataset['image'],dim=0).cpu().numpy()
    dataset['label'] = torch.cat(dataset['label'],dim=0).cpu().numpy()
    all_features = torch.cat(all_features,dim=0)
    pred_probs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels,dim=0)
    all_predictions = torch.cat(all_predictions,dim=0)

    unequal_indices = torch.nonzero(all_labels != all_predictions.cpu(),as_tuple=False)

    lab = Datalab(data=dataset, label_name="label", image_key="image")
    lab.find_issues(pred_probs = pred_probs.cpu().numpy(),features=all_features.cpu().numpy())
    lab.report()



    # ranked_label_issues = find_label_issues(all_labels.numpy(),pred_probs.cpu().numpy(),return_indices_ranked_by="self_confidence",)

    # print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
    # print(f"Top 15 most likely label errors: \n {ranked_label_issues[:15]}")
    # print()

if __name__ == '__main__':
    main()