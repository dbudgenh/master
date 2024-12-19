import torch
import gc
from cleanlab import Datalab
from transformations import old_transforms, default_transforms, default_collate_fn
from models import EfficientNet_V2_L
from dataset import BirdDataset, Split, BirdDataModuleV2
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold

BATCH_SIZE = 16
NUM_WORKERS = 4
checkpoint_path = r"D:\Users\david\Desktop\Python\master\statistics\524_Classes\EfficientNet_V2_L_Finetuned_V2_SGD_old\version_2\checkpoints\EfficientNet_V2_L_Pretrained_fine_tune_V2_epoch=99_validation_loss=1.0096_validation_accuracy=0.99_validation_mcc=0.99.ckpt"

def process_batch(model, images, labels):
    with torch.no_grad():
        batch_output = model(images.to('cuda'))
        pred_probs = torch.softmax(batch_output, dim=1)
        
        # Move to CPU and convert to numpy immediately
        images_np = images.cpu().numpy()
        labels_np = labels.cpu().numpy()
        features_np = batch_output.cpu().numpy()
        pred_probs_np = pred_probs.cpu().numpy()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        return images_np, labels_np, features_np, pred_probs_np

def process_datalab_in_chunks(images, labels, features, pred_probs, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    # Initialize an empty dataset to accumulate results
    lab = None
    
    # Process chunks
    for _, chunk_idx in kf.split(images):
        # Convert images to the correct format (assuming NCHW format)
        chunk_images = images[chunk_idx]
        if len(chunk_images.shape) == 4:  # NCHW format
            # Convert to NHWC format if needed
            chunk_images = np.transpose(chunk_images, (0, 2, 3, 1))
        
        chunk_data = {
            'image': chunk_images,
            'label': labels[chunk_idx]
        }
        chunk_pred_probs = pred_probs[chunk_idx]
        chunk_features = features[chunk_idx]
        
        # Clear memory
        gc.collect()
        
        # Create a Datalab object for the current chunk
        chunk_lab = Datalab(
            data=chunk_data,
            label_name="label",
            image_key="image"
        )
        
        # Set pred_probs and features before finding issues
        chunk_lab.data['pred_probs'] = chunk_pred_probs
        chunk_lab.data['features'] = chunk_features
        
        # Process the current chunk
        chunk_issues = chunk_lab.find_issues(
            pred_probs=chunk_pred_probs,
            features=chunk_features
        )
        
        # Merge chunk results into the main Datalab object
        if lab is None:
            lab = chunk_lab
        else:
            lab.data['image'] = np.concatenate([lab.data['image'], chunk_images])
            lab.data['label'] = np.concatenate([lab.data['label'], chunk_data['label']])
            lab.data['pred_probs'] = np.concatenate([lab.data['pred_probs'], chunk_pred_probs])
            lab.data['features'] = np.concatenate([lab.data['features'], chunk_features])
    
    return lab

def create_datalab_instance(images, labels, features, pred_probs):
    return process_datalab_in_chunks(images, labels, features, pred_probs)

def main():
    torch.set_float32_matmul_precision('medium')
    train_transform, valid_transform, version = old_transforms()
    collate_fn = default_collate_fn()

    datamodule = BirdDataModuleV2(root_dir='D:/Users/david/Desktop/Python/master/data',
                                train_transform=train_transform,
                                valid_transform=valid_transform,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                collate_fn=collate_fn)
    datamodule.setup("fit")
    data_loader = datamodule.val_dataloader()

    model = EfficientNet_V2_L.load_from_checkpoint(checkpoint_path=checkpoint_path).to('cuda')
    model.eval()

    # Initialize numpy arrays for accumulating results
    total_samples = len(data_loader.dataset)
    first_batch = next(iter(data_loader))
    feature_size = model(first_batch[0][:1].to('cuda')).shape[1]
    num_classes = 524  # Explicitly set number of classes to 524
    
    all_images = np.zeros((total_samples, *first_batch[0][0].shape), dtype=np.float32)
    all_labels = np.zeros(total_samples, dtype=np.int64)
    all_features = np.zeros((total_samples, feature_size), dtype=np.float32)
    all_pred_probs = np.zeros((total_samples, num_classes), dtype=np.float32)  # Now using correct number of classes

    # Process batches
    start_idx = 0
    for images, labels in tqdm(data_loader):
        batch_size = images.size(0)
        images_np, labels_np, features_np, pred_probs_np = process_batch(model, images, labels)
        
        end_idx = start_idx + batch_size
        all_images[start_idx:end_idx] = images_np
        all_labels[start_idx:end_idx] = labels_np
        all_features[start_idx:end_idx] = features_np
        all_pred_probs[start_idx:end_idx] = pred_probs_np
        
        start_idx = end_idx
        gc.collect()  # Force garbage collection after each batch

    # Clear model from GPU
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Create and process Datalab instance
    lab = create_datalab_instance(all_images, all_labels, all_features, all_pred_probs)
    lab.report()

if __name__ == '__main__':
    main()