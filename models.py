import math
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch 
import torch.nn.functional as F
from torch.nn.modules import Module
from torchvision.models import efficientnet_b0,efficientnet_v2_s,efficientnet_v2_m,efficientnet_v2_l,vit_l_16,vit_h_14,alexnet,resnet50,EfficientNet_V2_S_Weights,EfficientNet_V2_M_Weights,EfficientNet_B0_Weights,EfficientNet_V2_L_Weights,ViT_L_16_Weights
from torchvision.models import resnet18,resnet101
from torchvision.models import vit_b_16,ViT_B_16_Weights,ViT_H_14_Weights
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import Accuracy,F1Score,MatthewsCorrCoef,ConfusionMatrix,ROC,AUROC,Precision,Recall
from lion_pytorch import Lion
from abc import ABC,abstractmethod
from torch.optim import AdamW,SGD
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,ExponentialLR,ReduceLROnPlateau,StepLR,LinearLR,SequentialLR
import image_utils
from knowledge_distillation import knowledge_distillation_loss
import utils
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from sklearn.metrics import classification_report
from tqdm import tqdm
from functools import partial
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import IntegratedGradients,GuidedGradCam,GradientShap,Saliency,NoiseTunnel,Occlusion
from captum.attr import visualization as viz
from pytorch_grad_cam import GradCAM, ScoreCAM, HiResCAM, EigenCAM, AblationCAM, GradCAMPlusPlus, GradCAMElementWise
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget,ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.metrics.road import ROADCombined
import numpy as np
from transformations import denormalize
from torchvision.transforms import transforms
from transformations import IMAGENET_MEAN,IMAGENET_STD
from image_utils import create_multiple_images
import json

NUM_CLASSES = 524
TOP_K = 50


class ImageClassifierBase(ABC,pl.LightningModule):
    def __init__(self,
                 base_model:Module,
                 lr=0.1,
                 batch_size=32,
                 epochs=150,
                 momentum=0.9,
                 weight_decay=2e-5,
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 optimizer_algorithm='sgd',
                 num_workers=0,
                 log_config= None,
                 ):
        super().__init__()
        self.model = base_model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.lr_scheduler = lr_scheduler.lower()
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_warmup_method = lr_warmup_method.lower()
        self.optimizer_algorithm = optimizer_algorithm.lower()
        self.num_workers = num_workers
        self.epochs = epochs
        self.lr_warmup_decay = lr_warmup_decay
        self.norm_weight_decay = norm_weight_decay
        self.name = self.__class__.__name__
        self.log_config = log_config
        

        self.accuracy_top_1 = Accuracy(task="multiclass", num_classes=NUM_CLASSES,top_k=1)
        self.accuracy_top_5 = Accuracy(task="multiclass", num_classes=NUM_CLASSES,top_k=5)
        self.mcc = MatthewsCorrCoef(task="multiclass",num_classes=NUM_CLASSES)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.confusion_matrix = ConfusionMatrix(task="multiclass",num_classes=NUM_CLASSES)
        self.roc_curve = ROC(task='multiclass',num_classes=NUM_CLASSES)
        self.auroc = AUROC(task='multiclass',num_classes=NUM_CLASSES)
        self.recall = Recall(task='multiclass',num_classes=NUM_CLASSES,average='macro')
        self.precision = Precision(task='multiclass',num_classes=NUM_CLASSES,average='macro')
        self.f1 = F1Score(task='multiclass',num_classes=NUM_CLASSES,average='macro')
        
        #self.logger:TensorBoardLogger = TensorBoardLogger(save_dir=':/',log_graph=True)
        self.test_step_prediction = []
        self.test_step_label = []
        self.test_step_input = []


    
        if not self.log_config:
            self.log_config = {
            'confusion_matrix':True,
            'roc_curve':True,
            'auroc':True,
            'classification_report':True,
            'pytorch_cam':True,
            'captum_alg':True,
            'topk':True,
            'bottomk':True,
            'randomk':True,
            'all':False,
            'filter_type':None
            }

        self.save_hyperparameters({
            'lr':self.lr,
            'momentum':self.momentum,
            'weight_decay':self.weight_decay,
            'batch_size':self.batch_size,
            'label_smoothing':self.label_smoothing,
            'lr_scheduler':self.lr_scheduler,
            'lr_warmup_epochs':self.lr_warmup_epochs,
            'lr_warmup_method':self.lr_warmup_method,
            'optimizer_algorithm':self.optimizer_algorithm,
            'num_workers':self.num_workers,
            'epochs':self.epochs,
            'lr_warmup_decay':self.lr_warmup_decay,
            'norm_weight_decay':self.norm_weight_decay,
            'name':self.name
            })
        

    def forward(self,x):
        out = self.model(x)
        return out
    
    def freeze_layers(self):
        print('Freezing layers')
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_layers(self):
        print('Unfreezing layers')
        for param in self.model.parameters():
            param.requires_grad = True

    def log_text_to_tensorboard(self,tag,text_string,global_step=None,walltime=None):
        self.logger.experiment.add_text(tag,text_string,global_step,walltime)

    def _print_parameters(self):
        print(f''' Model was configured with the following parameters:
        lr={self.lr}
        momentum={self.momentum}
        lr_scheduler={self.lr_scheduler}
        lr_warmup_epochs={self.lr_warmup_epochs}
        lr_warmup_method={self.lr_warmup_method}
        lr_warmup_decay={self.lr_warmup_decay}
        weight_decay={self.weight_decay}
        norm_weight_decay={self.norm_weight_decay}
        label_smoothing={self.label_smoothing}
        batch_size={self.batch_size}
        epochs={self.epochs}
        optimizer={self.optimizer_algorithm}
        num_workers={self.num_workers}
        name={self.name}
        ''')

    def setup(self, stage: str) -> None:
        self.logger._log_graph = True
        print("Print computation graph")
        #self.logger.log_graph(model=self,input_array=torch.rand((1,3,224,224)).to('cuda'))


    def configure_optimizers(self) -> Any:
        parameters = utils.set_weight_decay(
            self,
            self.weight_decay,
            self.norm_weight_decay,
            None
        )
        if self.optimizer_algorithm == 'sgd':
            optimizer = SGD(parameters,lr=self.lr,momentum=self.momentum,weight_decay=self.weight_decay)
        elif self.optimizer_algorithm == 'adam':
            optimizer = AdamW(parameters,lr=self.lr,weight_decay=self.weight_decay)
        else:
            raise RuntimeError(
                f"Invalid optimizer algorithm '{self.lr_scheduler}'. Only sgd or adam is supported."
            )
        if self.lr_scheduler == 'cosineannealinglr':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs - self.lr_warmup_epochs)
        elif self.lr_scheduler == 'cosineannealingwarmrestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.epochs - self.lr_warmup_epochs,T_mult=1)
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{self.lr_scheduler}'. Only cosineannealinglr or cosineannealingwarmrestarts is supported."
            )
        if self.lr_warmup_epochs > 0:
            if self.lr_warmup_method == 'linear':
                warmup_lr_scheduler = LinearLR(optimizer, start_factor=self.lr_warmup_decay, total_iters=self.lr_warmup_epochs)
            else:
                raise RuntimeError(
                f"Invalid lr warmup method '{self.lr_warmup_method}'. Only linear is supported."
            )
            lr_scheduler = SequentialLR(optimizer,schedulers=[warmup_lr_scheduler,scheduler],milestones=[self.lr_warmup_epochs])
        else :
            lr_scheduler = scheduler
        return [optimizer],[{"scheduler": lr_scheduler,'interval':'epoch'}] 
    
    def training_step(self, batch,batch_idx):
        inputs, labels = batch
        output = self(inputs)

        #one-hot encoded (because of cutmix & mixup), convert to class label
        if labels.size(dim=-1) == NUM_CLASSES:
            labels = torch.argmax(labels,dim=1)

        loss = self.criterion(output,labels)
        accuracy_top_1 = self.accuracy_top_1(output,labels)
        accuracy_top_5 = self.accuracy_top_5(output,labels)
        mcc = self.mcc(output,labels)

        self.logger.experiment.add_scalars('loss',{'train':loss},global_step=self.global_step)
        self.logger.experiment.add_scalars('accuracy_top_1',{'train':accuracy_top_1},global_step=self.global_step)
        self.logger.experiment.add_scalars('accuracy_top_5',{'train':accuracy_top_5},global_step=self.global_step)
        self.logger.experiment.add_scalars('mcc',{'train':mcc},global_step=self.global_step)

        self.log("train_accuracy",accuracy_top_1,on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("train_accuracy_top_5",accuracy_top_5,on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("train_loss",loss,on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("train_mcc",mcc,on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        return loss
    

    def validation_step(self,batch,batch_idx):
        inputs, labels = batch
        output = self(inputs)
        loss = self.criterion(output,labels)
        accuracy_top_1 = self.accuracy_top_1(output,labels)
        accuracy_top_5 = self.accuracy_top_5(output,labels)
        mcc = self.mcc(output,labels)


        self.logger.experiment.add_scalars('loss',{'validation':loss},global_step=self.global_step)
        self.logger.experiment.add_scalars('accuracy_top_1',{'validation':accuracy_top_1},global_step=self.global_step)
        self.logger.experiment.add_scalars('accuracy_top_5',{'validation':accuracy_top_5},global_step=self.global_step)
        self.logger.experiment.add_scalars('mcc',{'validation':mcc},global_step=self.global_step)


        self.log("validation_accuracy",accuracy_top_1,on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("validation_accuracy_top_5",accuracy_top_5,on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("validation_loss",loss,on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("validation_mcc",mcc,on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        
        return loss

    def test_step(self,batch,batch_idx):
        inputs, labels = batch
        output = self(inputs)
        loss = self.criterion(output,labels)
        accuracy_top_1 = self.accuracy_top_1(output,labels)
        accuracy_top_5 = self.accuracy_top_5(output,labels)
        mcc = self.mcc(output,labels)
        f1 = self.f1(output,labels)
        recall = self.recall(output,labels)
        precision = self.precision(output,labels)

        #predictions = torch.argmax(output,dim=1)
        self.log("test_accuracy",accuracy_top_1,prog_bar=True,batch_size=self.batch_size)
        self.log("test_accuracy_top_5",accuracy_top_5,prog_bar=True,batch_size=self.batch_size)
        self.log("test_loss",loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_mcc",mcc,prog_bar=True,batch_size=self.batch_size)
        self.log("test_f1",f1,prog_bar=True,batch_size=self.batch_size)
        self.log("test_recall",recall,prog_bar=True,batch_size=self.batch_size)
        self.log("test_precision",precision,prog_bar=True,batch_size=self.batch_size)

        if self.log_config['testing']:
            #only add misclassified samples
            predictions = output.argmax(dim=1)  # Get the index of the max logit (class prediction)
            misclassified_mask = predictions != labels

            misclassified_input = inputs[misclassified_mask]
            misclassified_labels = labels[misclassified_mask]
            misclassified_prediction= output[misclassified_mask]

            self.test_step_prediction.append(misclassified_prediction)
            self.test_step_label.append(misclassified_labels)
            self.test_step_input.append(misclassified_input)
        else:
            self.test_step_prediction.append(output)
            self.test_step_label.append(labels)
            self.test_step_input.append(inputs)
        return loss
    

    def on_test_epoch_end(self) -> None:
        all_predictions = torch.cat(self.test_step_prediction) #(2625,NUM_CLASSES)
        all_labels = torch.cat(self.test_step_label) # (2625)
        all_images = torch.cat(self.test_step_input) # (2625,3,224,224)
        

        all_predictions_probabilities = torch.softmax(all_predictions,dim=1) #(2625,NUM_CLASSES)
        all_predictions_max_probabilities_values,all_predictions_max_probabilities_indices = torch.max(all_predictions_probabilities,dim=1) #(2625)
        all_predictions_idx = torch.argmax(all_predictions_probabilities,dim=1) #(2625)

        # Determine which filter to apply based on the configuration
        filter_type = self.log_config.get('filter_type', None)  # can be 'misclassified', 'correct', or None
        if filter_type == 'misclassified':
            filter_mask = all_predictions_idx != all_labels
        elif filter_type == 'correct':
            filter_mask = all_predictions_idx == all_labels
        else:
            # No filtering
            filter_mask = None

        # Apply filtering if a mask is set
        if filter_mask is not None:
            all_predictions = all_predictions[filter_mask]
            all_labels = all_labels[filter_mask]
            all_images = all_images[filter_mask]
            all_predictions_probabilities = all_predictions_probabilities[filter_mask]
            all_predictions_max_probabilities_values = all_predictions_max_probabilities_values[filter_mask]
            all_predictions_max_probabilities_indices = all_predictions_max_probabilities_indices[filter_mask]
            all_predictions_idx = all_predictions_idx[filter_mask]
        print(f'There are {len(all_predictions)} samples to analyze')



        top_k = partial(torch.topk,k = TOP_K)
        bottom_k = partial(torch.topk, k=TOP_K, largest=False)
        rand_k = partial(utils.get_k_random_values,k=TOP_K,device="cuda")
        all = partial(torch.topk,k=len(all_predictions))

        selection_functions = []
        selection_functions += [(top_k,"Top")] if self.log_config['topk'] else []
        selection_functions += [(bottom_k,"Bottom")] if self.log_config['bottomk'] else []
        selection_functions += [(rand_k,"Random")] if self.log_config['randomk'] else []
        selection_functions += [(all,"All")] if self.log_config['all'] else []

        top_k_propabilities = torch.topk(all_predictions_max_probabilities_values,10) #(k)
        bottom_k_propabilities = torch.topk(all_predictions_max_probabilities_values,10,largest=False) #(k)
        translation_dict = utils.get_translation_dict('class_mapping.txt')
        xai_results = []
        
        
        if utils.is_vision_transformer(self.model):
            target_layers = [self.model.encoder.layers[-1].ln_1]
            reshape_transform = utils.reshape_transform
        elif utils.is_resnet(self.model):
            target_layers = [self.model.layer4[-1]]
            reshape_transform = None
        elif utils.is_efficientnet(self.model):
            target_layers = [self.model.features[-1]]
            reshape_transform = None
        else:
            #target_layers = [self.model.layer4[-1]]
            target_layers = None
            reshape_transform = None

        if True:
            if self.log_config['pytorch_cam']:
                pytorch_gradcam_cams = [
                    GradCAM(model=self.model,target_layers=target_layers,reshape_transform=reshape_transform),
                    HiResCAM(model=self.model,target_layers=target_layers,reshape_transform=reshape_transform),
                    #AblationCAM(model=self.model,target_layers=target_layers,reshape_transform=reshape_transform),
                    GradCAMPlusPlus(model=self.model,target_layers=target_layers,reshape_transform=reshape_transform),
                    GradCAMElementWise(model=self.model,target_layers=target_layers,reshape_transform=reshape_transform)
                ]
            if self.log_config['captum_alg']:
                #Captum init cams
                captum_alg = [
                    IntegratedGradients(self.model),
                    #GuidedGradCam(model=self.model,layer=target_layers[0]),
                    Saliency(self.model),
                    Occlusion(self.model),
                ]

            #Create the confusion matrix
            if self.log_config['confusion_matrix']:
                print('Computing confusion matrix...')
                computed_confusion = self.confusion_matrix(all_predictions,all_labels)

                #fig = utils.get_confusion_matrix_figure(computed_confusion=computed_confusion.cpu().numpy().astype(int))
                #elf.logger.experiment.add_figure('Confusion matrix',fig,self.current_epoch)
                cm_normalized = computed_confusion.float() / computed_confusion.sum(1).view(-1, 1)
                misclassification_distribution = {}

                # Collect misclassification details
                for i in range(NUM_CLASSES):
                    misclassified_as = {
                        j: {
                            "count": computed_confusion[i, j].item(),
                            "percentage": cm_normalized[i, j].item() * 100
                        }
                        for j in range(NUM_CLASSES)
                        if i != j and computed_confusion[i, j] > 0
                    }
                    
                    total_misclassified = sum(computed_confusion[i, j].item() for j in range(NUM_CLASSES) if i != j)
                    misclassification_distribution[i] = {
                        "total_misclassified": total_misclassified,
                        "misclassified_as": misclassified_as
                    }

                # Sort the classes by total misclassifications in descending order
                sorted_misclassification_distribution = sorted(
                    misclassification_distribution.items(),
                    key=lambda x: x[1]['total_misclassified'],
                    reverse=True
                )

                # Format the misclassification distribution into a readable string
                distribution_text = "Misclassification Distribution (Sorted by Total Misclassified):\n"
                for class_id, data in sorted_misclassification_distribution:
                    distribution_text += f"Class {class_id}:\n"
                    distribution_text += f"  Total Misclassified: {data['total_misclassified']}\n"
                    if data["misclassified_as"]:
                        distribution_text += "  Misclassified As:\n"
                        for misclass_id, details in data["misclassified_as"].items():
                            distribution_text += (
                                f"    Class {misclass_id}: "
                                f"{details['count']} samples ({details['percentage']:.2f}%)\n"
                            )
                    else:
                        distribution_text += "  No Misclassifications\n"
                    distribution_text += "\n"

                # Log the readable text to TensorBoard
                self.logger.experiment.add_text('Misclassification Distribution', distribution_text)

            if self.log_config['roc_curve']:
                #Create ROC-Curve
                fpr, tpr, thresholds = self.roc_curve(all_predictions,all_labels)
                #For each class, create a seperate roc-curve
                for i in tqdm(range(NUM_CLASSES),desc="ROC curve"):
                    fig = utils.get_roc_curve_figure(fpr=fpr[i].cpu().numpy(),tpr=tpr[i].cpu().numpy(),thresholds=thresholds[i].cpu().numpy())
                    self.logger.experiment.add_figure(f'ROC curve for class {i}',fig,self.current_epoch)

            if self.log_config['auroc']:
                #Create AUROC
                print('Computing Area under ROC')
                auroc = self.auroc(all_predictions,all_labels)
                self.log('Area under ROC',auroc,batch_size=self.batch_size)

            #Create classification report
            if self.log_config['classification_report']:
                print('Computing classification report')
                report = classification_report(all_labels.cpu(),torch.argmax(all_predictions,dim=1).cpu())
                self.logger.experiment.add_text('Classification report',report)

            if self.log_config['topk'] or self.log_config['bottomk'] or self.log_config['randomk'] or self.log_config['all']:
                for function, suffix in selection_functions:
                    best_k_probabilities = function(all_predictions_max_probabilities_values)
                    best_propabilities = best_k_probabilities[0]
                    best_idx = best_k_probabilities[1]
                    best_images = all_images[best_idx]
                    best_labels = all_labels[best_idx]
                    
                    #get prediction
                    best_predictions = all_predictions_idx[best_idx]
                    #save/log softmax vector for each idx in tensorboard
                    best_softmax_idx = all_predictions_probabilities[best_idx]
                    #get ground truth probability for each of the top_k softmaxvectors
                    best_gt_prob = best_softmax_idx[torch.arange(best_softmax_idx.size(0)),best_labels]

                    targets_metric = [ClassifierOutputSoftmaxTarget(class_id) for class_id in best_predictions]
                    targets_cam = [ClassifierOutputTarget(class_id) for class_id in best_predictions]

                    resize = transforms.Resize(224)
                    rgb_images = resize(best_images)
                    denormalized_images = denormalize(rgb_images,IMAGENET_MEAN,IMAGENET_STD)
                                                            
                    #pytorch-cam
                    if self.log_config['pytorch_cam']:                                    
                        #apply pytorch_gradcam_cam
                        for cam in pytorch_gradcam_cams:
                            print(f"Computing {cam.__class__.__name__}")
                            with torch.enable_grad():
                                #cam_images = cam(input_tensor=best_images, targets=targets_cam,aug_smooth=False,eigen_smooth=False)
                                cam_images = utils.process_in_batches(batch_size=8,
                                                                    attribution_function=cam,
                                                                    input_data=best_images,
                                                                    targets=targets_cam,
                                                                    aug_smooth=False,
                                                                    eigen_smooth=False,
                                                                    )
                            cam_name = str(type(cam)).split(".")[-1][:-2]
                            for idx, cam_image in enumerate(cam_images):
                                permuted_image = denormalized_images[idx].permute(1,2,0).cpu().numpy().astype(np.float32)/255
                                cam_image_3ch = np.stack([cam_image, cam_image, cam_image], axis=-1)
                                
                                for isMostRelevant in [True,False]:
                                    for use_smoothing in [True,False]:
                                        xai_metric = 'ROAD' if use_smoothing else 'Pixel-Flip'
                                        xai_metric += ' (MoRF)' if isMostRelevant else ' (LeRF)'
                                        print(f"Computing pertubation for {xai_metric}")
                                        score, pertubation = utils.perturb_image(best_images[idx], best_predictions[idx], cam_image, self.model, remove_percent=45,isMostRelevant=isMostRelevant,use_smoothing=use_smoothing)
                                        arrow = '↓' if score > 0 else '↑'
                                        print(f"{xai_metric}: Predicted: {best_predictions[idx]} Probability prediction {best_propabilities[idx]*100:.2f}%({arrow} {abs(score)*100:.2f}%) Groundtruth: {best_labels[idx]} Probability groundtruth: {best_gt_prob[idx]*100:.2f}%")
                                        fig = image_utils.show_image_from_tensor(denormalize(pertubation.cpu(),IMAGENET_MEAN,IMAGENET_STD),show=False)
                                        gt_label_name = translation_dict[best_labels[idx].item()]
                                        pred_label_name = translation_dict[best_predictions[idx].item()]

                                        self.logger.experiment.add_figure(f'{suffix}_{TOP_K}/{cam_name} {xai_metric}: *** Groundtruth label: {best_labels[idx]}({gt_label_name}) Probability: {best_gt_prob[idx]*100:.2f} *** Predicted label: {best_predictions[idx]}({pred_label_name}) Probability: {best_propabilities[idx]*100:.2f} ({arrow} {abs(score)*100:.2f}%) *** Idx: {best_idx[idx]}',fig)
                                        xai_results.append({
                                            'idx':best_idx[idx].item() if hasattr(best_idx[idx], 'item') else best_idx[idx],
                                            'cam':cam_name,
                                            'xai_metric':xai_metric,
                                            'pertubation_value':score,
                                            'gt_label_name':gt_label_name,
                                            'pred_label_name':pred_label_name,
                                            'gt_prob':best_gt_prob[idx].item() if hasattr(best_gt_prob[idx], 'item') else best_gt_prob[idx],
                                            'pred_prob':best_propabilities[idx].item() if hasattr(best_propabilities[idx], 'item') else best_propabilities[idx],
                                        })

                                fig, axes = viz.visualize_image_attr_multiple(cam_image_3ch,
                                    permuted_image,
                                    methods=["original_image", "heat_map",'masked_image','alpha_scaling','blended_heat_map'],
                                    signs=['all', 'positive','positive','positive','positive'],
                                    #titles=['Original','Heatmap','Masked-image','Alpha-scaling','Blended heatmap'],
                                    #cmap=cmap,
                                    fig_size=(16,12),
                                    use_pyplot=False,
                                    show_colorbar=True,
                                )
                                self.logger.experiment.add_figure(f'{suffix}_{TOP_K}/{cam_name} *** Groundtruth label: {best_labels[idx]}({gt_label_name}) Probability: {best_gt_prob[idx]*100:.2f} *** Predicted label: {best_predictions[idx]}({pred_label_name}) Probability: {best_propabilities[idx]*100:.2f} *** Idx: {best_idx[idx]}',fig)
                    
                    #captum-alg
                    if self.log_config['captum_alg']:  
                        for alg in captum_alg:
                            prediction_score,pred_label_idx = torch.topk(best_predictions,1)
                            cam_name = str(type(alg)).split(".")[-1][:-2]
                            print(f"Computing {cam_name}")
                            #attributions = alg.attribute(best_images,target=pred_label_idx) #brauchen wir variable?
                            noise_tunnel = NoiseTunnel(alg)
                            if cam_name == "IntegratedGradients":
                                #attributions_nt = noise_tunnel.attribute(best_images, nt_samples=5,nt_samples_batch_size=2, nt_type='smoothgrad_sq', target=pred_label_idx,internal_batch_size=1,n_steps=50)
                                attributions_nt = utils.process_in_batches(batch_size=16,
                                                                    attribution_function=noise_tunnel.attribute,
                                                                    input_data=best_images,
                                                                    nt_samples=25,
                                                                    nt_samples_batch_size=1,
                                                                    nt_type='smoothgrad_sq',
                                                                    target=pred_label_idx,
                                                                    internal_batch_size=1,
                                                                    n_steps=200)
                            elif cam_name == "Occlusion":
                               # attributions_nt = noise_tunnel.attribute(best_images, nt_samples=5,nt_samples_batch_size=2, nt_type='smoothgrad_sq', target=pred_label_idx,sliding_window_shapes=(3,15,15),strides=(3,6,6))
                                attributions_nt = utils.process_in_batches(batch_size=16,
                                                                    attribution_function=noise_tunnel.attribute,
                                                                    input_data=best_images,
                                                                    nt_samples=25,
                                                                    nt_samples_batch_size=1,
                                                                    nt_type='smoothgrad_sq',
                                                                    target=pred_label_idx,
                                                                    sliding_window_shapes=(3,15,15),
                                                                    strides=(3,6,6),
                                                                    )
                            else:
                                #attributions_nt = noise_tunnel.attribute(best_images,nt_samples=5,nt_samples_batch_size=2, nt_type='smoothgrad_sq', target=pred_label_idx)
                                attributions_nt = utils.process_in_batches(batch_size=16,
                                                                    attribution_function=noise_tunnel.attribute,
                                                                    input_data=best_images,
                                                                    nt_samples=25,
                                                                    nt_samples_batch_size=1,
                                                                    nt_type='smoothgrad_sq',
                                                                    target=pred_label_idx,
                                                                    )
                                
                            for idx, alg_image in enumerate(attributions_nt):
                                transposed_alg_img = np.transpose(alg_image.cpu().detach().numpy(), (1,2,0))
                                permuted_image = denormalized_images[idx].permute(1,2,0).cpu().numpy().astype(np.float32)/255
                                for isMostRelevant in [True,False]:
                                    for use_smoothing in [True,False]:
                                        xai_metric = 'ROAD' if use_smoothing else 'Pixel-Flip'
                                        xai_metric += ' (MoRF)' if isMostRelevant else ' (LeRF)'
                                        print(f"Computing pertubation for {xai_metric}")
                                        score, pertubation = utils.perturb_image(best_images[idx], best_predictions[idx], transposed_alg_img, self.model, remove_percent=45,isMostRelevant=isMostRelevant,use_smoothing=use_smoothing)
                                        arrow = '↓' if score > 0 else '↑'
                                        print(f"Predicted: {best_predictions[idx]} Probability prediction: {best_propabilities[idx]*100:.2f}%({arrow} {abs(score)*100:.2f}%) Groundtruth: {best_labels[idx]} Probability groundtruth: {best_gt_prob[idx]*100:.2f}%")
                                        fig = image_utils.show_image_from_tensor(denormalize(pertubation.cpu(),IMAGENET_MEAN,IMAGENET_STD),show=False)
                                        gt_label_name = translation_dict[best_labels[idx].item()]
                                        pred_label_name = translation_dict[best_predictions[idx].item()]

                                        self.logger.experiment.add_figure(f'{suffix}_{TOP_K}/Captum_{cam_name} {xai_metric}: *** Groundtruth label: {best_labels[idx]}({gt_label_name}) Probability: {best_gt_prob[idx]*100:.2f} *** Predicted label: {best_predictions[idx]}({pred_label_name}) Probability: {best_propabilities[idx]*100:.2f} ({arrow} {abs(score)*100:.2f}%) *** Idx: {best_idx[idx]}',fig)
                                        xai_results.append({
                                            'idx':best_idx[idx].item() if hasattr(best_idx[idx], 'item') else best_idx[idx],
                                            'cam':cam_name,
                                            'xai_metric':xai_metric,
                                            'pertubation_value':score,
                                            'gt_label_name':gt_label_name,
                                            'pred_label_name':pred_label_name,
                                            'gt_prob':best_gt_prob[idx].item() if hasattr(best_gt_prob[idx], 'item') else best_gt_prob[idx],
                                            'pred_prob':best_propabilities[idx].item() if hasattr(best_propabilities[idx], 'item') else best_propabilities[idx],
                                        })
                                fig,axes = viz.visualize_image_attr_multiple(transposed_alg_img,
                                                    permuted_image,
                                                    methods=["original_image", "heat_map",'masked_image','alpha_scaling','blended_heat_map'],
                                                    signs=['all', 'positive','positive','positive','positive'],
                                                    #titles=['Original','Heatmap','Masked-image','Alpha-scaling','Blended heatmap'],
                                                    fig_size=(16,12),
                                                    use_pyplot=False,
                                                    show_colorbar=True)
                                

                                self.logger.experiment.add_figure(f'{suffix}_{TOP_K}/Captum_{cam_name} *** Groundtruth: label: {best_labels[idx]}({gt_label_name}) Probability: {best_gt_prob[idx]*100:.2f} *** Predicted label: {best_predictions[idx]}({pred_label_name}) Probability: {best_propabilities[idx]*100:.2f} *** Idx: {best_idx[idx]}',fig)
        self.logger.experiment.add_text('XAI results',json.dumps(xai_results,indent=4))
        self.test_step_prediction.clear()
        self.test_step_label.clear()
        self.test_step_input.clear()
class OfflineResponseBasedDistillation(ImageClassifierBase):
    def __init__(self,student_model,
                 teacher_model,
                 lr=0.1,
                 batch_size=32,
                 momentum=0.9,
                 weight_decay=2e-5,
                 norm_weight_decay=0.0,
                 label_smoothing=0.0,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 optimizer_algorithm='sgd',
                 num_workers=0,
                 log_config=None,
                 epochs=150,
                 alpha=0.95,
                 T=3.5):
        
        super().__init__(
            base_model=student_model,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            momentum=momentum,
            weight_decay=weight_decay,
            norm_weight_decay=norm_weight_decay,
            label_smoothing=label_smoothing,
            lr_scheduler=lr_scheduler,
            lr_warmup_epochs=lr_warmup_epochs,
            lr_warmup_method=lr_warmup_method,
            lr_warmup_decay=lr_warmup_decay,
            optimizer_algorithm=optimizer_algorithm,
            num_workers=num_workers,
            log_config=log_config,
        )
        self.student_model = student_model
        self.teacher_model = teacher_model
        #gradient computation not needed for teacher model!
        self.teacher_model.freeze_layers()
        self.alpha = alpha
        self.T = T
        self.name = f'Offline_Response_Based{self.student_model.name}_{self.teacher_model.name}_alpha={self.alpha}_T={self.T}'

        self.save_hyperparameters({
            'alpha':self.alpha,
            'T':self.T,
            'name':self.name
        })

        #add hyperparameters of each model to the hparams dict of the KD class
        for key in self.student_model.hparams:
            self.save_hyperparameters({f"student_{key}":self.student_model.hparams[key]})
            
        for key in self.teacher_model.hparams:
            self.save_hyperparameters({f"teacher_{key}":self.teacher_model.hparams[key]})

    
    def training_step(self, batch,batch_idx):
        images, labels = batch
        #Teacher-model should be in evaluation mode (this turns off dropout etc)
        self.teacher_model.eval()
        #No gradient computation for teacher-model
        with torch.no_grad():  
             output_teacher = self.teacher_model(images)

        #Training mode for student-model, gradient computation should be ON
        output_student = self(images)
        #one-hot encoded (because of cutmix & mixup), convert to class label
        if labels.size(dim=-1) == NUM_CLASSES:
            labels = torch.argmax(labels,dim=1)


        #Consider using self.criterion for the cr_loss
        #Try 3 different options
        #1. kd_loss + cr_loss (like in the original paper)
        #2. mse_loss + cr_loss (https://arxiv.org/pdf/2105.08919.pdf)
        #3. only mse_loss during training, but cross entroy during validation
        kd_loss, cr_loss, total_loss = knowledge_distillation_loss(student_output=output_student,
                                                                   teacher_output=output_teacher,
                                                                   labels=labels,
                                                                   label_smoothing=0.0,
                                                                   alpha=self.alpha,
                                                                   T=self.T)
        accuracy_top_1 = self.accuracy_top_1(output_student,labels)
        accuracy_top_5 = self.accuracy_top_5(output_student,labels)

        mcc = self.mcc(output_student,labels)

        #log all three loses in 1 graph, each step
        self.logger.experiment.add_scalars('loss_step',{
            'total_loss':total_loss,
            'cr_loss':cr_loss,
            'kd_loss':kd_loss
            },
        global_step=self.global_step)
        #log all three loses in 1 graph, each epoch
        self.logger.experiment.add_scalars('loss_epoch',{
            'total_loss':total_loss,
            'cr_loss':cr_loss,
            'kd_loss':kd_loss
            },
        global_step=self.current_epoch)

        
        self.log("train_accuracy",accuracy_top_1,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_accuracy_top_5",accuracy_top_5,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_kd_loss",kd_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_cr_loss",cr_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_total_loss",total_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_mcc",mcc,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        return total_loss
    
    def validation_step(self,batch,batch_idx):
        images, labels = batch
        #Teacher-model should be in evaluation mode (this turns off dropout etc)
        self.teacher_model.eval()
        #No gradient computation for teacher-model
        with torch.no_grad():  
             output_teacher = self.teacher_model(images)

        #Training mode for student-model, gradient computation should be ON
        output_student = self(images)

        kd_loss, cr_loss, total_loss = knowledge_distillation_loss(student_output=output_student,
                                                                   teacher_output=output_teacher,
                                                                   labels=labels,
                                                                   alpha=self.alpha,
                                                                   T=self.T)
        accuracy_top_1 = self.accuracy_top_1(output_student,labels)
        accuracy_top_5 = self.accuracy_top_5(output_student,labels)

        mcc = self.mcc(output_student,labels)

         #log all three loses in 1 graph, each step
        self.logger.experiment.add_scalars('loss_step',{
            'total_loss':total_loss,
            'cr_loss':cr_loss,
            'kd_loss':kd_loss
            },
        global_step=self.global_step)
        #log all three loses in 1 graph, each epoch
        self.logger.experiment.add_scalars('loss_epoch',{
            'total_loss':total_loss,
            'cr_loss':cr_loss,
            'kd_loss':kd_loss
            },
        global_step=self.current_epoch)

        self.log("validation_accuracy",accuracy_top_1,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("validation_accuracy_top_5",accuracy_top_5,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("validation_kd_loss",kd_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("validation_cr_loss",cr_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("validation_loss",total_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("validation_mcc",mcc,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        return total_loss
    
    def test_step(self,batch,batch_idx):
        images, labels = batch
        #Teacher-model should be in evaluation mode (this turns off dropout etc)
        self.teacher_model.eval()
        #No gradient computation for teacher-model
        with torch.no_grad():  
             output_teacher = self.teacher_model(images)

        #Training mode for student-model, gradient computation should be ON
        output_student = self(images)

        kd_loss, cr_loss, total_loss = knowledge_distillation_loss(student_output=output_student,
                                                                   teacher_output=output_teacher,
                                                                   labels=labels,
                                                                   alpha=self.alpha,
                                                                   T=self.T)
        accuracy_top_1 = self.accuracy_top_1(output_student,labels)
        accuracy_top_5 = self.accuracy_top_5(output_student,labels)
        mcc = self.mcc(output_student,labels)


        self.log("test_accuracy",accuracy_top_1,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("test_accuracy_top_5",accuracy_top_5,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("test_kd_loss",kd_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("test_cr_loss",cr_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("test_loss",total_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("test_mcc",mcc,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        return total_loss
class EfficientNet_B0(ImageClassifierBase):
    def __init__(self,
                 lr=0.1,
                 weight_decay=0.00002,
                 batch_size=32,
                 momentum=0.9,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 epochs=150,
                 norm_weight_decay=0.0,
                 optimizer_algorithm='sgd',
                 num_workers=0,
                 log_config=None
                 ):
        super().__init__(lr=lr,
                         log_config=log_config,
                         base_model=efficientnet_b0(weights=None, num_classes=NUM_CLASSES),
                         batch_size=batch_size,
                         epochs=epochs,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler=lr_scheduler,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         num_workers=num_workers)
class EfficientNet_V2_S(ImageClassifierBase):
    def __init__(self,lr=0.1,
                 weight_decay=0.00002,
                 batch_size=32,
                 momentum=0.9,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 epochs=150,
                 norm_weight_decay=0.0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(lr=lr,
                        log_config=log_config,
                         base_model=efficientnet_v2_s(weights=None, num_classes=NUM_CLASSES),
                         batch_size=batch_size,
                         epochs=epochs,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler=lr_scheduler,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         num_workers=num_workers)
class EfficientNet_V2_M(ImageClassifierBase):
    def __init__(self,lr=0.01,
                 weight_decay=0.00002,
                 momentum=0.9,
                 batch_size=32,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 epochs=150,
                 norm_weight_decay=0.0,
                 optimizer_algorithm='sgd',
                    log_config=None,
                 num_workers=0):
        super().__init__(lr=lr,
                        base_model=efficientnet_v2_m(weights=None, num_classes=NUM_CLASSES),
                         log_config=log_config,
                         batch_size=batch_size,
                         epochs=epochs,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler=lr_scheduler,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         num_workers=num_workers)
class EfficientNet_V2_L(ImageClassifierBase):
    def __init__(self,lr=0.01,
                 weight_decay=0.00002,
                 momentum=0.9,
                 batch_size=32,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 epochs=150,
                 norm_weight_decay=0.0,
                 optimizer_algorithm='sgd',
                    log_config=None,
                 num_workers=0):
        super().__init__(lr=lr,
                        base_model=efficientnet_v2_l(weights=None, num_classes=NUM_CLASSES),
                         log_config=log_config,
                         batch_size=batch_size,
                         epochs=epochs,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler=lr_scheduler,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         num_workers=num_workers)
class EfficientNetPretrainedBase(ImageClassifierBase):
    def __init__(self,
                 base_model,
                 lr=0.01,
                 batch_size=32,
                 epochs=150,
                 weight_decay=2e-5,
                 momentum=0.9,
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 training_mode='pre_train',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(lr=lr,
                         base_model=base_model,
                         log_config=log_config,
                        batch_size=batch_size,
                        momentum=momentum,
                        epochs=epochs,
                         weight_decay=weight_decay,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler='cosineannealinglr' if training_mode == 'fine_tune' else 'reducelronplateu',
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         num_workers=num_workers)
        self.training_mode = training_mode
        print(f"\ttraining_mode={self.training_mode}")
        self.freeze_layers()
        self.name = self.name + "_" + self.training_mode
        #Change final classification layer
        #This is necessary, because a pre-trained network is pre-trained on imagenet with 1000 classes
        #The final layer will only be trained
        in_features =self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features=in_features,out_features=NUM_CLASSES)

    def configure_optimizers(self) -> Any:
        parameters = utils.set_weight_decay(
            self,
            self.weight_decay,
            self.norm_weight_decay,
            None
        )
        if self.training_mode == 'pre_train':
            if self.optimizer_algorithm == 'sgd':
                optimizer = SGD(parameters,lr=self.lr,momentum=self.momentum,weight_decay=self.weight_decay)
            elif self.optimizer_algorithm == 'adam':
                optimizer = AdamW(parameters,lr=self.lr,weight_decay=self.weight_decay)
            else:
                raise RuntimeError(
                f"Invalid optimizer algorithm '{self.lr_scheduler}'. Only sgd or adam is supported."
            )
            scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=4)
            return [optimizer],[{"scheduler": scheduler,'monitor':'validation_loss'}]
        if self.training_mode == 'fine_tune':
            if self.optimizer_algorithm == 'sgd':
                optimizer = SGD(parameters,lr=self.lr,momentum=self.momentum,weight_decay=self.weight_decay)
            elif self.optimizer_algorithm == 'adam':
                optimizer = AdamW(parameters,lr=self.lr,weight_decay=self.weight_decay)
            else:
                raise RuntimeError(
                f"Invalid optimizer algorithm '{self.lr_scheduler}'. Only sgd or adam is supported."
            )
            if self.lr_scheduler == 'cosineannealinglr':
                scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs - self.lr_warmup_epochs)
            else:
                raise RuntimeError(
                    f"Invalid lr scheduler '{self.lr_scheduler}'. Only cosineannealinglr is supported."
                )
            if self.lr_warmup_epochs > 0:
                if self.lr_warmup_method == 'linear':
                    warmup_lr_scheduler = LinearLR(optimizer, start_factor=self.lr_warmup_decay, total_iters=self.lr_warmup_epochs)
                else:
                    raise RuntimeError(
                    f"Invalid lr warmup method '{self.lr_warmup_method}'. Only linear is supported."
                )
                lr_scheduler = SequentialLR(optimizer,schedulers=[warmup_lr_scheduler,scheduler],milestones=[self.lr_warmup_epochs])
            else :
                lr_scheduler = scheduler
            return [optimizer],[{"scheduler": lr_scheduler,'interval':'epoch'}]      
class EfficientNet_V2_S_Pretrained(EfficientNetPretrainedBase):
    def __init__(self,
                 lr=0.01,
                 batch_size=32,
                 epochs=150,
                 momentum=0.9,
                 weight_decay=2e-5,
                 training_mode='pre_train',
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(
                        base_model=efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT),
                        lr=lr,
                         weight_decay=weight_decay,
                         batch_size=batch_size,
                         momentum=momentum,
                         epochs=epochs,
                         training_mode=training_mode,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         log_config=log_config,
                         num_workers=num_workers)
class EfficientNet_V2_M_Pretrained(EfficientNetPretrainedBase):
    def __init__(self,
                 lr=0.01,
                 batch_size=32,
                 epochs=150,
                 momentum=0.9,
                 weight_decay=2e-5,
                 training_mode='pre_train',
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(
                        base_model=efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT),
                        lr=lr,
                         weight_decay=weight_decay,
                         batch_size=batch_size,
                         momentum=momentum,
                         epochs=epochs,
                         training_mode=training_mode,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                            log_config=log_config,
                         num_workers=num_workers)
class EfficientNet_V2_L_Pretrained(EfficientNetPretrainedBase):
    def __init__(self,
                 lr=0.01,
                 batch_size=32,
                 epochs=150,
                 momentum=0.9,
                 weight_decay=2e-5,
                 training_mode='pre_train',
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(
                         base_model=efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT),
                         lr=lr,
                         weight_decay=weight_decay,
                         batch_size=batch_size,
                         momentum=momentum,
                         epochs=epochs,
                         training_mode=training_mode,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         log_config=log_config,
                         num_workers=num_workers)
class VisionTransformer_B_16(ImageClassifierBase): 
    def __init__(self,lr=0.1,
                 weight_decay=0.00002,
                 batch_size=32,
                 momentum=0.9,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 epochs=150,
                 norm_weight_decay=0.0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(lr=lr,
                        log_config=log_config,
                         base_model=vit_b_16(weights=None,num_classes=NUM_CLASSES),
                         batch_size=batch_size,
                         epochs=epochs,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler=lr_scheduler,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         num_workers=num_workers)
class VisionTransformer_L_16(ImageClassifierBase): 
    def __init__(self,lr=0.1,
                 weight_decay=0.00002,
                 batch_size=32,
                 momentum=0.9,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 epochs=150,
                 norm_weight_decay=0.0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(lr=lr,
                        log_config=log_config,
                         base_model=vit_l_16(weights=None,num_classes=NUM_CLASSES),
                         batch_size=batch_size,
                         epochs=epochs,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler=lr_scheduler,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         num_workers=num_workers)
class VisionTransformer_H_14(ImageClassifierBase):
     def __init__(self,lr=0.1,
                 weight_decay=0.00002,
                 batch_size=32,
                 momentum=0.9,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 epochs=150,
                 norm_weight_decay=0.0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(lr=lr,
                        log_config=log_config,
                         base_model=vit_h_14(weights=None,num_classes=NUM_CLASSES),
                         batch_size=batch_size,
                         epochs=epochs,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler=lr_scheduler,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         num_workers=num_workers)
class VisionTransformerPretrainedBase(ImageClassifierBase):
    def __init__(self,
                 base_model,
                 lr=0.01,
                 batch_size=32,
                 epochs=150,
                 weight_decay=2e-5,
                 momentum=0.9,
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 training_mode='pre_train',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(lr=lr,
                         base_model=base_model,
                         log_config=log_config,
                        batch_size=batch_size,
                        momentum=momentum,
                        epochs=epochs,
                         weight_decay=weight_decay,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler='cosineannealinglr' if training_mode == 'fine_tune' else 'reducelronplateu',
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         num_workers=num_workers)
        self.training_mode = training_mode
        print(f"\ttraining_mode={self.training_mode}")
        self.freeze_layers()
        self.name = self.name + "_" + self.training_mode
        
        #Change final classification layer
        #This is necessary, because a pre-trained network is pre-trained on imagenet with 1000 classes
        #The final layer will only be trained
        assert hasattr(self.model.heads, "head") and isinstance(self.model.heads.head, nn.Linear)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, NUM_CLASSES)

    def configure_optimizers(self) -> Any:
        parameters = utils.set_weight_decay(
            self,
            self.weight_decay,
            self.norm_weight_decay,
            None
        )
        if self.training_mode == 'pre_train':
            if self.optimizer_algorithm == 'sgd':
                optimizer = SGD(parameters,lr=self.lr,momentum=self.momentum,weight_decay=self.weight_decay)
            elif self.optimizer_algorithm == 'adam':
                optimizer = AdamW(parameters,lr=self.lr,weight_decay=self.weight_decay)
            else:
                raise RuntimeError(
                f"Invalid optimizer algorithm '{self.lr_scheduler}'. Only sgd or adam is supported."
            )
            scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=4)
            return [optimizer],[{"scheduler": scheduler,'monitor':'validation_loss'}]
        if self.training_mode == 'fine_tune':
            if self.optimizer_algorithm == 'sgd':
                optimizer = SGD(parameters,lr=self.lr,momentum=self.momentum,weight_decay=self.weight_decay)
            elif self.optimizer_algorithm == 'adam':
                optimizer = AdamW(parameters,lr=self.lr,weight_decay=self.weight_decay)
            else:
                raise RuntimeError(
                f"Invalid optimizer algorithm '{self.lr_scheduler}'. Only sgd or adam is supported."
            )
            if self.lr_scheduler == 'cosineannealinglr':
                scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs - self.lr_warmup_epochs)
            else:
                raise RuntimeError(
                    f"Invalid lr scheduler '{self.lr_scheduler}'. Only cosineannealinglr is supported."
                )
            if self.lr_warmup_epochs > 0:
                if self.lr_warmup_method == 'linear':
                    warmup_lr_scheduler = LinearLR(optimizer, start_factor=self.lr_warmup_decay, total_iters=self.lr_warmup_epochs)
                else:
                    raise RuntimeError(
                    f"Invalid lr warmup method '{self.lr_warmup_method}'. Only linear is supported."
                )
                lr_scheduler = SequentialLR(optimizer,schedulers=[warmup_lr_scheduler,scheduler],milestones=[self.lr_warmup_epochs])
            else :
                lr_scheduler = scheduler
            return [optimizer],[{"scheduler": lr_scheduler,'interval':'epoch'}]
class VisionTransformer_L_16_Pretrained(VisionTransformerPretrainedBase):
    def __init__(self,
                 lr=0.01,
                 batch_size=32,
                 epochs=150,
                 momentum=0.9,
                 weight_decay=2e-5,
                 training_mode='pre_train',
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(
                         base_model=vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1),
                         lr=lr,
                         weight_decay=weight_decay,
                         batch_size=batch_size,
                         momentum=momentum,
                         epochs=epochs,
                         training_mode=training_mode,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         log_config=log_config,
                            num_workers=num_workers)
class VisionTransformer_H_14_Pretrained(VisionTransformerPretrainedBase):
    def __init__(self,
                 lr=0.01,
                 batch_size=32,
                 epochs=150,
                 momentum=0.9,
                 weight_decay=2e-5,
                 training_mode='pre_train',
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(
                         base_model=vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1),
                         lr=lr,
                         weight_decay=weight_decay,
                         batch_size=batch_size,
                         momentum=momentum,
                         epochs=epochs,
                         training_mode=training_mode,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         log_config=log_config,
                            num_workers=num_workers)
class Resnet_18(ImageClassifierBase):
    def __init__(self,lr=0.01,
                 weight_decay=0.00002,
                 momentum=0.9,
                 batch_size=32,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 epochs=150,
                 norm_weight_decay=0.0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(
                         base_model=resnet18(weights=None,num_classes=NUM_CLASSES),
                         lr=lr,
                         batch_size=batch_size,
                         epochs=epochs,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler=lr_scheduler,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                            log_config=log_config,
                         num_workers=num_workers)
        

class Resnet_18_Dropout(ImageClassifierBase):
    def __init__(self,dropout,
                 lr=0.01,
                 weight_decay=0.00002,
                 momentum=0.9,
                 batch_size=32,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 epochs=150,
                 norm_weight_decay=0.0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(lr=lr,
                         base_model=resnet18(weights=None,num_classes=NUM_CLASSES),
                         batch_size=batch_size,
                         epochs=epochs,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler=lr_scheduler,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         log_config=log_config,
                         num_workers=num_workers)
        self.dropout = dropout
        fc_layer = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True),
           fc_layer
        )
class Resnet_50(ImageClassifierBase):
    def __init__(self,lr=0.01,
                 weight_decay=0.00002,
                 momentum=0.9,
                 batch_size=32,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 epochs=150,
                 norm_weight_decay=0.0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(
                         base_model=resnet50(weights=None,num_classes=NUM_CLASSES),
                         lr=lr,
                         batch_size=batch_size,
                         epochs=epochs,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler=lr_scheduler,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         log_config=log_config,
                         num_workers=num_workers)
class Resnet_101(ImageClassifierBase):
    def __init__(self,lr=0.01,
                 weight_decay=0.00002,
                 momentum=0.9,
                 batch_size=32,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 epochs=150,
                 norm_weight_decay=0.0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(
                         base_model=resnet101(weights=None,num_classes=NUM_CLASSES),
                         lr=lr,
                         batch_size=batch_size,
                         epochs=epochs,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler=lr_scheduler,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         log_config=log_config,
                         num_workers=num_workers)
class AlexNet(ImageClassifierBase):
    def __init__(self,lr=0.01,
                 weight_decay=0.00002,
                 momentum=0.9,
                 batch_size=32,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 epochs=150,
                 norm_weight_decay=0.0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(
                         base_model=alexnet(weights=None,num_classes=NUM_CLASSES),
                         lr=lr,
                         batch_size=batch_size,
                         epochs=epochs,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler=lr_scheduler,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         log_config=log_config,
                         num_workers=num_workers)
        self.init_weights(self.model) #initialize weights for stability
    
    def init_weights(self,m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)
class NaiveClassifier(ImageClassifierBase):
    def __init__(self,lr=0.01,
                 weight_decay=0.00002,
                 momentum=0.9,
                 batch_size=32,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 epochs=150,
                 norm_weight_decay=0.0,
                 optimizer_algorithm='sgd',
                 log_config=None,
                 num_workers=0):
        super().__init__(
                         base_model=Naive(),
                         lr=lr,
                         batch_size=batch_size,
                         epochs=epochs,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         norm_weight_decay=norm_weight_decay,
                         label_smoothing=label_smoothing,
                         lr_scheduler=lr_scheduler,
                         lr_warmup_epochs=lr_warmup_epochs,
                         lr_warmup_method=lr_warmup_method,
                         lr_warmup_decay=lr_warmup_decay,
                         optimizer_algorithm=optimizer_algorithm,
                         log_config=log_config,
                         num_workers=num_workers)
class Naive(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5,2)
        self.conv3 = nn.Conv2d(64, 32, 5,2)
        self.conv4 = nn.Conv2d(32, 32, 5,2)
        self.conv5 = nn.Conv2d(32, 32, 5,2)
        # Adaptive pooling layer (so we can work with any input-shape)
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(3200, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        #x = self.adaptive_pool(x) #creates (1,1) features maps
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class SimKD(nn.Module):
    #https://arxiv.org/pdf/2203.14001.pdf
    def __init__(self, *, s_n, t_n, factor=2): 
        super(SimKD, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))       

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)
        # A bottleneck design to reduce extra parameters
        self.transfer = nn.Sequential(
            conv1x1(s_n, t_n//factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),
            conv3x3(t_n//factor, t_n//factor),
            #conv3x3(t_n//factor, t_n//factor, groups=t_n//factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),
            conv1x1(t_n//factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
        )
    def forward(self, feat_s, feat_t, cls_t):
        # Spatial Dimension Alignment
        s_H, t_H = feat_s.shape[2], feat_t.shape[2]
        if s_H > t_H:
            source = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
            target = feat_t
        else:
            source = feat_s
            target = F.adaptive_avg_pool2d(feat_t, (s_H, s_H))
        
        trans_feat_t=target
        
        # Channel Alignment
        trans_feat_s = self.transfer(source)

        # Prediction via Teacher Classifier
        temp_feat = self.avg_pool(trans_feat_s)
        temp_feat = temp_feat.view(temp_feat.size(0), -1)
        pred_feat_s = cls_t(temp_feat)
        
        return trans_feat_s, trans_feat_t, pred_feat_s
    
class OfflineFeatureBasedDistillation(ImageClassifierBase):
    def __init__(self,student_model,
                 teacher_model,
                 factor=2,
                 lr=0.1,
                 batch_size=32,
                 momentum=0.9,
                 weight_decay=2e-5,
                 norm_weight_decay=0.0,
                 label_smoothing=0.0,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 optimizer_algorithm='sgd',
                 num_workers=0,
                 log_config=None,
                 epochs=150,
                 ):
        
        super().__init__(
            base_model=student_model,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            momentum=momentum,
            weight_decay=weight_decay,
            norm_weight_decay=norm_weight_decay,
            label_smoothing=label_smoothing,
            lr_scheduler=lr_scheduler,
            lr_warmup_epochs=lr_warmup_epochs,
            lr_warmup_method=lr_warmup_method,
            lr_warmup_decay=lr_warmup_decay,
            optimizer_algorithm=optimizer_algorithm,
            num_workers=num_workers,
            log_config=log_config,
        )
        self.student_model = student_model
        self.teacher_model = teacher_model

        # See #https://arxiv.org/pdf/2203.14001.pdf for more details
        # Output channels of the last feature layer of the student model
        s_n = self.student_model.model.features[-1].out_channels
        # Output channels of the last feature layer of the teacher model
        t_n = self.teacher_model.model.features[-1].out_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)

        # A bottleneck design to reduce extra parameters
        self.transfer = nn.Sequential(
            conv1x1(s_n, t_n//factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),
            conv3x3(t_n//factor, t_n//factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),
            conv1x1(t_n//factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
        )

        #gradient computation not needed for teacher model!
        self.teacher_model.freeze_layers()
        self.name = f'Offline_Feature_Based_{self.student_model.name}_{self.teacher_model.name}'

        self.save_hyperparameters({
            'name':self.name
        })

        #add hyperparameters of each model to the hparams dict of the KD class
        for key in self.student_model.hparams:
            self.save_hyperparameters({f"student_{key}":self.student_model.hparams[key]})
            
        for key in self.teacher_model.hparams:
            self.save_hyperparameters({f"teacher_{key}":self.teacher_model.hparams[key]})

    def forward(self, x):
        # Extract features from the student model
        feat_s = self.student_model.model.features(x)
        # Teacher-model should be in evaluation mode (this turns off dropout etc)
        self.teacher_model.eval()
        feat_t_dummy = self.teacher_model.model.features(x)
            
        if feat_s.shape[2] > feat_t_dummy.shape[2]:
            feat_s = F.adaptive_avg_pool2d(feat_s, (feat_t_dummy.shape[2], feat_t_dummy.shape[2]))
        

        trans_feat_s = self.transfer(feat_s)

        # Prediction via Teacher Classifier
        temp_feat = self.avg_pool(trans_feat_s)
        temp_feat = torch.flatten(temp_feat, 1)
        pred_feat_s = self.teacher_model.model.classifier(temp_feat)
        return pred_feat_s
        

    def training_step(self, batch,batch_idx):
        images, labels = batch
        #Teacher-model should be in evaluation mode (this turns off dropout etc)
        self.teacher_model.eval()
        #No gradient computation for teacher-model
        with torch.no_grad():  
            #output_teacher = self.teacher_model(images)
            features_teacher = self.teacher_model.model.features(images)


        #Training mode for student-model, gradient computation should be ON
        features_student = self.student_model.model.features(images)
        
        #one-hot encoded (because of cutmix & mixup), convert to class label
        if labels.size(dim=-1) == NUM_CLASSES:
            labels = torch.argmax(labels,dim=1)
    
        

        if features_student.shape[2] > features_teacher.shape[2]:
            features_student = F.adaptive_avg_pool2d(features_student, (features_teacher.shape[2], features_teacher.shape[2]))
        else:
            with torch.no_grad():  
                features_teacher = F.adaptive_avg_pool2d(features_teacher, (features_student.shape[2], features_student.shape[2]))

        transformed_features_student = self.transfer(features_student)
        with torch.no_grad():  
            temp_features = self.avg_pool(transformed_features_student)
            temp_features = torch.flatten(temp_features, 1)
            predicted_features_student = self.teacher_model.model.classifier(temp_features)


        total_loss = F.mse_loss(transformed_features_student,features_teacher)
        accuracy_top_1 = self.accuracy_top_1(predicted_features_student,labels)
        accuracy_top_5 = self.accuracy_top_5(predicted_features_student,labels)


        self.log("train_accuracy",accuracy_top_1,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_accuracy_top_5",accuracy_top_5,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_loss",total_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        return total_loss
    
    def validation_step(self,batch,batch_idx):
        images, labels = batch
        #Teacher-model should be in evaluation mode (this turns off dropout etc)
        self.teacher_model.eval()
        #No gradient computation for teacher-model
        with torch.no_grad():  
            #output_teacher = self.teacher_model(images)
            features_teacher = self.teacher_model.model.features(images)


        #Training mode for student-model, gradient computation should be ON
        features_student = self.student_model.model.features(images)
        
        #one-hot encoded (because of cutmix & mixup), convert to class label
        if labels.size(dim=-1) == NUM_CLASSES:
            labels = torch.argmax(labels,dim=1)

        if features_student.shape[2] > features_teacher.shape[2]:
            features_student = F.adaptive_avg_pool2d(features_student, (features_teacher.shape[2], features_teacher.shape[2]))
        else:
            with torch.no_grad():  
                features_teacher = F.adaptive_avg_pool2d(features_teacher, (features_student.shape[2], features_student.shape[2]))

        transformed_features_student = self.transfer(features_student)
        with torch.no_grad():  
            temp_features = self.avg_pool(transformed_features_student)
            temp_features = torch.flatten(temp_features, 1)
            predicted_features_student = self.teacher_model.model.classifier(temp_features)


        total_loss = F.mse_loss(transformed_features_student,features_teacher)
        accuracy_top_1 = self.accuracy_top_1(predicted_features_student,labels)
        accuracy_top_5 = self.accuracy_top_5(predicted_features_student,labels)


        self.log("validation_accuracy",accuracy_top_1,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("validation_accuracy_top_5",accuracy_top_5,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("validation_loss",total_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        return total_loss
    
    def test_step(self,batch,batch_idx):
        pass