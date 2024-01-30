from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch 
import torch.nn.functional as F
from torch.nn.modules import Module
from torchvision.models import efficientnet_b0,efficientnet_v2_s,efficientnet_v2_m,efficientnet_v2_l,EfficientNet_V2_S_Weights,EfficientNet_V2_M_Weights,EfficientNet_B0_Weights,EfficientNet_V2_L_Weights
from torchvision.models import resnet18,resnet101
from torchvision.models import vit_b_16,ViT_B_16_Weights
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import Accuracy,F1Score,MatthewsCorrCoef,ConfusionMatrix,ROC,AUROC
from lion_pytorch import Lion
from abc import ABC,abstractmethod
from torch.optim import AdamW,SGD
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,ExponentialLR,ReduceLROnPlateau,StepLR,LinearLR,SequentialLR
from knowledge_distillation import knowledge_distillation_loss
import utils
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from sklearn.metrics import classification_report
from tqdm import tqdm
from functools import partial
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import IntegratedGradients,GuidedGradCam,GradientShap,Saliency,NoiseTunnel
from captum.attr import visualization as viz
from pytorch_grad_cam import GradCAM, ScoreCAM, HiResCAM, EigenCAM, AblationCAM, GradCAMPlusPlus, GradCAMElementWise
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from transformations import denormalize
from torchvision.transforms import transforms
from transformations import IMAGENET_MEAN,IMAGENET_STD
from image_utils import create_multiple_images

NUM_CLASSES = 524
TOP_K = 10


class ImageClassifierBase(ABC,pl.LightningModule):
    def __init__(self,lr=0.1,
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
                 model = None
                 ):
        super().__init__()
        self.model = self.init_base_model()
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
        

        self.accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.mcc = MatthewsCorrCoef(task="multiclass",num_classes=NUM_CLASSES)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.confusion_matrix = ConfusionMatrix(task="multiclass",num_classes=NUM_CLASSES)
        self.roc_curve = ROC(task='multiclass',num_classes=NUM_CLASSES)
        self.auroc = AUROC(task='multiclass',num_classes=NUM_CLASSES)
        
        #self.logger:TensorBoardLogger = TensorBoardLogger(save_dir=':/',log_graph=True)
        self.test_step_prediction = []
        self.test_step_label = []
        self.test_step_input = []


        self.log_config = {
            'confusion_matrix':True,
            'roc_curve':True,
            'auroc':True,
            'classification_report':True,
            'pytorch_cam':True,
            'captum_alg':True,
            'topk':True,
            'bottomk':True,
            'randomk':True
        }

        if log_config:
            self.log_config.update(log_config)

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
        

    @abstractmethod
    def init_base_model(self) -> nn.Module:
        pass

    def forward(self,x,is_feat=False):
        if is_feat:
            out = self.model(x,is_feat)
        else:
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
        self.logger.log_graph(model=self,input_array=torch.rand((1,3,224,224)).to('cuda'))


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
    
    def training_step(self, batch,batch_idx):
        inputs, labels = batch
        output = self(inputs)

        #one-hot encoded (because of cutmix & mixup), convert to class label
        if labels.size(dim=-1) == NUM_CLASSES:
            labels = torch.argmax(labels,dim=1)

        loss = self.criterion(output,labels)
        accuracy = self.accuracy(output,labels)
        mcc = self.mcc(output,labels)

        self.logger.experiment.add_scalars('loss',{'train':loss},global_step=self.global_step)
        self.logger.experiment.add_scalars('accuracy',{'train':accuracy},global_step=self.global_step)
        self.logger.experiment.add_scalars('mcc',{'train':mcc},global_step=self.global_step)

        self.log("train_accuracy",accuracy,on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("train_loss",loss,on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("train_mcc",mcc,on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        return loss
    

    def validation_step(self,batch,batch_idx):
        inputs, labels = batch
        output = self(inputs)
        loss = self.criterion(output,labels)
        accuracy = self.accuracy(output,labels)
        mcc = self.mcc(output,labels)


        self.logger.experiment.add_scalars('loss',{'validation':loss},global_step=self.global_step)
        self.logger.experiment.add_scalars('accuracy',{'validation':accuracy},global_step=self.global_step)
        self.logger.experiment.add_scalars('mcc',{'validation':mcc},global_step=self.global_step)


        self.log("validation_accuracy",accuracy,on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("validation_loss",loss,on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("validation_mcc",mcc,on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        
        return loss



    def test_step(self,batch,batch_idx):
        inputs, labels = batch
        output = self(inputs)
        loss = self.criterion(output,labels)
        accuracy = self.accuracy(output,labels)
        mcc = self.mcc(output,labels)

        #predictions = torch.argmax(output,dim=1)
        self.log("test_accuracy",accuracy,prog_bar=True,batch_size=self.batch_size)
        self.log("test_loss",loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_mcc",mcc,prog_bar=True,batch_size=self.batch_size)

        self.test_step_prediction.append(output)
        self.test_step_label.append(labels)
        self.test_step_input.append(inputs)
        return loss
    

    def on_test_epoch_end(self) -> None:
        all_predictions = torch.cat(self.test_step_prediction) #(2625,NUM_CLASSES)
        all_labels = torch.cat(self.test_step_label) # (2625)
        all_images = torch.cat(self.test_step_input) # (2625,3,224,224)

        all_predictions_probabilities = torch.softmax(all_predictions,dim=1) #(2625,NUM_CLASSES)
        all_predictions_max_probabilities = torch.max(all_predictions,dim=1) #(2625)

        top_k = partial(torch.topk,k = TOP_K)
        bottom_k = partial(torch.topk, k=TOP_K, largest=False)
        rand_k = partial(utils.get_k_random_values,k=TOP_K,device="cuda")

        selection_functions = []
        selection_functions += [(top_k,"Top")] if self.log_config['topk'] else []
        selection_functions += [bottom_k,"Bottom"] if self.log_config['bottomk'] else []
        selection_functions += [rand_k,"Random"] if self.log_config['randomk'] else []

        target_layers = [self.model.layer4[-1]]

        top_k_propabilities = torch.topk(all_predictions_max_probabilities.values,10) #(k)
        bottom_k_propabilities = torch.topk(all_predictions_max_probabilities.values,10,largest=False) #(k)
        all_predictions_idx = torch.argmax(all_predictions_probabilities,dim=1) #(2625)

        target_layers = [self.model.layer4[-1]]
        targets = [ClassifierOutputTarget(NUM_CLASSES-1)]
        pytorch_gradcam_cams = [
            GradCAM(model=self.model,target_layers=target_layers,use_cuda=True),
            HiResCAM(model=self.model,target_layers=target_layers,use_cuda=True),
            AblationCAM(model=self.model,target_layers=target_layers,use_cuda=True),
            GradCAMPlusPlus(model=self.model,target_layers=target_layers,use_cuda=True),
            GradCAMElementWise(model=self.model,target_layers=target_layers,use_cuda=True)
        ]

        #Captum init cams
        captum_alg = [
            IntegratedGradients(self.model),
            GuidedGradCam(model=self.model,layer=target_layers[0]),
            Saliency(self.model)
        ]

        #Create the confusion matrix
        if self.log_config['confusion_matrix']:
            print('Computing confusion matrix...')
            computed_confusion = self.confusion_matrix(all_predictions,all_labels)
            fig = utils.get_confusion_matrix_figure(computed_confusion=computed_confusion.cpu().numpy().astype(int))
            self.logger.experiment.add_figure('Confusion matrix',fig,self.current_epoch)

        if self.log_config['roc_curve']:
            #Create ROC-Curve
            fpr, tpr, thresholds = self.roc_curve(all_predictions,all_labels)
            
            #For each class, create a seperate roc-curve
            for i in tqdm(range(NUM_CLASSES),desc="ROC curve"):
                fig = utils.get_roc_curve_figure(fpr=fpr[i],tpr=tpr[i],thresholds=thresholds[i])
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

        for function, suffix in selection_functions:
            best_k_probabilities = function(all_predictions_max_probabilities.values)
            best_propabilities = best_k_probabilities[0]
            best_idx = best_k_probabilities[1]
            best_images = all_images[best_idx]
            best_labels = all_labels[best_idx]
            
            #get prediction
            best_predictions = all_predictions_idx[best_idx]
            #save/log softmax vector for each idx in tensorboard
            best_softmax_idx = all_predictions_probabilities[best_idx]
            #get ground truth probability for each of the top_k softmaxvectors
            best_gt_prob = best_softmax_idx[torch.arange(TOP_K),best_labels]
                                                    
            #pytorch-cam
            if self.log_config['pytorch_cam']:                                    
                resize = transforms.Resize(224)
                rgb_images = resize(best_images)
                denormalized_images = denormalize(rgb_images,IMAGENET_MEAN,IMAGENET_STD)
        
                #apply pytorch_gradcam_cam
                for cam in pytorch_gradcam_cams:
                    with torch.enable_grad():
                        cam_images = cam(input_tensor=best_images, targets=targets,aug_smooth=True,eigen_smooth=False)
                    cam_name = str(type(cam)).split(".")[-1][:-2]
                    
                    for idx, cam_image in enumerate(cam_images):
                        permuted_image = denormalized_images[idx].permute(1,2,0).cpu()
                        visualization = show_cam_on_image(permuted_image.numpy().astype(np.float32)/255, cam_image, use_rgb=True,image_weight=0.5)
                        result = create_multiple_images(images=[permuted_image,cam_image,visualization],titles=['Original','Heat map', 'Combined'])
                        self.logger.experiment.add_figure(f'{suffix}_{TOP_K}/{cam_name} *** Groundtruth: label: {best_labels[idx]} probability: {best_gt_prob[idx]:.2f} *** Prediction: label: {best_predictions[idx]} probability: {best_propabilities[idx]:.2f} *** Idx: {best_idx[idx]}',result)
            #captum-alg
            if self.log_config['captum_alg']:  
                for alg in captum_alg:
                    prediction_score,pred_label_idx = torch.topk(best_predictions,1)
                    cam_name = str(type(alg)).split(".")[-1][:-2]
                    attributions = alg.attribute(best_images,target=pred_label_idx) #brauchen wir variable?
                    noise_tunnel = NoiseTunnel(alg)
                    if cam_name == "IntegratedGradients":
                        attributions_nt = noise_tunnel.attribute(best_images, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx,internal_batch_size=TOP_K)
                    else:
                        attributions_nt = noise_tunnel.attribute(best_images, nt_type='smoothgrad_sq', target=pred_label_idx)
                        
                    for idx, alg_image in enumerate(attributions_nt):
                        result = viz.visualize_image_attr_multiple(np.transpose(alg_image.cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(denormalized_images[idx].cpu().detach().numpy(), (1,2,0)),
                                            methods=["original_image", "heat_map",'masked_image','alpha_scaling','blended_heat_map'],
                                            signs=['all', 'positive','positive','positive','positive'],
                                            titles=['Original','Heatmap','Masked-image','Alpha-scaling','Blended heatmap'],
                                            fig_size=(12,8),
                                            show_colorbar=True)
                        
                        self.logger.experiment.add_figure(f'{suffix}_{TOP_K}/Captum_{cam_name} *** Groundtruth: label: {best_labels[idx]}) gt_prob: {best_gt_prob[idx]:.2f} *** Prediction: label: {best_predictions[idx]}) probability: {best_propabilities[idx]:.2f} *** Idx: {best_idx[idx]}',result[0])
        self.test_step_prediction.clear()
        self.test_step_label.clear()
        self.test_step_input.clear()

class KnowledgeDistillationModule(pl.LightningModule):
    def __init__(self,student_model,
                 teacher_model,
                 lr=0.1,
                 momentum=0.9,
                 weight_decay=2e-5,
                 norm_weight_decay=0.0,
                 batch_size=32,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 optimizer_algorithm='sgd',
                 num_workers=0,
                 epochs=150,
                 alpha=0.95,
                 T=3.5):
        self.student_model = student_model
        self.teacher_model = teacher_model
        #gradient computation not needed for teacher model!
        self.teacher_model.freeze_layers()
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.norm_weight_decay = norm_weight_decay
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_warmup_method = lr_warmup_method
        self.lr_warmup_decay = lr_warmup_decay
        self.optimizer_algorithm = optimizer_algorithm
        self.num_workers = num_workers
        self.epochs = epochs
        self.alpha = alpha
        self.T = T
        self.name = f'{self.student_model.name}_{self.teacher_model.name}_alpha={self.alpha}_T={self.T}'
        self.accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.mcc = MatthewsCorrCoef(task="multiclass",num_classes=NUM_CLASSES)

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
            'name':self.name,
            'alpha':self.alpha,
            'T':self.T,
        })

        #add hyperparameters of each model to the hparams dict of the KD class
        for key in self.student_model.hparams:
            self.save_hyperparameters({f"student_{key}":self.student_model.hparams[key]})
            
        for key in self.teacher_model.hparams:
            self.save_hyperparameters({f"teacher_{key}":self.teacher_model.hparams[key]})

    def forward(self, x) -> Any:
        out = self.student_model(x)
        return out

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
    
    def training_step(self, batch,batch_idx):
        images, labels = batch
        #Teacher-model should be in evaluation mode (this turns off dropout etc)
        self.teacher_model.eval()
        #No gradient computation for teacher-model
        with torch.no_grad():  
             output_teacher = self.teacher_model(images)

        #Training mode for student-model, gradient computation should be ON
        output_student = self(images)

        tqdm([1,2,3])
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
        accuracy = self.accuracy(output_student,labels)
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

        self.log("train_accuracy",accuracy,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
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
        accuracy = self.accuracy(output_student,labels)
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

        self.log("validation_accuracy",accuracy,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
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
        accuracy = self.accuracy(output_student,labels)
        mcc = self.mcc(output_student,labels)


        self.log("test_accuracy",accuracy,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("test_kd_loss",kd_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("test_cr_loss",cr_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("test_loss",total_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("test_mcc",mcc,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        return total_loss

class EfficientNet_B0(ImageClassifierBase):
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
                 num_workers=0):
        super().__init__(lr=lr,
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
    def init_base_model(self):
        return efficientnet_b0(weights=None, num_classes=NUM_CLASSES)   
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
                 num_workers=0):
        super().__init__(lr=lr,
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
    def init_base_model(self):
        return efficientnet_v2_s(weights=None, num_classes=NUM_CLASSES)
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
                 num_workers=0):
        super().__init__(lr=lr,
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
    def init_base_model(self):
        return efficientnet_v2_m(weights=None, num_classes=NUM_CLASSES)
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
                 num_workers=0):
        super().__init__(lr=lr,
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
    def init_base_model(self):
        return efficientnet_v2_l(weights=None, num_classes=NUM_CLASSES)
    
class EfficientNetPretrainedBase(ImageClassifierBase):
    def __init__(self,
                 lr,
                 batch_size,
                 epochs,
                 weight_decay=2e-5,
                 momentum=0.9,
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 training_mode='pre_train',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 optimizer_algorithm='sgd',
                 num_workers=0):
        super().__init__(lr=lr,
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
                 lr,
                 batch_size,
                 epochs,
                 momentum=0.9,
                 weight_decay=2e-5,
                 training_mode='pre_train',
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0,
                 optimizer_algorithm='sgd',
                 num_workers=0):
        super().__init__(lr=lr,
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
                         num_workers=num_workers)
    def init_base_model(self):
        return efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
class EfficientNet_V2_M_Pretrained(EfficientNetPretrainedBase):
    def __init__(self,
                 lr,
                 batch_size,
                 epochs,
                 momentum=0.9,
                 weight_decay=2e-5,
                 training_mode='pre_train',
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0,
                 optimizer_algorithm='sgd',
                 num_workers=0):
        super().__init__(lr=lr,
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
                         num_workers=num_workers)
    def init_base_model(self):
        return efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
class EfficientNet_V2_L_Pretrained(EfficientNetPretrainedBase):
    def __init__(self,
                 lr,
                 batch_size,
                 epochs,
                 momentum=0.9,
                 weight_decay=2e-5,
                 training_mode='pre_train',
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0,
                 optimizer_algorithm='sgd',
                 num_workers=0):
        super().__init__(lr=lr,
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
                         num_workers=num_workers)
    def init_base_model(self):
        return efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)

class VisionTransformer_B_16(ImageClassifierBase): 
    def __init__(self,lr,weight_decay,batch_size,mode='pre_train'):
        pass
class VisionTransformer_L_16(ImageClassifierBase): 
    pass
class VisionTransformer_H_14(ImageClassifierBase):
    pass

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
                 num_workers=0):
        super().__init__(lr=lr,
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
    def init_base_model(self):
        return resnet18(weights=None,num_classes=NUM_CLASSES)


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
                 num_workers=0):
        super().__init__(lr=lr,
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
        self.dropout = dropout
        fc_layer = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True),
           fc_layer
        )
    def init_base_model(self):
        return resnet18(weights=None,num_classes=NUM_CLASSES) 
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
                 num_workers=0):
        super().__init__(lr=lr,
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
    def init_base_model(self):
        return resnet101(weights=None, num_classes=NUM_CLASSES)
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
                 num_workers=0):
        super().__init__(lr=lr,
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
    def init_base_model(self) -> Module:
        return Naive()


class Naive(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 32, 5)
        # Adaptive pooling layer (so we can work with any input-shape)
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        #x = self.adaptive_pool(x) #creates (1,1) features maps

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimKD(nn.Module):
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