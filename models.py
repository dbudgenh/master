from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch 
import torch.nn.functional as F
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


NUM_CLASSES = 525
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
                 num_workers=0
                 ):
        super().__init__()
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
        self.save_hyperparameters()

        self.accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.mcc = MatthewsCorrCoef(task="multiclass",num_classes=NUM_CLASSES)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.confusion_matrix = ConfusionMatrix(task="multiclass",num_classes=NUM_CLASSES)
        self.roc_curve = ROC(task='multiclass',num_classes=NUM_CLASSES)
        self.auroc = AUROC(task='multiclass',num_classes=NUM_CLASSES)
        
        #self.logger:TensorBoardLogger = TensorBoardLogger(save_dir=':/',log_graph=True)
        self.test_step_prediction = []
        self.test_step_label = []

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
        ''')

    def setup(self, stage: str) -> None:
        self.logger._log_graph = True
        self.logger.log_graph(model=self,input_array=torch.rand((1,3,176,176)).to('cuda'))


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
        loss = self.criterion(output,labels)
        #one-hot encoded (because of cutmix & mixup), convert to class label
        if labels.size(dim=-1) == NUM_CLASSES:
            labels = torch.argmax(labels,dim=1)
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
        self.log("validation_accuracy",accuracy,prog_bar=True,batch_size=self.batch_size)
        self.log("validation_loss",loss,prog_bar=True,batch_size=self.batch_size)
        self.log("validation_mcc",mcc,prog_bar=True,batch_size=self.batch_size)
        
        return loss



    def test_step(self,batch,batch_idx):
        inputs, labels = batch
        output = self(inputs)
        loss = self.criterion(output,labels)
        accuracy = self.accuracy(output,labels)
        mcc = self.mcc(output,labels)

        #predictions = torch.argmax(output,dim=1)
        self.logger.experiment.add_scalars('loss',{'test':loss},global_step=self.global_step)
        self.logger.experiment.add_scalars('accuracy',{'test':accuracy},global_step=self.global_step)
        self.logger.experiment.add_scalars('mcc',{'test':mcc},global_step=self.global_step)
        self.log("test_accuracy",accuracy,prog_bar=True,batch_size=self.batch_size)
        self.log("test_loss",loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_mcc",mcc,prog_bar=True,batch_size=self.batch_size)

        self.test_step_prediction.append(output)
        self.test_step_label.append(labels)
        return loss
    
    def on_test_epoch_end(self) -> None:
        all_predictions = torch.cat(self.test_step_prediction)
        all_labels = torch.cat(self.test_step_label)

        #Create the confusion matrix
        self.confusion_matrix(all_predictions,all_labels)
        computed_confusion = self.confusion_matrix.compute().detach().cpu().numpy().astype(int)
        fig = utils.get_confusion_matrix_figure(computed_confusion=computed_confusion)
        self.logger.experiment.add_figure('Confusion matrix',fig,self.current_epoch)

        #Create ROC-Curve
        fpr, tpr, thresholds = self.roc_curve(all_predictions,all_labels)
        fig = utils.get_roc_curve_figure(fpr=fpr,tpr=tpr,thresholds=thresholds)
        self.logger.experiment.add_figure('ROC curve',fig,self.current_epoch)

        #Create AUROC
        auroc = self.auroc(all_predictions,all_labels)
        self.log('Area under ROC',auroc,batch_size=self.batch_size)

        #Create classification report
        report = classification_report(all_labels,all_predictions)
        self.logger.experiment.add_text('Classification report',report)

        #Clear values
        self.test_step_prediction.clear()
        self.test_step_label.clear()

class EfficientNetPretrainedBase(ImageClassifierBase,ABC):
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
        self.efficient_net = self.init_base_model()
        print(f"\ttraining_mode={self.training_mode}")
        self.freeze_layers()
        #Change final classification layer
        in_features = in_features=self.efficient_net.classifier[1].in_features
        self.efficient_net.classifier[1] = nn.Linear(in_features=in_features,out_features=NUM_CLASSES)

    @abstractmethod
    def init_base_model(self):
        pass

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
    def forward(self,x):
        out = self.efficient_net(x)
        return out
    def freeze_layers(self):
        print('Freezing layers')
        for param in self.efficient_net.parameters():
            param.requires_grad = False
    def unfreeze_layers(self):
        print('Unfreezing layers')
        for param in self.efficient_net.parameters():
            param.requires_grad = True

class KnowledgeDistillationModule(pl.LightningModule):
    def __init__(self,student_model,teacher_model,
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
        super().__init__()
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
        self.save_hyperparameters(ignore=['student_model','teacher_model'])
        self.accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.mcc = MatthewsCorrCoef(task="multiclass",num_classes=NUM_CLASSES)

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

        kd_loss, cr_loss, total_loss = knowledge_distillation_loss(student_output=output_student,
                                                                   teacher_output=output_teacher,
                                                                   labels=labels,
                                                                   alpha=self.alpha,
                                                                   T=self.T)
        accuracy = self.accuracy(output_student,labels)
        mcc = self.mcc(output_student,labels)


        self.log("train_accuracy",accuracy,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_kd_loss",kd_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_cr_loss",cr_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_loss",total_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
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
                 weight_decay=2e-5,
                 momentum=0.9,
                 batch_size=32):
        super().__init__(lr=lr,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         batch_size=batch_size)
        self.efficient_net = efficientnet_b0(weights=None, num_classes=NUM_CLASSES)
    def forward(self,x):
        out = self.efficient_net(x)
        return out

    
class EfficientNet_B0_Pretrained(ImageClassifierBase):
    pass
    
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
        self.efficient_net = efficientnet_v2_s(weights=None, num_classes=NUM_CLASSES)
        self.name = "EfficientNet_V2_S"
    def forward(self,x):
        out = self.efficient_net(x)
        return out
class EfficientNet_V2_S_Pretrained(EfficientNetPretrainedBase):
    def __init__(self,
                 lr,
                 weight_decay,
                 momentum,
                 batch_size,
                 epochs,
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
        self.name = "EfficientNet_V2_S_" + f'{self.training_mode}'
    def init_base_model(self):
        return efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
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
        self.efficient_net = efficientnet_v2_m(weights=None, num_classes=NUM_CLASSES)
        self.name = "EfficientNet_V2_M"
    def forward(self,x):
        out = self.efficient_net(x)
        return out
class EfficientNet_V2_M_Pretrained(EfficientNetPretrainedBase):
    def __init__(self,
                 lr,
                 weight_decay,
                 momentum,
                 batch_size,
                 epochs,
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
        self.name = "EfficientNet_V2_M_" + f'{self.training_mode}'
    def init_base_model(self):
        return efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
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
        self.efficient_net = efficientnet_v2_l(weights=None,num_classes=NUM_CLASSES)
        self.name = "EfficientNet_V2_L"
    def forward(self,x):
        out = self.efficient_net(x)
        return out
    def freeze_layers(self):
        for param in self.efficient_net.parameters():
            param.requires_grad = False
class EfficientNet_V2_L_Pretrained(EfficientNetPretrainedBase):
    def __init__(self,
                 lr,
                 weight_decay,
                 momentum,
                 batch_size,
                 epochs,
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
        
        self.name = "EfficientNet_V2_L_" + f'{self.training_mode}'
    
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
    def __init__(self,lr,weight_decay,batch_size):
        super().__init__(lr=lr,weight_decay=weight_decay,batch_size=batch_size)
        self.name = "Resnet18"
        self.model = resnet18(weights=None, num_classes=NUM_CLASSES)
    def forward(self,x):
        out = self.model(x)
        return out
    
class Resnet_18_Dropout(ImageClassifierBase):
    def __init__(self,lr,weight_decay,batch_size,dropout=0.2):
        super().__init__(lr=lr,weight_decay=weight_decay,batch_size=batch_size)
        self.name = "Resnet18"
        self.dropout = dropout
        self.model = resnet18(weights=None, num_classes=NUM_CLASSES)
        fc_layer = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True),
           fc_layer
        )
    def forward(self,x):
        out = self.model(x)
        return out
class Resnet_101(ImageClassifierBase):
    def __init__(self,lr,weight_decay,batch_size):
        super().__init__(lr=lr,weight_decay=weight_decay,batch_size=batch_size)
        self.name = "Resnet101"
        self.model = resnet101(weights=None, num_classes=NUM_CLASSES)
    def forward(self,x):
        out = self.model(x)
        return out


class NaiveClassifier(ImageClassifierBase):
    def __init__(self,lr,momentum,batch_size):
        super().__init__(lr,momentum,batch_size)
        self.name = "Naive"
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 32, 5)

        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


