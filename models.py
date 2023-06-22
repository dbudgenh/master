from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch 
import torch.nn.functional as F
from torchvision.models import efficientnet_b0,efficientnet_v2_s,efficientnet_v2_m,efficientnet_v2_l,EfficientNet_V2_S_Weights,EfficientNet_B0_Weights,EfficientNet_V2_L_Weights
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import Accuracy,F1Score,MatthewsCorrCoef
from lion_pytorch import Lion
from abc import ABC
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,ExponentialLR,ReduceLROnPlateau,StepLR

NUM_CLASSES = 525



class ImageClassifierBase(ABC,pl.LightningModule):
    def __init__(self,lr,weight_decay,batch_size):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        #self.f1 = F1Score(task="multiclass",num_classes=NUM_CLASSES)
        self.mcc = MatthewsCorrCoef(task="multiclass",num_classes=NUM_CLASSES)
        self.wrong_classifications = []

    def configure_optimizers(self) -> Any:
        #optimizer = Lion(self.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        optimizer = AdamW(self.parameters(),lr=self.lr,weight_decay=self.weight_decay)

        #scheduler = ExponentialLR(optimizer, gamma=0.95, verbose=False)
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2,verbose=False)
        #scheduler = CosineAnnealingLR(optimizer,T_max=600)
        #scheduler =  StepLR(optimizer,step_size=7,gamma=0.1)
        #scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=2)
        return [optimizer],[{"scheduler": scheduler}] #,'monitor':'validation_loss'
    
    def training_step(self, batch,batch_idx):
        inputs, labels = batch
        output = self(inputs)
        loss = F.cross_entropy(input=output,target
                               =labels)
        accuracy = self.accuracy(output,labels)
        #f1_score = self.f1(output,labels)
        mcc = self.mcc(output,labels)

        self.log("train_accuracy",accuracy,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_loss",loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        #self.log("train_f1_score",f1_score,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_mcc",mcc,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        return loss
    
    def validation_step(self,batch,batch_idx):
        inputs, labels = batch
        output = self(inputs)
        loss = F.cross_entropy(input=output,target=labels)
        accuracy = self.accuracy(output,labels)
        #f1_score = self.f1(output,labels)
        mcc = self.mcc(output,labels)

        self.log("validation_accuracy",accuracy,prog_bar=True,logger=True,batch_size=self.batch_size),
        self.log("validation_loss",loss,prog_bar=True,logger=True,batch_size=self.batch_size)
        #self.log("validation_f1_score",f1_score,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("validation_mcc",mcc,prog_bar=True,logger=True,batch_size=self.batch_size)
        return loss
    
    def test_step(self,batch,batch_idx):
        inputs, labels = batch
        output = self(inputs)
        loss = F.cross_entropy(input=output,target=labels)
        accuracy = self.accuracy(output,labels)
        #f1_score = self.f1(output,labels)
        mcc = self.mcc(output,labels)
        predictions = torch.argmax(output,dim=1)
        self.log("test_loss",loss,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("test_mcc",mcc,prog_bar=True,logger=True,batch_size=self.batch_size)
        return loss

class PreTrainedBase(ImageClassifierBase,ABC):
    def __init__(self,lr,weight_decay,batch_size,mode='pre_train'):
        super().__init__(lr,weight_decay,batch_size) 
        self.mode = mode

class EfficientNet_B0(ImageClassifierBase):
    def __init__(self,lr,weight_decay,batch_size):
        super().__init__(lr,weight_decay,batch_size)
        self.efficient_net = efficientnet_b0(weights=None, num_classes=NUM_CLASSES)
    def forward(self,x):
        out = self.efficient_net(x)
        return out

    
class EfficientNet_B0_Pretrained(ImageClassifierBase):
    def __init__(self,lr,weight_decay,batch_size,mode='pre_train'):
        super().__init__(lr,weight_decay,batch_size)
        self.efficient_net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.mode = mode
        self.freeze_layers()
        #Change final classification layer
        in_features = in_features=self.efficient_net.classifier[1].in_features
        self.efficient_net.classifier[1] = nn.Linear(in_features=in_features,out_features=NUM_CLASSES)
        self.name = "EfficientNet_B0_Pretrained"

    def configure_optimizers(self) -> Any:
        if self.mode == 'pre_train':
            optimizer = AdamW(self.parameters(),lr=self.lr,weight_decay=self.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=4)
        if self.mode == 'fine_tune':
            optimizer = AdamW(self.parameters(),lr=self.lr,weight_decay=self.weight_decay)
            scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2,verbose=False)    
        return [optimizer],[{"scheduler": scheduler}]

    def forward(self,x):
        out = self.efficient_net(x)
        return out
    
    def freeze_layers(self):
        #Freeze all layers
        for param in self.efficient_net.parameters():
            param.requires_grad = False
    def unfreeze_layers(self):
        for param in self.efficient_net.parameters():
            param.requires_grad = True
    
class EfficientNet_V2_S(ImageClassifierBase):
    def __init__(self,lr,weight_decay,batch_size):
        super().__init__(lr,weight_decay,batch_size)
        self.efficient_net = efficientnet_v2_s(weights=None, num_classes=NUM_CLASSES)
        self.name = "EfficientNet_V2_S"
    def forward(self,x):
        out = self.efficient_net(x)
        return out
    
    
class EfficientNet_V2_S_Pretrained(ImageClassifierBase):
    def __init__(self,lr,weight_decay,batch_size,mode='pre_train'):
        print(f"Initializing model with lr={lr} weight_decay={weight_decay} batch_size={batch_size} mode={mode}")
        super().__init__(lr,weight_decay,batch_size)
        self.efficient_net = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.mode = mode
        self.freeze_layers()
        #Change final classification layer
        in_features = in_features=self.efficient_net.classifier[1].in_features
        self.efficient_net.classifier[1] = nn.Linear(in_features=in_features,out_features=NUM_CLASSES)
        self.name = "EfficientNet_V2_S_Pretrained"

    def configure_optimizers(self) -> Any:
        if self.mode == 'pre_train':
            optimizer = AdamW(self.parameters(),lr=self.lr,weight_decay=self.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=4)
            return [optimizer],[{"scheduler": scheduler,'monitor':'validation_loss'}]
        if self.mode == 'fine_tune':
            optimizer = AdamW(self.parameters(),lr=self.lr,weight_decay=self.weight_decay)
            scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=100,T_mult=2,verbose=False)
            return [optimizer],[{"scheduler": scheduler,'interval':'step'}]    

    def forward(self,x):
        out = self.efficient_net(x)
        return out
    
    def freeze_layers(self):
        #Freeze all layers
        for param in self.efficient_net.parameters():
            param.requires_grad = False
    def unfreeze_layers(self):
        for param in self.efficient_net.parameters():
            param.requires_grad = True

class EfficientNet_V2_M(ImageClassifierBase):
    def __init__(self,lr,weight_decay,batch_size):
        super().__init__(lr,weight_decay,batch_size)
        self.efficient_net = efficientnet_v2_m(weights=None, num_classes=NUM_CLASSES)
        self.name = "EfficientNet_V2_M"
    def forward(self,x):
        out = self.efficient_net(x)
        return out
    
class EfficientNet_V2_L(ImageClassifierBase):
    def __init__(self,lr,weight_decay,batch_size):
        super().__init__(lr,weight_decay,batch_size)
        self.efficient_net = efficientnet_v2_l(weights=None,num_classes=NUM_CLASSES)
        self.name = "EfficientNet_V2_L"
    def forward(self,x):
        out = self.efficient_net(x)
        return out

class EfficientNet_V2_L_Pretrained(ImageClassifierBase):
    def __init__(self,lr,weight_decay,batch_size,mode='pre_train'):
        print(f"Initializing model with lr={lr} weight_decay={weight_decay} batch_size={batch_size} mode={mode}")
        super().__init__(lr,weight_decay,batch_size)
        self.efficient_net = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
        self.mode = mode
        self.freeze_layers()
        #Change final classification layer
        in_features = in_features=self.efficient_net.classifier[1].in_features
        self.efficient_net.classifier[1] = nn.Linear(in_features=in_features,out_features=NUM_CLASSES)
        self.name = "EfficientNet_V2_L_Pretrained"

    def configure_optimizers(self) -> Any:
        if self.mode == 'pre_train':
            optimizer = AdamW(self.parameters(),lr=self.lr,weight_decay=self.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=4)
            return [optimizer],[{"scheduler": scheduler,'monitor':'validation_loss'}]
        if self.mode == 'fine_tune':
            optimizer = AdamW(self.parameters(),lr=self.lr,weight_decay=self.weight_decay)
            scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=100,T_mult=2,verbose=False)
            return [optimizer],[{"scheduler": scheduler,'interval':'step'}]    
        

    def forward(self,x):
        out = self.efficient_net(x)
        return out
    
    def freeze_layers(self):
        #Freeze all layers
        for param in self.efficient_net.parameters():
            param.requires_grad = False
    def unfreeze_layers(self):
        for param in self.efficient_net.parameters():
            param.requires_grad = True
    
class VisionTransformer_B_16(ImageClassifierBase): 
    pass
class VisionTransformer_L_16(ImageClassifierBase): 
    pass
class VisionTransformer_H_14(ImageClassifierBase):
    pass

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
    


