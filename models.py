from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch 
import torch.nn.functional as F
from torchvision.models import efficientnet_b0,efficientnet_v2_m
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import Accuracy,F1Score,MatthewsCorrCoef
from lion_pytorch import Lion

NUM_CLASSES = 525

class ImageClassifierBase(pl.LightningModule):
    def __init__(self,lr,weight_decay,batch_size):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.f1 = F1Score(task="multiclass",num_classes=NUM_CLASSES)
        self.mcc = MatthewsCorrCoef(task="multiclass",num_classes=NUM_CLASSES)

    def configure_optimizers(self) -> Any:
        optimizer = Lion(self.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        return optimizer
    
    def training_step(self, batch,batch_idx):
        inputs, labels = batch['image'], batch['class_id']
        output = self(inputs)
        loss = F.cross_entropy(input=output,target
                               =labels)
        accuracy = self.accuracy(output,labels)
        f1_score = self.f1(output,labels)
        mcc = self.mcc(output,labels)

        self.log("train_accuracy",accuracy,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_loss",loss,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_f1_score",f1_score,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("train_mcc",mcc,on_step=True,on_epoch=True,prog_bar=True,logger=True,batch_size=self.batch_size)
        return loss
    
    def validation_step(self,batch,batch_idx):
        inputs, labels = batch['image'], batch['class_id']
        output = self(inputs)
        loss = F.cross_entropy(input=output,target=labels)
        accuracy = self.accuracy(output,labels)
        f1_score = self.f1(output,labels)
        mcc = self.mcc(output,labels)

        self.log("validation_accuracy",accuracy,prog_bar=True,logger=True,batch_size=self.batch_size),
        self.log("validation_loss",loss,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("validation_f1_score",f1_score,prog_bar=True,logger=True,batch_size=self.batch_size)
        self.log("validation_mcc",mcc,prog_bar=True,logger=True,batch_size=self.batch_size)
        return loss
    
class EfficientNet_B0(ImageClassifierBase):
    def __init__(self,lr,weight_decay,batch_size):
        super().__init__(lr,weight_decay,batch_size)
        self.efficient_net = efficientnet_b0(weights=None, num_classes=NUM_CLASSES)
    def forward(self,x):
        out = self.efficient_net(x)
        return out
    
class EfficientNet_V2_M(ImageClassifierBase):
    def __init__(self,lr,weight_decay,batch_size):
        super().__init__(lr,weight_decay,batch_size)
        self.efficient_net = efficientnet_v2_m(weights=None, num_classes=NUM_CLASSES)
    def forward(self,x):
        out = self.efficient_net(x)
        return out

class NaiveClassifier(ImageClassifierBase):
    def __init__(self,lr,momentum,batch_size):
        super().__init__(lr,momentum,batch_size)
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