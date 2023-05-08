from models import EfficientNet_B0,NaiveClassifier
from torchvision.models import efficientnet_b0

def get_model(model_name:str ):
    model = model_name.lower()
    if model == 'naive':
        return NaiveClassifier()
    if model == 'efficientnet_b0':
        return efficientnet_b0(weights=None,num_classes=525)
    if model == 'resnet50':
        return None
    raise NotImplementedError

