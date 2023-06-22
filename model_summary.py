from models import EfficientNet_V2_S,EfficientNet_V2_S_Pretrained
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights
from torchvision.datasets import ImageFolder

data = ImageFolder(root='C:/Users/david/Desktop/Python/master/data/train')
print(data)
