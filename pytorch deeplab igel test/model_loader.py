import segmentation_models_pytorch as smp
from torch import nn
import torch
import torchvision


class SegmentationModel(nn.Module):
    """
    Class to load segmentation model
    Arguments:
        model_type (str): 
                    "deeplabv3" - 11M params, mobilenetv3_large bacbone.<br>
                    "deeplabv3plus_s" - 4.7M params, mobilenetv3_large backbone.<br>
                    "deeplabv3plus_l" - 12.3M params, resnet_18 backbone.<br>
    """
    def __init__(self, model_type, classes_n):
        super().__init__()
        self.model_type = model_type
        match self.model_type:
            case "deeplabv3":
                self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
                self.model.backbone["0"][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
                self.model.classifier[4] = torch.nn.Conv2d(256, classes_n, kernel_size=(1, 1))
            case "deeplabv3plus_s":
                self.model = smp.DeepLabV3Plus(encoder_name="timm-mobilenetv3_large_100", encoder_weights="imagenet", in_channels=1, classes=classes_n)
            case "deeplabv3plus_l":
                self.model = smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights="imagenet", in_channels=1, classes=classes_n)
            

    def forward(self, X):
        if self.model_type == "deeplabv3":
            return self.model(X)["out"]
        else:
            return self.model(X)