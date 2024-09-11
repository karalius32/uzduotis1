import segmentation_models_pytorch as smp
from torch import nn


class SegmentationModel(nn.Module):
    def __init__(self, model_type, encoder, classes_n):
        """
        model_type:
            - deeplabv3plus
            - pspnet
        encoder:
            - resnet18
            - timm-regnety_002
            - mit_b0
            - tu-mobilevitv2_100
        """
        super().__init__()
        self.model_type = model_type
        self.encoder = encoder
        match self.model_type:
            case "deeplabv3plus":
                self.model = smp.DeepLabV3Plus(encoder_name=self.encoder, encoder_weights="imagenet", in_channels=1, classes=classes_n)
            case "pspnet":
                self.model = smp.PSPNet(encoder_name=self.encoder, encoder_weights="imagenet", in_channels=1, classes=classes_n)
            

    def forward(self, X):
        return self.model(X)