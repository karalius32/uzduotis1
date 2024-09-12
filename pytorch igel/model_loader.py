import segmentation_models_pytorch as smp
from torch import nn


class SegmentationModel(nn.Module):
    def __init__(self, model_type, encoder, classes_n, use_background):
        """
        model_type:
            - deeplabv3plus
            - pspnet
        encoder:
            - resnet18
            - timm-regnety_002
            - mit_b0
            - tu-mobilevitv2_100
            - timm-mobilenetv3_small_100
        """
        super().__init__()
        self.model_type = model_type
        self.encoder = encoder
        self.use_background = use_background
        if use_background:
            classes_n += 1
        match self.model_type:
            case "deeplabv3plus":
                self.model = smp.DeepLabV3Plus(encoder_name=self.encoder, encoder_weights="imagenet", in_channels=1, classes=classes_n, activation="softmax")
            case "pspnet":
                self.model = smp.PSPNet(encoder_name=self.encoder, encoder_weights="imagenet", in_channels=1, classes=classes_n, activation="softmax")
            

    def forward(self, X):
        outputs = self.model(X)
        if self.training or not self.use_background:
            return outputs
        else:
            return outputs[:, 1:, :, :]