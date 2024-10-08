import segmentation_models_pytorch as smp
from torch import nn
from model_torch import Model as CustomModel


class SegmentationModel(nn.Module):
    def __init__(self, model_type, encoder, classes_n, use_background, dont_slice=False):
        """
        model_type:
            - deeplabv3plus
            - pspnet
            - manet
            - pan
            - custom
        encoder:
            - resnet18
            - timm-mobilenetv3_small_100
            - tu-mobilevitv2_100
            - tu-mobilevitv2_050
            - timm-regnety_002
            - mit_b0
        """
        super().__init__()
        self.model_type = model_type
        self.encoder = encoder
        self.use_background = use_background
        self.dont_slice = dont_slice
        if use_background:
            classes_n += 1
        match self.model_type:
            case "deeplabv3plus":
                self.model = smp.DeepLabV3Plus(encoder_name=self.encoder, encoder_weights="imagenet", in_channels=1, classes=classes_n, activation="softmax")
            case "pspnet":
                self.model = smp.PSPNet(encoder_name=self.encoder, encoder_weights="imagenet", in_channels=1, classes=classes_n, activation="softmax")
            case "manet":
                self.model = smp.MAnet(encoder_name=self.encoder, encoder_weights="imagenet", in_channels=1, classes=classes_n, activation="softmax")
            case "pan":
                self.model = smp.PAN(encoder_name=self.encoder, encoder_weights="imagenet", in_channels=1, classes=classes_n, activation="softmax")
            case "custom":
                self.model = CustomModel(in_channels=1, num_of_classes=classes_n)
            

    def forward(self, X):
        outputs = self.model(X)
        if self.training or not self.use_background or self.dont_slice:
            return outputs
        else:
            
            return outputs[:, 1:, :, :]