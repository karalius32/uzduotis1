import torch
from torch import nn
from model_torch import Model


def main():
    input = torch.rand((16, 3, 320, 320))
    model = Model(in_channels=3, num_of_classes=4)
    outputs = model(input)


if __name__ == "__main__":
    main()