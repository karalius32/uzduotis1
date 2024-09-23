import torch
from torch import nn
from model_torch import Model


def main():
    input = torch.rand((13, 1, 320, 320)).to("cuda")
    model = Model(in_channels=1, num_of_classes=4).to("cuda")
    outputs = model(input)
    print(outputs.shape)


if __name__ == "__main__":
    main()