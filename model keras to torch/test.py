import torch
from torch import nn


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3)
    
    def forward(self, input):
        flow = self.conv(input)
        res = flow
        flow = self.max_pool(flow)
        return res, flow
    

def main():
    input = torch.rand((16, 3, 320, 320))
    model = Module()
    res, flow = model(input)
    print(res.shape)
    print(flow.shape)


if __name__ == "__main__":
    main()