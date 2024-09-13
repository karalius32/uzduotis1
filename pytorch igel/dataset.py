import os 
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from matplotlib import pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, use_background, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.use_background = use_background
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))
        image = np.array(Image.open(img_path).convert("L")) / 255
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = image.to(torch.float32)
        mask = mask.long()
        if not self.use_background:
            mask -= 1

        return image, mask