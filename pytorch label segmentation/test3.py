import numpy as np
from matplotlib import pyplot as plt
import torch

prototypes = torch.load("prototypes.pt").state_dict()["0"][1]
prototypes = np.array(prototypes.cpu())

num_images = len(prototypes)
num_cols = 8  # Change this according to your preference
num_rows = (num_images + num_cols - 1) // num_cols

# Plot the images
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    if i < num_images:
        ax.imshow(prototypes[i], cmap='gray')
        ax.axis('off')
    else:
        ax.axis('off')  # Hide axes for empty subplots

plt.tight_layout()
plt.show()