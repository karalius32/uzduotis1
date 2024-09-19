import torch
from torch import nn
import torchvision.transforms.functional as F
import numpy as np


def evaluate_model(model, dataloader, loss_fn, device, class_n):
    model.eval()
    running_loss = 0
    total_iou = 0
    class_n = class_n + 1

    precisions = [0 for _ in range(class_n)]
    recalls = [0 for _ in range(class_n)]
    ious = [0 for _ in range(class_n)]
    dices = [0 for _ in range(class_n)]
    classes_counts = [0 for _ in range(class_n)]

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device).squeeze(1)

            # Forward pass
            model.dont_slice = True
            outputs = model(images)
            model.dont_slice = False
            loss = loss_fn(outputs, masks)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            # Total iou
            total_iou += (torch.where(preds & masks > 0, 1, 0).sum() / torch.where(preds | masks > 0, 1, 0).sum()).item()

            # precision, recall and iou for each class
            for c in range(1, class_n):
                tp = torch.sum(preds[masks == c] == c).item()
                fn = torch.sum(preds[masks == c] != c).item()
                fp = torch.sum(masks[preds == c] != c).item()

                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                iou = (((preds == c) & (masks == c)).sum() / (((preds == c) | (masks == c)).sum() + 1e-6)).item()
                dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)

                precisions[c] += precision
                recalls[c] += recall
                ious[c] += iou
                dices[c] += dice

                if (torch.sum(masks == c) > 0):
                    classes_counts[c] += 1

    for c in range(1, class_n):
        precisions[c] = precisions[c] / classes_counts[c]
        recalls[c] = recalls[c] / classes_counts[c]
        ious[c] = ious[c] / classes_counts[c]
        dices[c] = dices[c] / classes_counts[c]
    avg_loss = running_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return precisions, recalls, ious, dices, avg_loss, avg_iou


class PlateuChecker:
    def __init__(self, stop_patience=15, lr_decay_patience=7, eps=1e-3):
        self.stop_patience = stop_patience
        self.decay_patience = lr_decay_patience
        self.eps = eps
        self.stop_counter = 0
        self.decay_counter = 0
        self.last_loss = float("inf")

    def check_plateu(self, training_loss):
        print(f"\n{self.last_loss - training_loss}")
        print(f"{self.last_loss - self.eps - training_loss}")
        print(self.stop_counter)
        print(self.decay_counter)
        if self.last_loss - self.eps < training_loss:
            self.stop_counter += 1
            self.decay_counter += 1
        else:
            self.stop_counter = 0
            self.decay_counter = 0
        self.last_loss = training_loss
        if self.stop_counter >= self.stop_patience:
            return {"stop": True, "decay": False}
        elif self.decay_counter >= self.decay_patience:
            self.decay_counter = 0
            return {"stop": False, "decay": True}
        else: 
            return {"stop": False, "decay": False}


class MosaicTransform(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    
    def forward(self, images_batch, masks_batch):
        images_mosaic, masks_mosaic = [], []
        n = len(images_batch) // 4

        for i in range(n):
            images, masks = images_batch[i*4:(i+1)*4], masks_batch[i*4:(i+1)*4]
            images, masks = F.resize(images, (self.size // 2, self.size // 2)).squeeze(1), F.resize(masks, (self.size // 2, self.size // 2)).squeeze(1)
            # Generate mosaic
            h_stack_image0, h_stack_mask0 = torch.hstack((images[0], images[1])), torch.hstack((masks[0], masks[1]))
            h_stack_image1, h_stack_mask1 = torch.hstack((images[2], images[3])), torch.hstack((masks[2], masks[3]))
            full_stack_image, full_stack_mask = torch.vstack((h_stack_image0, h_stack_image1)), torch.vstack((h_stack_mask0, h_stack_mask1))
            # roll
            size_divided_by_3 = self.size // 3
            shift_size_x = np.random.randint(-size_divided_by_3, size_divided_by_3)
            shift_size_y = np.random.randint(-size_divided_by_3, size_divided_by_3)
            full_stack_image = torch.roll(full_stack_image, shifts=(shift_size_x, shift_size_y), dims=(0, 1))
            full_stack_mask = torch.roll(full_stack_mask, shifts=(shift_size_x, shift_size_y), dims=(0, 1))

            images_mosaic.append(full_stack_image.unsqueeze(0))
            masks_mosaic.append(full_stack_mask.unsqueeze(0))

        return torch.cat(images_mosaic, axis=0).unsqueeze(1), torch.cat(masks_mosaic, axis=0)


            


            