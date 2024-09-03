from torch.utils.data import DataLoader
from dataset import CustomDataset
import torchvision.transforms as transforms
import torch
from torch import nn
import os
from matplotlib import pyplot as plt
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import model_loader
from utils import evaluate_model

# should try: 
# https://github.com/XuJiacong/PIDNet
# https://github.com/Fourier7754/AsymFormer


# deeplabv3: Validation Loss: 0.0159, Validation IoU: 0.9680
# deeplabv3plus_l: Validation Loss: 0.0111, Validation IoU: 0.9782
# deeplabv3plus_s: Validation Loss: 0.0143, Validation IoU: 0.9700


# Hyperparameters and other model training params
DEVICE = "cuda"
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 960
CLASSES_N = 2 # 0 - background, 1 - label
LEARNING_RATE = 0.001
EPOCHS = 500
BATCH_SIZE = 4

CHECKPOINT_PATH = "saved_models/"
SAVED_MODEL_NAME = "model_unetplusplus_"
SAVE_CHECKPOINT_IN_BETWEEN_N_EPOCHS = 10

MODEL_TYPE = "unetplusplus"
LOAD_STATE_DICT = False
STATE_DICT_PATH = "saved_models/model_pspnet50_BEST.pth.tar"


def main():
    # Defining transformations
    transform_A = A.Compose([
        A.Resize(IMAGE_WIDTH, IMAGE_HEIGHT),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomSizedCrop((IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2), (IMAGE_WIDTH, IMAGE_HEIGHT), p=0.25),
        ToTensorV2()
    ])
    val_transform_A = A.Compose([
        A.Resize(IMAGE_WIDTH, IMAGE_HEIGHT),
        ToTensorV2()
    ])

    # Loading train dataset
    train_dataset = CustomDataset(image_dir="data/train_images", mask_dir="data/train_masks", transform=transform_A)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Loading validation dataset
    val_dataset = CustomDataset(image_dir="data/val_images", mask_dir="data/val_masks", transform=val_transform_A)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Loading model and changing input, output shapes
    model = model_loader.SegmentationModel(MODEL_TYPE, CLASSES_N)
    if LOAD_STATE_DICT:
        model.load_state_dict(torch.load(STATE_DICT_PATH))

    model.to(DEVICE)

    # Defining loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        running_loss = 0
        total_iou = 0

        for images, masks in train_dataloader:
            images, masks = images.to(DEVICE), masks.to(DEVICE).squeeze(1)
            optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Calculate IoU
            preds = torch.argmax(outputs, dim=1)
            iou = (preds & masks).sum() / (preds | masks).sum()
            total_iou += iou.item()

            running_loss += loss.item()

            print("#", end="")
        
        avg_train_loss = running_loss / len(train_dataloader)
        avg_train_iou = total_iou / len(train_dataloader)
        if (epoch+1) % SAVE_CHECKPOINT_IN_BETWEEN_N_EPOCHS == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f"{SAVED_MODEL_NAME}{epoch+1}.pth.tar"))

        # Evaluation
        avg_val_loss, avg_val_iou = evaluate_model(model, val_dataloader, loss_fn, DEVICE)

        end = time.time()

        print(f"\nEpoch {epoch+1}/{EPOCHS}. Time elapsed: {(end - start):.2f}s")
        print(f"    Training Loss: {avg_train_loss:.4f}, Training IoU: {avg_train_iou:.4f}")
        print(f"    Validation Loss: {avg_val_loss:.4f}, Validation IoU: {avg_val_iou:.4f}")
            

if __name__ == "__main__":
    main()


