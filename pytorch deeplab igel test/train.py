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
import tiles
import utils


# should try: 
# unet++ and pspnet


# Hyperparameters and other model training params
DEVICE = "cuda"
IMAGE_SIZE = 320
CLASSES_N = 4 # 3 classes + 1 background
LEARNING_RATE = 0.001
EPOCHS = 1
BATCH_SIZE = 16

CHECKPOINT_PATH = "checkpoints/"
SAVED_MODEL_NAME = "model"
SAVE_CHECKPOINT_IN_BETWEEN_N_EPOCHS = 1

MODEL_TYPE = "deeplabv3plus_l"
LOAD_STATE_DICT = False
STATE_DICT_PATH = ""

TRAIN_IMAGES_PATH = "cache/images"
TRAIN_MASKS_PATH = "cache/labels"
VAL_IMAGES_PATH = "cache/val_images"
VAL_MASKS_PATH = "cache/val_masks"


def main():
    # Tiling and caching data
    tiles.generate_tile_cache("data//images", "data//masks", "cache", size=IMAGE_SIZE, zero_sampling=0)
    utils.generate_validation_set("cache", 0.1)

    # Defining transformations
    transform_A = A.Compose([
        #A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #A.RandomSizedCrop((IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2), (IMAGE_WIDTH, IMAGE_HEIGHT), p=0.25),
        ToTensorV2()
    ])
    val_transform_A = A.Compose([
        #A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        ToTensorV2()
    ])

    # Loading train dataset
    train_dataset = CustomDataset(image_dir=TRAIN_IMAGES_PATH, mask_dir=TRAIN_MASKS_PATH, transform=transform_A)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Loading validation dataset
    val_dataset = CustomDataset(image_dir=VAL_IMAGES_PATH, mask_dir=VAL_MASKS_PATH, transform=val_transform_A)
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


