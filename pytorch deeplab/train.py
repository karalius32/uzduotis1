from torch.utils.data import DataLoader
from dataset import CustomDataset
import torchvision.transforms as transforms
import torch
from torch import nn
import torchvision
import os
from matplotlib import pyplot as plt
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Hyperparameters
DEVICE = "cuda"
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 960
CLASSES_N = 2 # 0 - background, 1 - label
LEARNING_RATE = 0.001
EPOCHS = 500
BATCH_SIZE = 4
CHECKPOINT_PATH = "saved_models/"
LOAD_STATE_DICT = True
STATE_DICT_PATH = "saved_models/best.pth.tar"


def evaluate_model(model, dataloader, loss_fn):
    model.eval()
    running_loss = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(DEVICE), masks.to(DEVICE).squeeze(1)

            # Forward pass
            outputs = model(images)["out"]
            loss = loss_fn(outputs, masks)
            running_loss += loss.item()

            # Calculate IoU
            preds = torch.argmax(outputs, dim=1)
            iou = (preds & masks).sum() / (preds | masks).sum()
            total_iou += iou.item()
    
    avg_loss = running_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_iou


def main():
    # Defining transformations
    transform = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
    ])
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
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    model.backbone["0"][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[4] = torch.nn.Conv2d(256, CLASSES_N, kernel_size=(1, 1))
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
                outputs = model(images)["out"]
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
        if (epoch+1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f"model{epoch+1}.pth.tar"))

        # Evaluation
        avg_val_loss, avg_val_iou = evaluate_model(model, val_dataloader, loss_fn)

        end = time.time()

        print(f"\nEpoch {epoch+1}/{EPOCHS}. Time elapsed: {(end - start):.2f}s")
        print(f"    Training Loss: {avg_train_loss:.4f}, Training IoU: {avg_train_iou:.4f}")
        print(f"    Validation Loss: {avg_val_loss:.4f}, Validation IoU: {avg_val_iou:.4f}")
            

if __name__ == "__main__":
    main()


