from torch.utils.data import DataLoader
from dataset import CustomDataset
import torchvision.transforms as transforms
import torch
from torch import nn
import torchvision
import os


# Hyperparameters
DEVICE = "cuda"
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 960
CLASSES_N = 2 # 0 - background, 1 - label
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 4
CHECKPOINT_PATH = "saved_models/"


def evaluate_model(model, dataloader, loss_fn):
    model.eval()
    running_loss = 0
    total_iou = 0
    total_samples = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(DEVICE), masks.to(DEVICE).long().squeeze(1)

            # Forward pass
            outputs = model(images)["out"]
            loss = loss_fn(outputs, masks)
            running_loss += loss.item()

            # Calculate IoU
            preds = torch.argmax(outputs, dim=1)
            iou = (preds & masks).float().sum() / (preds | masks).float().sum()
            total_iou += iou.item()
            total_samples += images.size(0)
    
    avg_loss = running_loss / len(dataloader)
    avg_iou = total_iou / total_samples

    return avg_loss, avg_iou


def main():
    # Defining transformations. Would be good to add augmentations.
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
    ])

    # Loading train dataset
    train_dataset = CustomDataset(image_dir="data/train_images", mask_dir="data/train_masks", transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Loading validation dataset
    val_dataset = CustomDataset(image_dir="data/val_images", mask_dir="data/val_masks", transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Loading model and changing input, output shapes
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=True)
    model.backbone["0"][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[4] = torch.nn.Conv2d(256, CLASSES_N, kernel_size=(1, 1))
    model.to(DEVICE)

    # Defining loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for images, masks in train_dataloader:
            images, masks = images.to(DEVICE), masks.to(DEVICE).long().squeeze(1)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)["out"]
            loss = loss_fn(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print("#", end="")
        
        avg_train_loss = running_loss / len(train_dataloader)
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f"model{epoch}.pth.tar"))

        # Evaluation
        avg_val_loss, avg_val_iou = evaluate_model(model, val_dataloader, loss_fn)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"    Training Loss: {avg_train_loss:.4f}")
        print(f"    Validation Loss: {avg_val_loss:.4f}, Validation IoU: {avg_val_iou:.4f}")
            

if __name__ == "__main__":
    main()


