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
import sys 
import json
import pandas as pd
from segmentation_models_pytorch.losses import DiceLoss



def main(config):
    # Tiling and caching data
    tiles.generate_tile_cache(config["train_images_path"], config["train_masks_path"], "cache", size=config["image_size"], zero_sampling=0)
    if config["do_validation"]:
        tiles.generate_tile_cache(config["val_images_path"], config["val_masks_path"], "cache_val", size=config["image_size"], zero_sampling=0)

    # Defining transformations
    transform_A = A.Compose([
        #A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=1),
        A.GaussianBlur(p=0.25),
        A.RandomSizedCrop((config["image_size"] / 2, config["image_size"] / 2), (config["image_size"], config["image_size"]), p=0.25),
        ToTensorV2()
    ])
    val_transform_A = A.Compose([
        #A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        ToTensorV2()
    ])

    # Loading train dataset
    train_dataset = CustomDataset(image_dir="cache/images", mask_dir="cache/labels", transform=transform_A)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle_dataset"])

    # Loading validation dataset
    if config["do_validation"]:
        val_dataset = CustomDataset(image_dir="cache_val/images", mask_dir="cache_val/labels", transform=val_transform_A)
        val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Loading model and changing input, output shapes
    model = model_loader.SegmentationModel(config["model_type"], config["encoder"], config["classes_n"], config["use_background"])
    if config["load_state_dict"]:
        model.load_state_dict(torch.load(config["state_dict_path"]))

    model.to(config["device"])

    # Defining loss function and optimizer
    if config["loss"] == "dice":
        if config["classes_n"] > 1 or config["use_background"]:
            mode = "multiclass"
        else:
            mode = "binary"
        loss_fn = DiceLoss(mode, from_logits=False)
    elif config["loss"] == "crossentropy":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Training loop
    history = {"loss": [], "iou": [], "val_loss": [], "val_iou": []}
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config["epochs"]):
        start = time.time()
        model.train()
        running_loss = 0
        total_iou = 0

        for images, masks in train_dataloader:
            images, masks = images.to(config["device"]), masks.to(config["device"]).squeeze(1)
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
            total_iou += (torch.where(preds & masks > 0, 1, 0).sum() / torch.where(preds | masks > 0, 1, 0).sum()).item()

            running_loss += loss.item()

            print("#", end="", flush=True)
        
        avg_train_loss = running_loss / len(train_dataloader)
        avg_train_iou = total_iou / len(train_dataloader)

        end = time.time()
        print(f"\nEpoch {epoch+1}/{config['epochs']}. Time elapsed: {(end - start):.2f}s")
        print(f"    Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}")
        history["loss"].append(avg_train_loss)
        history["iou"].append(avg_train_iou)

        if epoch == 0 or (epoch+1) % config["save_checkpoint_in_between_n_epochs"] == 0:
            torch.save(model.state_dict(), os.path.join(config["checkpoint_path"], f"{config['checkpoint_name']}{epoch+1}.pth"))

            # Evaluation
            if config["do_validation"]:
                precisions, recalls, ious, dices, avg_val_loss, avg_val_iou = evaluate_model(model, val_dataloader, loss_fn, config["device"], config["classes_n"], config["batch_size"])
                print(f"    Validation Loss: {avg_val_loss:.4f}, Validation IoU: {avg_val_iou:.4f}")
                print(f"    Metrics by class:")
                print(f"        IoU:        {[round(x, 2) for x in ious[1:]]}")
                print(f"        Dice (F1):  {[round(x, 2) for x in dices[1:]]}")
                print(f"        Precision:  {[round(x, 2) for x in precisions[1:]]}")
                print(f"        Recall:     {[round(x, 2) for x in recalls[1:]]}")
                history["val_loss"].append(avg_val_loss)
                history["val_iou"].append(avg_val_iou)

            pd.DataFrame(history).to_csv(os.path.join(config["history_path"], f"{config['checkpoint_name']}_{epoch+1}.csv"))

            

if __name__ == "__main__":
    config = {}
    with open(sys.argv[1]) as config_file:
        config = json.load(config_file)
    main(config)


