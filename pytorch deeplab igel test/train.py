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


# should try: 
# unet++ and pspnet

""" deeplabv3_r18
Epoch 6/10. Time elapsed: 32.18s
    Training Loss: 0.0077, Training IoU: 0.7097
    Validation Loss: 0.0214, Validation IoU: 0.5455
    Metrics by class:
        IoU:        [0.19, 0.51, 0.44]
        Dice (F1):  [0.31, 0.65, 0.6]
        Precision:  [0.64, 0.67, 0.66]
        Recall:     [0.24, 0.67, 0.58]
"""
""" pspnet
Epoch 980/1000. Time elapsed: 22.35s
    Training Loss: 0.0058, Training IoU: 0.7140
    Validation Loss: 0.0253, Validation IoU: 0.4821
    Metrics by class:
        IoU:        [0.19, 0.5, 0.27]
        Dice (F1):  [0.28, 0.65, 0.41]
        Precision:  [0.52, 0.73, 0.61]
        Recall:     [0.23, 0.6, 0.34]
"""



"""# Hyperparameters and other model training params
DEVICE = "cuda"
IMAGE_SIZE = 320
CLASSES_N = 4 # 3 classes + 1 background
LEARNING_RATE = 0.001
EPOCHS = 1000
BATCH_SIZE = 32

CHECKPOINT_PATH = "checkpoints/"
SAVED_MODEL_NAME = "model_pspnet"
SAVE_CHECKPOINT_IN_BETWEEN_N_EPOCHS = 10

MODEL_TYPE = "pspnet_r18"
LOAD_STATE_DICT = False
STATE_DICT_PATH = "checkpoints/model_BEST.pth"

TRAIN_IMAGES_PATH = "cache/images"
TRAIN_MASKS_PATH = "cache/labels"
VAL_IMAGES_PATH = "cache_val/images"
VAL_MASKS_PATH = "cache_val/labels"
"""



def main(config):
    # Tiling and caching data
    tiles.generate_tile_cache("data//images", "data//masks", "cache", size=config["image_size"], zero_sampling=0)
    if config["do_validation"]:
        tiles.generate_tile_cache("validation//images", "validation//masks", "cache_val", size=config["image_size"], zero_sampling=0)

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
    train_dataset = CustomDataset(image_dir=config["train_images_path"], mask_dir=config["train_masks_path"], transform=transform_A)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle_dataset"])

    # Loading validation dataset
    if config["do_validation"]:
        val_dataset = CustomDataset(image_dir=config["val_images_path"], mask_dir=config["val_masks_path"], transform=val_transform_A)
        val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Loading model and changing input, output shapes
    model = model_loader.SegmentationModel(config["model_type"], config["encoder"], config["classes_n"])
    if config["load_state_dict"]:
        model.load_state_dict(torch.load(config["state_dict_path"]))

    model.to(config["device"])

    # Defining loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Training loop
    history = {"loss": [], "iou": []}
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
            pd.DataFrame(history).to_csv(f"histories/history_{config['checkpoint_name']}_{epoch+1}.csv")

            # Evaluation
            if config["do_validation"]:
                precisions, recalls, ious, dices, avg_val_loss, avg_val_iou = evaluate_model(model, val_dataloader, loss_fn, config["device"], config["classes_n"], config["batch_size"])
                print(f"    Validation Loss: {avg_val_loss:.4f}, Validation IoU: {avg_val_iou:.4f}")
                print(f"    Metrics by class:")
                print(f"        IoU:        {[round(x, 2) for x in ious[1:]]}")
                print(f"        Dice (F1):  {[round(x, 2) for x in dices[1:]]}")
                print(f"        Precision:  {[round(x, 2) for x in precisions[1:]]}")
                print(f"        Recall:     {[round(x, 2) for x in recalls[1:]]}")


            

if __name__ == "__main__":
    config = {}
    with open(sys.argv[1]) as config_file:
        config = json.load(config_file)
    main(config)


