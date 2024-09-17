from torch.utils.data import DataLoader
from dataset import CustomDataset
import torchvision.transforms as transforms
import torch
from torch import nn
import os
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import model_loader
from train_utils import evaluate_model, EarlyStopper
import tiles
import sys 
import json
import pandas as pd
from segmentation_models_pytorch.losses import DiceLoss
import utils



def main(config):
    # Defining transformations
    transform_A = A.Compose([
        #A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=1),
        A.GaussianBlur(p=0.25),
        A.RandomSizedCrop((config['image_size'] / 2, config['image_size'] / 2), (config['image_size'], config['image_size']), p=0.25),
        ToTensorV2()
    ])
    val_transform_A = A.Compose([
        #A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        ToTensorV2()
    ])

    # Loading train dataset
    train_dataset = CustomDataset(image_dir=config['train_images_path'], mask_dir=config['train_masks_path'], transform=transform_A)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle_dataset'])

    # Loading validation dataset
    if config['do_validation']:
        val_dataset = CustomDataset(image_dir=config['val_images_path'], mask_dir=config['val_masks_path'], transform=val_transform_A)
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Loading model
    model = model_loader.SegmentationModel(config['model_type'], config['encoder'], config['classes_n'], config['use_background'])
    if config['load_state_dict']:
        model.load_state_dict(torch.load(config['state_dict_path']))
    model.to(config['device'])

    # Defining loss function
    if config['loss'] == "dice":
        if config['classes_n'] > 1 or config['use_background']:
            mode = "multiclass"
        else:
            mode = "binary"
        loss_fn = DiceLoss(mode, from_logits=False)
    elif config['loss'] == "crossentropy":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Defining optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Training loop
    history = {"loss": [], "iou": [], "val_loss": [], "val_iou": []}
    scaler = torch.cuda.amp.GradScaler()
    earlyStopper = EarlyStopper()
    for epoch in range(config['epochs']):
        start = time.time()
        model.train()
        running_loss = 0
        total_iou = 0

        for images, masks in train_dataloader:
            images, masks = images.to(config['device']), masks.to(config['device']).squeeze(1)
            optimizer.zero_grad()

            # Forward pass
            with torch.autocast(device_type=config['device']):
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

        if earlyStopper.early_stop(avg_train_loss):
            break

        end = time.time()
        # !!! \n
        print(f"Epoch {epoch+1}/{config['epochs']}. Time elapsed: {(end - start):.2f}s")
        print(f"    Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}")
        if not config['do_validation']:
            history['loss'].append(avg_train_loss)
            history['iou'].append(avg_train_iou)

        if epoch == 0 or (epoch+1) % config['save_checkpoint_in_between_n_epochs'] == 0:
            utils.MakeDirectory(config['checkpoint_path'])
            utils.MakeDirectory(config['history_path'])
            torch.save(model.state_dict(), os.path.join(config['checkpoint_path'], f"{config['checkpoint_name']}{epoch+1}.pth"))

            # Evaluation
            if config['do_validation']:
                precisions, recalls, ious, dices, avg_val_loss, avg_val_iou = evaluate_model(model, val_dataloader, loss_fn, config['device'], config['classes_n'])
                print(f"    Validation Loss: {avg_val_loss:.4f}, Validation IoU: {avg_val_iou:.4f}")
                print(f"    Metrics by class:")
                print(f"        IoU:        {[round(x, 2) for x in ious[1:]]}")
                print(f"        Dice (F1):  {[round(x, 2) for x in dices[1:]]}")
                print(f"        Precision:  {[round(x, 2) for x in precisions[1:]]}")
                print(f"        Recall:     {[round(x, 2) for x in recalls[1:]]}")
                history['loss'].append(avg_train_loss)
                history['iou'].append(avg_train_iou)
                history['val_loss'].append(avg_val_loss)
                history['val_iou'].append(avg_val_iou)
                pd.DataFrame(history).to_csv(os.path.join(config['history_path'], f"{config['checkpoint_name']}_{epoch+1}.csv"))
            else:
                pd.DataFrame(columns=[history["iou"], history["loss"]]).to_csv(os.path.join(config['history_path'], f"{config['checkpoint_name']}_{epoch+1}.csv"))

            

if __name__ == "__main__":
    config = {}
    with open(sys.argv[1]) as config_file:
        config = json.load(config_file)
    main(config)


