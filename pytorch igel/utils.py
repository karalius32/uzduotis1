import torch
import os
import numpy as np
import shutil


def evaluate_model(model, dataloader, loss_fn, device, class_n, batch_size):
    model.eval()
    running_loss = 0
    total_iou = 0
    class_n = class_n + 1 # if use background

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


def crop(img, x, y, w, h):
    height, width = img.shape
    x, y, w, h = int(x), int(y), int(w), int(h)
    img_crop = np.zeros((h, w), np.uint8)
    if x + w < 0 or y + h < 0 or x > width or y > height:
        print('crop out of bounds: %i %i %i %i to %i %i' % (x, y, w, h, width, height))
    elif x < 0 or y < 0 or x + w > width or y + h > height:
        x1 = 0
        y1 = 0
        w1 = w
        h1 = h
        x2 = x
        y2 = y
        w2 = x + w
        h2 = y + h
        if x < 0:
            x1 = -x
            x2 = 0
        if x + w > width - 1:
            w1 = width - 1 - x
            w2 = width - 1
        if y < 0:
            y1 = -y
            y2 = 0
        if y + h > height - 1:
            h1 = height - 1 - y
            h2 = height - 1
        img_crop[y1:h1, x1:w1] = img[y2:h2, x2:w2]
    else:
        img_crop[0:h, 0:w] = img[y:y+h, x:x+w]
    return img_crop


def MakeDirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_file_name(path):
    if not os.path.exists(path):
        print('file %s does not exist' % path)
        return ''
    name = os.path.basename(path)
    return os.path.splitext(name)[0]


def get_files(dir):
    files = []
    for file_name in os.listdir(dir):
        files.append(os.path.join(dir, file_name))
    return files


def get_file_pairs(*args):
    """
    returns all file pairs that match by name in given directories
    e.g. get_file_pairs(directory1, directory2, directory3)
    """
    pairs = []
    all_files = [get_files(directory) for directory in args]
    for root_file in all_files[0]:
        root_file_name = get_file_name(root_file)
        matches = []
        for i in range(1, len(all_files)):
            for file in all_files[i]:
                file_name = get_file_name(file)
                if file_name == root_file_name:
                    matches.append(file)
                    all_files[i].remove(file)
                    break
        if len(matches) == len(all_files) - 1:
            pairs.append(np.append(root_file, matches))
    return pairs


def generate_validation_set(dir, size):
    """
    Generates validation set over images (jpg) and masks (png).
    assuming dir looks like this:
    -dir:
    ----images
    ----labels
    the result will be:
    -dir:
    ----images
    ----masks
    ----val_images
    ----val_masks
    
    Args:
    size: percent of all data to put in validation set (from 0 to 1)
    """
    if os.path.exists(os.path.join(dir, "val_images")):
        return
    images = os.listdir(os.path.join(dir, "images"))
    length = len(images)
    n = int(length * size)
    indices = np.random.choice(length, n, replace=False)

    val_img_dir = os.path.join(dir, "val_images")
    val_masks_dir = os.path.join(dir, "val_masks")
    MakeDirectory(val_img_dir)
    MakeDirectory(val_masks_dir)
    for i in indices:
        image_path = os.path.join(dir, "images", images[i])
        mask_path = os.path.join(dir, "labels", images[i].replace("jpg", "png"))
        shutil.move(image_path, val_img_dir)
        shutil.move(mask_path, val_masks_dir)

