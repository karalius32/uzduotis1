import os
import numpy as np
import shutil


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

