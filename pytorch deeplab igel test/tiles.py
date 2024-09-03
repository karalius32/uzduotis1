import random
import os
import utils
import math
import cv2
import numpy as np
import multiprocessing
import shutil
import misc


def get_tiles(image, label, tile_size, zero_sampling):
    """
    Gets a list of tiles, where each tile represents a region where at least one label pixel exists.
    Zero sampling can be used to sample tiles with zero label pixels, value range is [0, 1].
    Region structure is [x, y, width, height] where each value is normalized to [0, 1].
    """
    min_x_overlap = 0.25
    max_x_overlap = 0.75
    min_y_overlap = 0.25
    max_y_overlap = 0.75
    image_height, image_width = image.shape
    tile_width = float(tile_size[1])
    tile_height = float(tile_size[0])
    all_tiles = []
    out_tiles = []
    y = 0
    while y < image_height:
        x = 0
        while x < image_width:
            all_tiles.append(misc.Rect(x, y, tile_width, tile_height))
            x += (1. - random.uniform(min_x_overlap, max_x_overlap)) * tile_width
        y += (1. - random.uniform(min_y_overlap, max_y_overlap)) * tile_height
    i = 0
    last = 0
    for tile in all_tiles:
        rect = [float(tile.x) / image_width, float(tile.y) / image_height, float(tile.width) / image_width, float(tile.height) / image_height]
        crop_label = utils.crop(label, tile.x, tile.y, tile.width, tile.height)
        curr = math.floor(i * zero_sampling)
        if np.count_nonzero(crop_label) > 0 or curr != last:
            out_tiles.append(rect)
        i += 1
        last = curr
    return out_tiles


def generate_tiles(image_file, label_file, out_dir, tile_size, zero_sampling):
    utils.MakeDirectory(os.path.join(out_dir, 'images'))
    utils.MakeDirectory(os.path.join(out_dir, 'labels'))
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    lbl = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
    tiles = get_tiles(img, lbl, tile_size, zero_sampling)
    h, w = img.shape
    for rect in tiles:
        rect[0] = int(rect[0] * w)
        rect[1] = int(rect[1] * h)
        rect[2] = int(rect[2] * w)
        rect[3] = int(rect[3] * h)
        img_tile = utils.crop(img, rect[0], rect[1], rect[2], rect[3])
        lbl_tile = utils.crop(lbl, rect[0], rect[1], rect[2], rect[3])
        file_name = utils.get_file_name(image_file) + ('_%i_%i_%i_%i' % (rect[0], rect[1], rect[2], rect[3]))
        cv2.imwrite(os.path.join(out_dir, 'images', file_name + '.jpg'), img_tile)
        cv2.imwrite(os.path.join(out_dir, 'labels', file_name + '.png'), lbl_tile)


def generate_tile_cache(image_dir, label_dir, output_dir, **kwargs):
    files = utils.get_file_pairs(image_dir, label_dir)
    if len(files) == 0:
        return
    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, '.files')):
        files_match = True
        all_lines = open(os.path.join(output_dir, '.files'), 'r').read().splitlines()
        names_list = [name for name in all_lines if name]
        for pair in files:
            file_name = utils.get_file_name(pair[0])
            file_exists = False
            for name in names_list:
                if file_name == name:
                    file_exists = True
                    break
            if not file_exists:
                files_match = False
                break
        if files_match:
            return
    print('--- generating cache ---')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    utils.MakeDirectory(os.path.join(output_dir, 'images'))
    utils.MakeDirectory(os.path.join(output_dir, 'labels'))
    with open(os.path.join(output_dir, '.files'), 'x') as file:
        file.writelines([(utils.get_file_name(x) + '\n') for x, y in files])
    print('workers in use: %i' % multiprocessing.cpu_count())
    with multiprocessing.Pool() as pool:
        process_batch = 1000
        steps = len(files) // process_batch + 1
        for i in range(steps):
            print('%i/%i' % (i * process_batch, len(files)))
            low = i * process_batch
            high = min(low + process_batch, len(files))
            items = [(img_file, lbl_file, output_dir, (kwargs['size'], kwargs['size']), kwargs['zero_sampling']) for img_file, lbl_file in files[low:high]]
            pool.starmap(generate_tiles, items)