import sys
import tiles
import os


def main(input_dir, output_dir, size, zero_sampling):
    tiles.generate_tile_cache(os.path.join(input_dir, "images"), os.path.join(input_dir, "masks"), output_dir, size, zero_sampling)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])