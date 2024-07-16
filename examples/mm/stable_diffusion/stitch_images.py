# script to stitch images into one
from glob import glob
import matplotlib.pyplot as plt
import argparse
import os.path as osp
from PIL import Image

def main(args):
    images = sorted(glob(osp.join(args.path, "img*.png")) + glob(osp.join(args.path, "img*.jpg")))
    print(f"found {len(images)} images.")
    fig, axs = plt.subplots(2, 4, figsize=(17, 8))
    idx = 0
    for i in range(2):
        for j in range(4):
            axs[i][j].imshow(Image.open(images[idx]))
            axs[i][j].axis('off')
            idx += 1
    plt.savefig(osp.join(osp.dirname(images[0]), "stitched_images.png"), bbox_inches='tight')
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--path", required=True, type=str)
    args = args.parse_args()
    main(args)