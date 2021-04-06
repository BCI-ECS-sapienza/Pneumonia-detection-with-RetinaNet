import sys
import os
import numpy as np
import pandas as pd
import collections
import pickle
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from dataloader import CXRimages, collater2d
from RetinaNet.retinanet import RetinaNet
from RetinaNet.encoder_resnet import resnet50 
# IMPORT ALL ENCODERS

import matplotlib.pyplot as plt
import torch.optim as lr_scheduler
from torch import nn, optim
from tqdm import tqdm

# PARAMS and CONFIGS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8

TEST_CSV = ''
IMAGES_DIR = ''
PICS_DIR = './pics'
CHECKPOINTS = './checkpoints'
EPOCHS = 5

AUGMENTATION = "resize_only"


def test(
    test_dataloader: nn.Module,
    checkpoint: str, 
    pics_dir: str,
    test_dataset
    ):

    # Load model
    model = torch.load('./checkpoints/resize_only_resnet50/resnet50_final.pt') #TODO: LOAD MODEL
    model = model.to(device)
    model.eval()

    # Show a smart progress bar
    data_iter = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    for iter_num, data in data_iter:

    # Run model and save the return values of the model (see Model section for further info)
    (
        classification_loss,
        regression_loss,
        global_classification_loss,
        nms_scores,
        nms_class,
        transformed_anchors,
    ) = model(
        [
            data["img"].to(device).float(),
            data["annot"].to(device).float(),
            data["category"].to(device),
        ],
        return_loss=True,
        return_boxes=True,
    )

    nms_scores = nms_scores.cpu().detach().numpy()
    nms_class = nms_class.cpu().detach().numpy()
    transformed_anchors = transformed_anchors.cpu().detach().numpy()

    # Print results
    print(f"nms_scores {nms_scores}, transformed_anchors.shape {transformed_anchors.shape}")
    print(f"cls loss: {float(classification_loss)}, global cls loss: {global_classification_loss}, reg loss: {float(regression_loss)}")
    print(
        "category:",
        data["category"].numpy()[0],
        np.exp(nms_class[0]),
        test_dataset.categories[data["category"][0]],
    )
    # PLOT RESULTS

    # plot data and ground truth
    plt.figure(iter_num, figsize=(6, 6))
    plt.cla()
    plt.imshow(data["img"][0, 0].cpu().detach().numpy(), cmap=plt.cm.gist_gray)
    plt.axis("off")
    gt = data["annot"].cpu().detach().numpy()[0]
    for i in range(gt.shape[0]):
        if np.all(np.isfinite(gt[i])):
            p0 = gt[i, 0:2]
            p1 = gt[i, 2:4]
            plt.gca().add_patch(
                plt.Rectangle(
                    p0,
                    width=(p1 - p0)[0],
                    height=(p1 - p0)[1],
                    fill=False,
                    edgecolor="b",
                    linewidth=2,
                )
            )

    # add predicted boxes to the plot
    for i in range(len(nms_scores)):
        nms_score = nms_scores[i]
        if nms_score < 0.1:
            break
        p0 = transformed_anchors[i, 0:2]
        p1 = transformed_anchors[i, 2:4]
        color = "r"
        if nms_score < 0.3:
            color = "y"
        if nms_score < 0.25:
            color = "g"
        plt.gca().add_patch(
            plt.Rectangle(
                p0,
                width=(p1 - p0)[0],
                height=(p1 - p0)[1],
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
        )
        plt.gca().text(p0[0], p0[1], f"{nms_score:.3f}", color=color)
    plt.show()

    os.makedirs(pics_dir, exist_ok=True)
    plt.savefig(
        f"{pics_dir}/predict_{iter_num}.eps", dpi=300, bbox_inches="tight", pad_inches=0,
    )
    plt.savefig(
        f"{pics_dir}/predict_{iter_num}.png", dpi=300, bbox_inches="tight", pad_inches=0,
    )
    plt.close()

    print(nms_scores)

def main():
    # load custom dataset
    test_dataset = CXRimages(csv_file = TEST_CSV , images_dir = IMAGES_DIR, augmentations=AUGMENTATION, transform = None)
    print(f'Samples in test set: {len(test_dataset)}')

    # set batch size
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collater2d) 

    
    ### RUNNER ###
    print("TESTS STARTED")
    test(test_dataloader, CHECKPOINTS, PICS_DIR, test_dataset)
    print("TESTS FINISHED\n")


if __name__ == "__main__":

    if len(sys.argv[1:]) < 3:
        print('USAGE: python3 test.py [test_csv_path] [images_dir_path] [checkpoints_path] {[augmentation_level]} \
            \n Augmentation levels: resize_only (default), light, heavy, heavy_with_rotations')
        sys.exit(1)

    TEST_CSV = sys.argv[1]
    IMAGES_DIR = sys.argv[2]
    CHECKPOINTS = sys.argv[3]
    elif len(sys.argv[1:]) == 4:
        AUGMENTATION = sys.argv[4]
    
    print(f' labels_csv_path: {TEST_CSV}\n images_dir_path: {IMAGES_DIR}\n checkpoints_path: {CHECKPOINTS}\n augmentation_level: {AUGMENTATION}\n')
    
    main()
