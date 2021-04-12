import sys
import os
import numpy as np
import pandas as pd
import argparse
import collections
import pickle
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from dataloader import CXRimages, collater2d
from RetinaNet.retinanet import RetinaNet
from RetinaNet.encoder_resnet import resnet50 
from RetinaNet.encoder_se_resnext50 import se_resnext50
from RetinaNet.encoder_xception import xception

import matplotlib.pyplot as plt
import torch.optim as lr_scheduler
from torch import nn, optim
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PICS_DIR = './pics'

def test(
    test_dataloader: nn.Module,
    checkpoint: str, 
    pics_dir: str,
    test_dataset
    ):

    # Load model
    path = f"./models/{MODEL}.pt"
    model = torch.load(path) #TODO: LOAD MODEL
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
            return_boxes=True
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


def main(LABELS_DIR, IMAGES_DIR, MODEL, BATCH_SIZE):
    # load custom dataset
    test_df = pd.read_csv(LABELS_DIR+'test_labels.csv')
    test_dataset = CXRimages(csv_file = test_df , images_dir = IMAGES_DIR, transform = None)
    print(f'Samples in test set: {len(test_dataset)}')

    # set batch size
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collater2d) 

    
    ### RUNNER ###
    print("TESTS STARTED")
    test(test_dataloader, MODEL, PICS_DIR, test_dataset)
    print("TESTS FINISHED\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--labels_folder", type=str, default="dataset/tmp/", help="CSVs folder path")
    arg("--images_folder", type=str, default="dataset/stage_2_train_images/", help="images folder path")
    arg("--model", type=str, default='resnet50_resize_only', help="encoder")
    arg("--batch_size", type=int, default=8, help="batch size")
    args = parser.parse_args()

    LABELS_DIR = args.labels_folder
    IMAGES_DIR = args.images_folder
    MODEL = args.model
    BATCH_SIZE = args.batch_size
  
    print(f' labels_folder_path: {LABELS_DIR}\n images_dir_path: {IMAGES_DIR}\n model: {MODEL}\n batch_size: {BATCH_SIZE}\n')
    main(LABELS_DIR, IMAGES_DIR, MODEL, BATCH_SIZE)