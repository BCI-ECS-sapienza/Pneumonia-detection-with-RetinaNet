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

import torch.optim as lr_scheduler
from torch import nn, optim
from tqdm import tqdm


# PARAMS and CONFIGS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8

LABELS_CSV = ''
IMAGES_DIR = ''
EPOCHS = 0

AUGMENTATION = "resize_only"
ENCODER = "resnet50"


def validation(
    retinanet: nn.Module, 
    dataloader_valid: nn.Module, 
    epoch_num: int, 
    predictions_dir: str, 
    save_oof=True,
) -> tuple:

  print(f'START VALIDATION EPOCH {epoch_num}')
  with torch.no_grad():
      retinanet.eval()
      loss_hist_valid, loss_cls_hist_valid, loss_cls_global_hist_valid, loss_reg_hist_valid = [],[],[],[]
      data_iter = tqdm(enumerate(dataloader_valid), total=len(dataloader_valid))

      if save_oof:
          oof = collections.defaultdict(list)
          
      for iter_num, data in data_iter:
          (
            classification_loss,
            regression_loss,
            global_classification_loss,
            nms_scores,
            global_class,
            transformed_anchors,
          ) = retinanet(
              [
                  data["img"].to(device).float(),
                  data["annot"].to(device).float(),
                  data["category"].to(device),
              ],
              return_loss=True,
              return_boxes=True,
          )

          if save_oof:
              # predictions
              oof["gt_boxes"].append(data["annot"].cpu().numpy().copy())
              oof["gt_category"].append(data["category"].cpu().numpy().copy())
              oof["boxes"].append(transformed_anchors.cpu().numpy().copy())
              oof["scores"].append(nms_scores.cpu().numpy().copy())
              oof["category"].append(global_class.cpu().numpy().copy())

          # get losses
          classification_loss = classification_loss.mean()
          regression_loss = regression_loss.mean()
          global_classification_loss = global_classification_loss.mean()
          loss = classification_loss + regression_loss + global_classification_loss * 0.1

          # loss history
          loss_hist_valid.append(float(loss))
          loss_cls_hist_valid.append(float(classification_loss))
          loss_cls_global_hist_valid.append(float(global_classification_loss))
          loss_reg_hist_valid.append(float(regression_loss))
          #data_iter.set_description(
          #    f"{epoch_num} cls: {np.mean(loss_cls_hist_valid):1.4f} cls g: {np.mean(loss_cls_global_hist_valid):1.4f} Reg: {np.mean(loss_reg_hist_valid):1.4f} Loss {np.mean(loss_hist_valid):1.4f}"
          #)
          print(
            '\n VALIDATION Epoch: {} | Iteration: {} | Classification loss: {:1.5f} |\n Regression loss: {:1.5f} | Global loss: {:1.5f} | Running loss: {:1.5f}'.format(
              epoch_num, iter_num, float(np.mean(loss_cls_hist_valid)), float(np.mean(loss_reg_hist_valid)), float(np.mean(loss_cls_global_hist_valid)), np.mean(loss_hist_valid))
          )
          del classification_loss
          del regression_loss

      if save_oof:  # save predictions
          pickle.dump(oof, open(f"{predictions_dir}/{epoch_num:03}.pkl", "wb"))

  print(f'END VALIDATION EPOCH {epoch_num}')
  return loss_hist_valid, loss_cls_hist_valid, loss_cls_global_hist_valid, loss_reg_hist_valid


def train(
    model_name: str,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    epochs: int,
    resume_weights: str="",
    resume_epoch: int=0):

  pretrained = True

  if model_name == 'resnet50':
    retinanet = resnet50(1, pretrained)
  elif model_name == 'se_resnext50':
    retinanet = se_resnext50(1, pretrained)
  elif model_name == 'pnasnet5':
    retinanet = PNasnet5(1, pretrained)
  elif model_name == 'xception':
    retinanet = xception(1, pretrained)

  checkpoints_dir = f'./checkpoints/{AUGMENTATION}_{ENCODER}'
  predictions_dir = f'./predictions/{AUGMENTATION}_{ENCODER}'
  tensorboard_dir = f'./tensorboard/{AUGMENTATION}_{ENCODER}'
  os.makedirs(checkpoints_dir, exist_ok=True)
  os.makedirs(predictions_dir, exist_ok=True)
  os.makedirs(tensorboard_dir, exist_ok=True)
  logger = SummaryWriter(tensorboard_dir)

  # load weights to continue training
  if resume_weights != "":
    print("load model from: ", resume_weights)
    retinanet = torch.load(resume_weights).to(device)
  else:
    retinanet = retinanet.to(device)
  
  retinanet = torch.nn.DataParallel(retinanet).to(device)

  retinanet.training = True
  optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
  
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=4, verbose=True, factor=0.2
  )
  scheduler_by_epoch = False

  #for epoch_num in range(resume_epoch+1, epochs):
  for epoch_num in range(epochs):  
    retinanet.train() 
    
    if epoch_num < 1:
      # train FC layers with freezed encoder for the first epoch
      retinanet.module.freeze_encoder()  
    else:
      retinanet.module.unfreeze_encoder()
    
    retinanet.module.freeze_bn()

    # losses
    epoch_loss, loss_cls_hist, loss_cls_global_hist, loss_reg_hist = [], [], [], []

    with torch.set_grad_enabled(True):
      data_iter = tqdm(enumerate(train_dataloader), total = len(train_dataloader))

      for iter_num, data in data_iter:
        optimizer.zero_grad()

        inputs = [
                  data['img'].to(device).float(),      
                  data['annot'].to(device).float(),    
                  data['category'].to(device)
        ]

        (classification_loss, regression_loss, global_classification_loss,) = retinanet(
            inputs, return_loss=True, return_boxes=False
            )

        classification_loss = classification_loss.mean() 
        regression_loss = regression_loss.mean()
        global_classification_loss = global_classification_loss.mean()
        loss = classification_loss + regression_loss + global_classification_loss*0.1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.05)
        optimizer.step()
        
        # loss history
        loss_cls_hist.append(float(classification_loss))
        loss_cls_global_hist.append(float(global_classification_loss))
        loss_reg_hist.append(float(regression_loss))
        epoch_loss.append(float(loss))
        
        print(
          '\n Epoch: {} | Iteration: {} | Classification loss: {:1.5f} |\n Regression loss: {:1.5f} | Global loss: {:1.5f} | Running loss: {:1.5f}'.format(
            epoch_num, iter_num, float(classification_loss), float(regression_loss), float(global_classification_loss), np.mean(epoch_loss))
        )

        del classification_loss
        del regression_loss

    torch.save(retinanet.module, f"{checkpoints_dir}/{model_name}_{epoch_num:03}.pt")
    logger.add_scalar('Loss_train', np.mean(epoch_loss), epoch_num)
    logger.add_scalar("loss_train_classification", np.mean(loss_cls_hist), epoch_num)
    logger.add_scalar("loss_train_global_classification", np.mean(loss_cls_global_hist), epoch_num)
    logger.add_scalar("loss_train_regression", np.mean(loss_reg_hist), epoch_num)

    # validation
    (
      loss_hist_valid,
      loss_cls_hist_valid,
      loss_cls_global_hist_valid,
      loss_reg_hist_valid,
    ) = validation(retinanet,
        validation_dataloader,
        epoch_num,
        predictions_dir,
        save_oof=True,
    )
  
    # log validation loss history
    logger.add_scalar("loss_valid", np.mean(loss_hist_valid), epoch_num)
    logger.add_scalar("loss_valid_classification", np.mean(loss_cls_hist_valid), epoch_num)
    logger.add_scalar("loss_valid_global_classification", np.mean(loss_cls_global_hist_valid), epoch_num,)
    logger.add_scalar("loss_valid_regression", np.mean(loss_reg_hist_valid), epoch_num)

    if scheduler_by_epoch:
        scheduler.step(epoch=epoch_num)
    else:
        scheduler.step(np.mean(loss_reg_hist_valid))
  
  retinanet.eval()
  torch.save(retinanet, f"{checkpoints_dir}/{model_name}_final.pt")
  logger.close()



def main():
  np.random.seed(13)

  train_class_df = pd.read_csv(LABELS_CSV)
  msk = np.random.rand(len(train_class_df)) < 0.8

  # split train and val/test + add indexes from 0 as required by class definition
  train_df = train_class_df[msk].reset_index()  
  val_train_df = train_class_df[~msk]

  # split val/test
  split_val = int(len(val_train_df)/2)
  val_df = val_train_df.iloc[:split_val,:].reset_index()
  test_df = val_train_df.iloc[split_val:,:].reset_index()

  # load custom dataset
  train_dataset = CXRimages(csv_file = train_df , images_dir = IMAGES_DIR, augmentations=AUGMENTATION, transform = None)
  val_dataset = CXRimages(csv_file = val_df , images_dir = IMAGES_DIR, augmentations=AUGMENTATION, transform = None)
  test_dataset = CXRimages(csv_file = test_df , images_dir = IMAGES_DIR, augmentations=AUGMENTATION, transform = None)
  print(f' Samples in train set: {len(train_dataset)} \n Samples in validation set: {len(val_dataset)} \n Samples in test set: {len(test_dataset)}')

  # set batch size
  train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collater2d) 
  val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collater2d)
  test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collater2d) 

  ### RUNNER ###
  print("\nTRAINING STARTED")
  train(ENCODER, train_dataloader, val_dataloader, EPOCHS, '', 0)
  print("TRAINING FINISHED\n")


  ### TEST ###
  print("TESTS STARTED")
  # test => test_dataloader
  print("TESTS FINISHED\n")


if __name__ == "__main__":

  if len(sys.argv[1:]) <3:
    print('USAGE: python3 train.py [labels_csv_path] [images_dir_path] [epochs] {[augmentation_level]} {[encoder]} \
          \n Augmentation levels: resize_only (default), light, heavy, heavy_with_rotations \
          \n Encoders: resnet_50 (default), se_resnext50, pnasnet5, xception')
    sys.exit(1)

  LABELS_CSV = sys.argv[1]
  IMAGES_DIR = sys.argv[2]
  EPOCHS = int(sys.argv[3])
  if len(sys.argv[1:]) == 4:
    AUGMENTATION = sys.argv[4]
  elif len(sys.argv[1:]) == 5:
    AUGMENTATION = sys.argv[4]
    ENCODER = sys.argv[5]
  
  print(f' labels_csv_path: {LABELS_CSV}\n images_dir_path: {IMAGES_DIR}\n images_dir_path: {EPOCHS}\n augmentation_level: {AUGMENTATION}\n encoder: {ENCODER}\n')

  
  main()