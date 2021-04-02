import os
import pandas as pd
import numpy as np
import pydicom

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def augmentation_pipeline(level):
  if level == 'resize_only':
    list_augmentations = [
      iaa.Resize(512)            
    ]

  elif level == 'light':
    list_augmentations = [
      iaa.Resize(512),
      iaa.Affine(
        scale=1.1, 
        shear=(2.5,2.5), 
        rotate=(-5, 5), 
      ),    
    ]
    
  elif level == 'heavy': #no rotation included
    list_augmentations = [
      iaa.Resize(512),
      iaa.Affine(
        scale=1.15, 
        shear=(4.0,4.0),
      ),   
      iaa.Fliplr(0.2), # horizontally flip 20% of the images
      iaa.Sometimes(0.1, iaa.CoarseSaltAndPepper(p=(0.01, 0.01), size_percent=(0.1, 0.2))),
      iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 2.0))),
      iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255))),            
           
    ]

  elif level == 'heavy_with_rotations':
    list_augmentations = [
      iaa.Resize(512),
      iaa.Affine(
        scale=1.15, 
        shear=(4.0,4.0),
        rotate=(-6, 6), 
      ),   
      iaa.Fliplr(0.2), # horizontally flip 20% of the images
      iaa.Sometimes(0.1, iaa.CoarseSaltAndPepper(p=(0.01, 0.01), size_percent=(0.1, 0.2))),
      iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 2.0))),
      iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255))),            
    ]

  return list_augmentations


def get_image_array(image_path):
  try:
    dcm_data = pydicom.read_file(image_path)
    img = dcm_data.pixel_array
    return img
  except:
      pass


def parse_one_annot(box_coord, filename):
  boxes_array = box_coord[box_coord["patientId"] == filename][["x", "y", "x_max", "y_max"]].values
  return boxes_array 


class CXRimages(Dataset):
    def __init__(self, csv_file, images_dir, augmentations='resize_only', transform=None):
      self.path = images_dir      
      self.annotations = csv_file
      self.categories = ["No Lung Opacity / Not Normal", "Normal", "Lung Opacity"]
      self.augmentations = augmentation_pipeline(augmentations)    # augmentations with imgaug
      self.transform = transform                                   # Images ToTensor and normalize
      #self.imgs = sorted(os.listdir(images_dir))


    def num_classes(self):
      return 3


    def __len__(self):
      return len(self.annotations)


    def __getitem__(self, idx):   # requires to define new indexes from 0
        patient_id = self.annotations['patientId'][idx]
        category = self.categories.index(self.annotations['class'][idx])
        target = self.annotations['Target'][idx]

        # load image
        img_path = os.path.join(self.path, patient_id +'.dcm')
        img = get_image_array(img_path)  

        # get bounding boxes from csv
        box_list = parse_one_annot(self.annotations, patient_id)
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        num_objs = len(box_list)

        # get box encoding for imaug
        list_boxes = []
        for j in range(num_objs):
          list_boxes.append(BoundingBox(x1=boxes[j][0].item(), x2=boxes[j][2].item(), y1=boxes[j][1].item(), y2=boxes[j][3].item()))
        bbs = BoundingBoxesOnImage(list_boxes, shape=img.shape)

        # augmentation
        seq_training = iaa.Sequential(self.augmentations)
        image_aug, bbs_aug = seq_training(image=img, bounding_boxes=bbs)     

        # set bounding boxes on required encoding for the model
        final_boxes = np.zeros((0, 5))

        if target == 1:
          for box in bbs_aug.bounding_boxes:
            annotation  = np.zeros((1, 5))
            annotation[0, :4] = [box.x1, box.y1, box.x2, box.y2]
            #annotation[0, 4]  = target
            annotation[0, 4]  = 0
            final_boxes       = np.append(final_boxes, annotation, axis=0)  
          
          final_boxes = np.row_stack(final_boxes)


        if self.transform is not None:
                image_aug = self.transform(image_aug.copy())  # .copy() avoid negative values in tensor

        output = {"img": image_aug, "annot": final_boxes, "scale": 1.0, 'category': category}
        return output


def To_tensor_tfms():
   transforms = []
   transforms.append(T.ToTensor())
   return T.Compose(transforms)

# padding for tensors
def collater2d(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    cats = np.array([s['category'] for s in data])

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 1)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), 0] = torch.from_numpy(img)

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = torch.from_numpy(annot)
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales, 'category': torch.from_numpy(cats)}