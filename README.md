# Pneumonia Detection with RetinaNet

Identifying cases of Pneumonia is tedious and often leads to a disagreement between radiologists. However, computer-aided diagnosis systems showed the potential for improving diagnostic accuracy. In this work, taking inspiration from the reference paper [1], we replicate and build some computational approaches for pneumonia regions detection.

For the experiment documents:
* ##### [Full report](./docs/report.pdf)
* ##### [Presentation slides](./docs/slides.pdf)


## Data & methodology

The dataset was publicly provided by the *US National Institutes of Health Clinical Center*. 
It comprises frontal-view X-ray images from 26684 unique patients. Each image was labelled with one of 
three different classes from the associated radiological reports:
- The **"Normal"** class contained data of healthy patients without any pathologies found (including, but not limited to pneumonia, pneumothorax, atelectasis, etc.).
- The **"Lung Opacity"** class had images with the presence of fuzzy clouds of white in the lungs, associated with pneumonia. The regions of lung opacities were labelled with bounding boxes. Any given patient could have multiple boxes if more than one area with pneumonia was detected. There are different kinds of lung opacities, some are related to pneumonia and some are not.
-  The class **"No Lung Opacity / Not Normal"** illustrated data for patients with visible on CXR lung opacity regions, but without diagnosed pneumonia.


Once preprocessed the dataset, we built a Pytorch *RetinaNet*-based model [2], with the following encoders:
- **ResNet50:** short for Residual Networks, it is a classic neural network used as a backbone for many computer vision tasks. It is based on the idea of skip connections, or shortcuts to jump over some layers. In this case, we are using the 50 layers version.
- **Pnasnet5:** short for Progressive Neural Architecture Search, that is based on the idea of Neural architecture search, which is a technique for automating the design of artificial neural networks. It uses a sequential model-based optimization (SMBO) strategy, that searches for structures in order of increasing complexity, while simultaneously learning a surrogate model to guide the search through structure space.
- **Se_resnext50:** inherited from ResNet, VGG, and Inception, the basic ResNeXt includes shortcuts from the previous block to next block, stacking layers and adapting split- transform-merge strategy. Moreover, in this version we have the Squeeze-and-Excitation blocks, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels. 

Moreover, for each encoder we tried four different augmentations:
- **resize_only:** resize only images, no real augmentation.
- **light**: affine and perspective changes (scale=0.1, shear=2.5), and rotations (angle=5.0).
- **heavy:** random horizontal flips, affine and perspective changes (scale=0.15, shear=4.0), occasional Gaussian noise, Gaussian blur, and additive noise.
- **heavy_with_rotations:** random horizontal flips, affine and perspective changes (scale=0.15, shear=4.0), rotations (angle=6.0), occasional Gaussian noise, Gaussian blur, and additive noise.


## Repository files
- network => the retinanet-based structure, the encoders, the dataloader, and so on...
- tensorboard => the training logs 
- Augmentation.ipynb => shows some example augmentations we are using for the model training
- Dataset_overview.ipynb => dataset overview and statistics
- merge_and_split_dataset.py => script to merge the input CSV files and then make train/valid/test CSVs

## How to run
1. Download the dataset folder from [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) and rename it ```dataset```
2. run merge_and_split_dataset.py, so that to make the ```dataset/tmp/``` folder containg the train/valid/test_labels.csv files (add ```--sample``` for a sample subset)
3. run network/train.py to train the model with choosen encoder and augmentation. Example parameters:
    ``` python3 network/train.py --labels_folder=dataset/tmp/ --images_folder=dataset/stage_2_train_images/ --epochs=8 --batch_size=8 --encoder=resnet50 --augmentation=resize_only --resume_epoch=0```
4. run network/test.py  to test the choosen model. Example parameters:
    ``` python3 network/test.py --labels_folder=dataset/tmp/ --images_folder=dataset/stage_2_train_images/ --model=resnet50_resize_only --batch_size=8```

In case you want to visualize our results on tensorboard: 
```python3 -m tensorboard.main --logdir=tensorboard```

---
## Authors
* ##### [Manuel Ivagnes](https://www.linkedin.com/in/manuel-ivagnes-4a5ba018b)
* ##### [Riccardo Bianchini](http://linkedin.com/in/riccardo-bianchini-7a391219b)
* ##### [Valerio Coretti](https://www.linkedin.com/in/valerio-coretti-2913721a3)


---
## Reference papers
[1] Gabruseva, Tatiana and Poplavskiy, Dmytro and Kalinin, Alexandr A.. Deep Learning for Automatic Pneumonia Detection, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. June, 2020

[2] Lin T.Y., Goyal P., Girshick R., He K., and Dollr P. Focal loss for dense object detection. IEEE International Conference on Computer Vision, page 29993007, 2017.