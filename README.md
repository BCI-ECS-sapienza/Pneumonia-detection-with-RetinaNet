# Neural Network for Automatic Pneumonia Detection

Identifying cases of Pneumonia is tedious and often leads to a disagreement between radiologists. However, computer-aided diagnosis systems showed the potential for improving diagnostic accuracy. In this work, taking inspiration from the assigned paper, we replicate and build some computational approaches for pneumonia regions detection.

For the experiment documents:
* ##### [Full report](./report.pdf)
* ##### [Presentation slides](/slides.pdf)


## Data & methodology

The labelled dataset of the chest X-ray images and patients metadata was publicly provided for the challenge by the *US National Institutes of Health Clinical Center*. This database comprises frontal-view X-ray images from 26684 unique patients. Each image was labelled with one of three different classes from the associated radiological reports:
- The ”Normal” class contained data of healthy patients without any pathologies found (including, but not limited to pneumonia, pneumothorax, atelectasis, etc.).
- The ”Lung Opacity” class had images with the presence of fuzzy clouds of white in the lungs, associated with pneumonia. The regions of lung opacities were labelled with bounding boxes. Any given patient could have multiple boxes if more than one area with pneumonia was detected. There are different kinds of lung opacities, some are related to pneumonia and some are not.
-  The class ”No Lung Opacity / Not Normal” illustrated data for patients with visible on CXR lung opacity regions, but without diagnosed pneumonia.


Once preprocessed the dataset, we built a *RetinaNet* based model, with the following characteristics:
- a



---
## Authors
* ##### [Manuel Ivagnes](https://www.linkedin.com/in/manuel-ivagnes-4a5ba018b)
* ##### [Riccardo Bianchini](http://linkedin.com/in/riccardo-bianchini-7a391219b)
* ##### [Valerio Coretti](https://www.linkedin.com/in/valerio-coretti-2913721a3)


---
## Reference paper
Gabruseva, Tatiana and Poplavskiy, Dmytro and Kalinin, Alexandr A.. Deep Learning for Automatic Pneumonia Detection, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. June, 2020