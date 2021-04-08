import sys
import os
import numpy as np
import pandas as pd

# check the path!
LABELS_CSV = open('./dataset/stage_2_train_labels.csv')
CLASS_INFO_CSV = open('./dataset/stage_2_detailed_class_info.csv')
sample = False
n_samples = 100


if len(sys.argv[1:]) > 0:
    if (sys.argv[1] == "--sample"):
        sample = True
        if  len(sys.argv[1:]) > 1:
            n_samples = int(sys.argv[2])

    if (sys.argv[1] == "-h"):
        print(' Normal usage: python3 merge_dataset_script.py \n For sample: python3 merge_dataset_script.py -sample [#samples] \n (#samples default value = 100)')
        sys.exit(1)


# MERGE phase
train_labels_df = pd.read_csv(LABELS_CSV)
class_info_df = pd.read_csv(CLASS_INFO_CSV)

# fill empty labels (class 0) with all 0
train_labels_df.x.fillna(0, inplace=True)
train_labels_df.y.fillna(0, inplace=True)
train_labels_df.width.fillna(0, inplace=True)
train_labels_df.height.fillna(0, inplace=True)

# add Pascal bounding boxes encoding
train_labels_df['x_max'] = train_labels_df['x']+train_labels_df['width']
train_labels_df['y_max'] = train_labels_df['y']+train_labels_df['height']

train_labels_df = train_labels_df.merge(class_info_df, left_on='patientId', right_on='patientId', how='inner')
train_labels_df = train_labels_df.drop_duplicates()


if sample:
    train_labels_df = train_labels_df.sample(n_samples)


# SPLIT phase
np.random.seed(13)
msk = np.random.rand(len(train_labels_df)) < 0.8

# split train and val/test + add indexes from 0 as required by class definition
train_df = train_labels_df[msk].reset_index()  
val_train_df = train_labels_df[~msk]

# split val/test
split_val = int(len(val_train_df)/2)
val_df = val_train_df.iloc[:split_val,:].reset_index()
test_df = val_train_df.iloc[split_val:,:].reset_index()

os.makedirs('dataset/tmp', exist_ok=True)
train_df.to_csv(path_or_buf='./dataset/tmp/train_labels.csv', index=False)
val_df.to_csv(path_or_buf='./dataset/tmp/valid_labels.csv', index=False)
val_df.to_csv(path_or_buf='./dataset/tmp/test_labels.csv', index=False)
print(f' Samples in train set: {len(train_df)} \n Samples in validation set: {len(val_df)} \n Samples in test set: {len(test_df)}')
