import sys
import numpy as np
import pandas as pd

# check the path!
LABELS_CSV = open('./dataset/stage_2_train_labels.csv')
CLASS_INFO_CSV = open('./dataset/stage_2_detailed_class_info.csv')
sample = False
n_samples = 300


if len(sys.argv[1:]) > 0:
    if (sys.argv[1] == "-sample"):
        sample = True
        if  len(sys.argv[1:]) > 1:
            n_samples = int(sys.argv[2])

    if (sys.argv[1] == "-h"):
        print(' Normal usage: python merge_dataset_script.py \n For sample: python merge_dataset_script.py -sample [#samples] \n (#samples default value = 100)')
        sys.exit(1)


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
    sample_df = train_labels_df.sample(n_samples)
    sample_df.to_csv(path_or_buf='sampled_labels.csv', index=False)
    print(sample_df.count())
    print(sample_df[:20])
else:
    train_labels_df.to_csv(path_or_buf='merged_labels.csv', index=False)
    print(train_labels_df.count())
    print(train_labels_df[:20])