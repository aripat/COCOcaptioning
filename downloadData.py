"""
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.coco as fouc

fo.config.dataset_zoo_dir = 'data/'

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    max_samples=50,
    shuffle=False,
)"""

import json
import pandas as pd
from pycocotools.coco import COCO
import os

# initialize COCO API for instance annotations
dataDir = '.'
dataType = 'train2017'
instances_annFile = os.path.join(dataDir, 'data/cocoapi/annotations/instances_{}.json'.format(dataType))
coco = COCO(instances_annFile)

# initialize COCO API for caption annotations
captions_annFile = os.path.join(dataDir, 'data/cocoapi/annotations/captions_{}.json'.format(dataType))
coco_caps = COCO(captions_annFile)

# get image ids
ids = list(coco.anns.keys())


img = [{'img_id': int(file.split('.')[0]), 'file': file} for file in os.listdir('data/cocoapi/images/train2017')]
img_ids = [x['img_id'] for x in img]

# get captions for
annIds = coco_caps.getAnnIds(imgIds=img_ids)
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)

# if not os.path.exists('./data/anns-50.csv'):
data = pd.DataFrame(anns).set_index('image_id')
img_df = pd.DataFrame(img).set_index('img_id')
data = data.join(img_df)
data.to_csv('./data/anns-50.csv')






