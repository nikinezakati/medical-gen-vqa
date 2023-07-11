import os
from os import listdir
from PIL import Image
from tqdm import tqdm
import torch
from torch.nn import LSTM
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import pickle
import numpy as np
import json
import sys

from src.constants import IMGS_DIR, IMGS_RESIZED_DIR, IMG_FEATTURES_DIR

class Image_Encoder(nn.Module):

    def __init__(self, embed_dim):

        super(Image_Encoder, self).__init__()
        self.model = models.vgg19(pretrained=True)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1]) # remove vgg19 last layer
        self.fc = nn.Linear(in_features, embed_dim)

    def forward(self, image):

        with torch.no_grad():
            img_feature = self.model(image) # (batch, channel, height, width)
        img_feature = self.fc(img_feature)

        l2_norm = F.normalize(img_feature, p=2, dim=1).detach()
        return l2_norm
    

def extract_features(img_dir=IMGS_DIR, resized_dir=IMGS_RESIZED_DIR, feat_dir=IMG_FEATTURES_DIR):
    image_ids = []
    for dir in os.listdir(resized_dir):
        if dir != '.DS_Store':
            id = ''.join(char for char in dir if char.isdigit())
            if int(id)>641:
                continue
            image_ids.append(int(id))
    image_ids = sorted(image_ids)

    img_boxes = []

    for id in tqdm(image_ids):
        img = Image.open(img_dir+'xmlab'+str(id)+'/source.jpg').convert('RGB')
        height = img.height

        f = open(img_dir+'xmlab'+str(id)+'/detection.json', encoding="utf8")
        detection_data = json.load(f)
        f.close()
        lst=[]
        for data in detection_data:
            boxes = (list(data.values()))
            temp = [i * (224/height) for i in boxes[0]]
            lst.append(torch.FloatTensor(temp))

        print("Adding boxes of image ", id)
        if len(lst) ==0:
            lst = torch.zeros(1,4)

        padded_box = torch.nn.utils.rnn.pad_sequence(lst, batch_first=True, padding_value=0.)
        img_boxes.append(padded_box)

    
    image_trans=[]
    image_features=[]


    folder_dir = resized_dir

    transform = transforms.Compose([
            transforms.ToTensor(),  # convert to (C,H,W) and [0,1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # mean=0; std=1
        ])


    img_encoder = Image_Encoder(1024)

    for start_idx in tqdm(range(0, len(image_ids), 32)):
        ids = image_ids[start_idx: start_idx + 32]

        for image_id in ids:
            img = np.array(Image.open(folder_dir +str(image_id)+'.png').convert('RGB'))
            print('Transforming image with id:', image_id)
            img = transform(img)
            image_trans.append(img)

        image_trans = torch.stack(image_trans, dim = 0)
        image_features = img_encoder(image_trans)
        image_trans = []

        for idx,id in enumerate(ids):
            dct = {}
            dct['img_id'] = id
            dct['features'] = torch.FloatTensor(image_features[idx])
            dct['boxes'] = torch.FloatTensor(img_boxes[id])
            dct['img_h'] = 224
            dct['img_w'] = 224

            print('Saving features of image with id:', str(id))
            with open(feat_dir + str(id)+'.pkl', 'wb') as f:
                pickle.dump(dct, f)
    


if __name__ == '__main__':
    try:
        feats_dir = sys.argv[1]
    except:
        feats_dir = IMG_FEATTURES_DIR
    extract_features(feats_dir)      