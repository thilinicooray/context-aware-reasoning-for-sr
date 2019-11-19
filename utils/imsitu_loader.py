'''
Loading dataset for training and evaluation based on different model information requirements
'''

import torch.utils.data as data
from PIL import Image
import os
import random
import torch
import pickle as cPickle
import h5py
import numpy as np

class imsitu_loader(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, dictionary, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.dictionary = dictionary
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        img = self.transform(img)

        verb, labels = self.encoder.encode(ann)
        return _id, img, verb, labels

    def __len__(self):
        return len(self.annotations)


class imsitu_loader_agent(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, dictionary, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.dictionary = dictionary
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        img = self.transform(img)

        labels = self.encoder.encode_agent(ann)
        return _id, img, labels

    def __len__(self):
        return len(self.annotations)


class imsitu_loader_place(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, dictionary, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.dictionary = dictionary
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        img = self.transform(img)

        labels = self.encoder.encode_place(ann)
        return _id, img, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_verb(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, dictionary, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.dictionary = dictionary
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        img = self.transform(img)

        verb = self.encoder.encode_verb(ann)
        return _id, img, verb

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_agentplace_4_verb(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, split, transform=None, dataroot='data'):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        img = self.transform(img)

        verb = self.encoder.encode_verb(ann)
        agents = self.encoder.encode_agent(ann)
        places = self.encoder.encode_place(ann)


        return _id, img, verb, agents, places

    def __len__(self):
        return len(self.annotations)