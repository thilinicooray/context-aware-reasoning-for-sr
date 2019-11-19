'''
This is the full CNN classifier for verb or if any single role classification needed.
'''

import torch
import torch.nn as nn

import torchvision as tv

class vgg16_modified(nn.Module):
    def __init__(self, num_classes):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        self.vgg_features = vgg.features
        self.num_ans_classes = num_classes

        num_features = vgg.classifier[6].in_features
        features = list(vgg.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, num_classes)]) # Add our layer with 4 outputs
        self.classifier = nn.Sequential(*features)


    def forward(self,x):
        features = self.vgg_features(x)
        out = self.classifier(features.view(-1, 512*7*7))

        return out

    def calculate_verb_loss(self, verb_pred, gt_verbs):

        verb_criterion = nn.CrossEntropyLoss()
        verb_loss = verb_criterion(verb_pred, gt_verbs.squeeze())
        return verb_loss

def build_vgg_verb_classifier(num_ans_classes):

    covnet = vgg16_modified(num_ans_classes)

    return covnet


