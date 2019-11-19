'''
TDA model for verb predictions
'''

import torch
import torch.nn as nn
import torchvision as tv

from lib.attention import Attention
from lib.classifier import SimpleClassifier
from lib.fc import FCNet



class vgg16_modified(nn.Module):
    def __init__(self):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        self.vgg_features = vgg.features

    def forward(self,x):
        features = self.vgg_features(x)
        return features

class Top_Down_Verb(nn.Module):
    def __init__(self, covnet, role_module, label_emb, query_composer, v_att, q_net,
                 v_net, resize_img_flat, classifier, Dropout_C):
        super(Top_Down_Verb, self).__init__()
        self.convnet = covnet
        self.role_module = role_module
        self.label_emb = label_emb
        self.query_composer = query_composer
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.resize_img_flat = resize_img_flat
        self.classifier = classifier
        self.Dropout_C = Dropout_C
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Dropout_extctx= nn.Dropout(0.5)

    def forward(self, v_org):

        #given image, get agent and place predictions for query composing and context gen
        agent_place_pred, agent_place_rep = self.role_module.forward_agentplace_noverb(v_org)

        role_rep_combo = torch.sum(agent_place_rep, 1)

        label_idx = torch.max(agent_place_pred,-1)[1].squeeze()
        agent_embd = self.label_emb(label_idx[:,0])
        place_embd = self.label_emb(label_idx[:,1])
        concat_query = torch.cat([ agent_embd, place_embd], -1)
        q_emb = self.Dropout_C(self.query_composer(concat_query))

        img_features = self.convnet(v_org)
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        img_feat_flat = self.avg_pool(img_features)
        img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())
        #soft query based context
        ext_ctx = img_feat_flat * role_rep_combo

        img_org = img_features.view(batch_size, -1, conv_h* conv_w)
        v = img_org.permute(0, 2, 1)

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)
        v_repr = self.v_net(v_emb)
        q_repr = self.q_net(q_emb)

        out = q_repr * v_repr + ext_ctx

        logits = self.classifier(out)

        return logits

    def calculate_verb_loss(self, verb_pred, gt_verbs):

        criterion = nn.CrossEntropyLoss()

        loss = criterion(verb_pred, gt_verbs.squeeze())

        return loss

def build_top_down_verb(num_labels, num_ans_classes, role_module):

    hidden_size = 1024
    word_embedding_size = 300
    img_embedding_size = 512

    covnet = vgg16_modified()
    role_module = role_module
    label_emb = nn.Embedding(num_labels + 1, word_embedding_size, padding_idx=num_labels)
    query_composer = FCNet([word_embedding_size * 2, hidden_size])
    v_att = Attention(img_embedding_size, hidden_size, hidden_size)
    q_net = FCNet([hidden_size, hidden_size ])
    v_net = FCNet([img_embedding_size, hidden_size])
    resize_img_flat = nn.Linear(img_embedding_size, hidden_size)
    classifier = SimpleClassifier(
        hidden_size, 2 * hidden_size, num_ans_classes, 0.5)

    Dropout_C = nn.Dropout(0.1)

    return Top_Down_Verb(covnet, role_module, label_emb, query_composer, v_att, q_net,
                         v_net, resize_img_flat, classifier, Dropout_C)


