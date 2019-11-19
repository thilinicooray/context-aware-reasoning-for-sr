'''
CAIR model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lib.attention import Attention
from lib.classifier import SimpleClassifier
from lib.fc import FCNet
import torchvision as tv

class vgg16_modified(nn.Module):
    def __init__(self):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        self.vgg_features = vgg.features

    def forward(self,x):
        features = self.vgg_features(x)
        return features


class Top_Down_Img_Recons(nn.Module):
    def __init__(self, convnet, role_emb, verb_emb, query_composer, v_att, q_net, v_net, Dropout_C, flatten_img, classifier, encoder, reconstruct_img):
        super(Top_Down_Img_Recons, self).__init__()
        self.convnet = convnet
        self.role_emb = role_emb
        self.verb_emb = verb_emb
        self.query_composer = query_composer
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.Dropout_C = Dropout_C
        self.flatten_img = flatten_img
        self.classifier = classifier
        self.encoder = encoder
        self.reconstruct_img = reconstruct_img

    def forward(self, v_org, gt_verb):

        img_features = self.convnet(v_org)
        flattened_img = self.flatten_img(img_features.view(-1, 512*7*7))
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, -1, conv_h* conv_w)
        v = img_org.permute(0, 2, 1)

        batch_size = v.size(0)

        role_idx = self.encoder.get_role_ids_batch(gt_verb)

        if torch.cuda.is_available():
            role_idx = role_idx.to(torch.device('cuda'))

        img = v

        img = img.expand(self.encoder.max_role_count, img.size(0), img.size(1), img.size(2))

        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))

        verb_embd = self.verb_emb(gt_verb)
        role_embd = self.role_emb(role_idx)

        verb_embed_expand = verb_embd.expand(self.encoder.max_role_count, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1)
        concat_query = torch.cat([ verb_embed_expand, role_embd], -1)
        role_verb_embd = concat_query.contiguous().view(-1, role_embd.size(-1)*2)
        q_emb = self.query_composer(role_verb_embd)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1)
        v_repr = self.v_net(v_emb)
        q_repr = self.q_net(q_emb)

        mfb_iq_eltwise = torch.mul(q_repr, v_repr)

        mfb_iq_drop = self.Dropout_C(mfb_iq_eltwise)

        mfb_iq_resh = mfb_iq_drop.view(batch_size* self.encoder.max_role_count, 1, -1, 1)
        mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)
        mfb_out = torch.squeeze(mfb_iq_sumpool)
        mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
        mfb_l2 = F.normalize(mfb_sign_sqrt)
        out = mfb_l2

        cur_group = out.contiguous().view(v.size(0), -1)
        #reconstruct image
        constructed_img = self.reconstruct_img(cur_group)

        logits = self.classifier(out)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        return role_label_pred, constructed_img, flattened_img

    def calculate_loss(self, gt_verbs, role_label_pred, gt_labels, constructed_img, flattened_img, beta=10):

        l2_criterion = nn.MSELoss()

        batch_size = role_label_pred.size()[0]
        criterion = nn.CrossEntropyLoss(ignore_index=self.encoder.get_num_labels())

        gt_label_turned = gt_labels.transpose(1,2).contiguous().view(batch_size* self.encoder.max_role_count*3, -1)

        role_label_pred = role_label_pred.contiguous().view(batch_size* self.encoder.max_role_count, -1)
        role_label_pred = role_label_pred.expand(3, role_label_pred.size(0), role_label_pred.size(1))
        role_label_pred = role_label_pred.transpose(0,1)
        role_label_pred = role_label_pred.contiguous().view(-1, role_label_pred.size(-1))

        loss = criterion(role_label_pred, gt_label_turned.squeeze(1)) * 3

        loss_recons = beta *l2_criterion(constructed_img, flattened_img)
        return loss, loss_recons

        return loss


def build_top_down_img_recons(n_roles, n_verbs, num_ans_classes, encoder):

    hidden_size = 1024
    word_embedding_size = 300
    img_embedding_size = 512

    covnet = vgg16_modified()
    role_emb = nn.Embedding(n_roles+1, word_embedding_size, padding_idx=n_roles)
    verb_emb = nn.Embedding(n_verbs, word_embedding_size)
    query_composer = FCNet([word_embedding_size * 2, hidden_size])
    v_att = Attention(img_embedding_size, hidden_size, hidden_size)
    q_net = FCNet([hidden_size, hidden_size ])
    v_net = FCNet([img_embedding_size, hidden_size])
    Dropout_C = nn.Dropout(0.1)

    flatten_img = nn.Sequential(
        nn.Linear(512*7*7, 1024),
        nn.BatchNorm1d(1024, momentum=0.01)
    )

    reconstruct_img = FCNet([hidden_size*6, hidden_size])

    classifier = SimpleClassifier(
        hidden_size, 2 * hidden_size, num_ans_classes, 0.5)

    return Top_Down_Img_Recons(covnet, role_emb, verb_emb, query_composer, v_att, q_net,
                             v_net, Dropout_C, flatten_img, classifier, encoder, reconstruct_img)


