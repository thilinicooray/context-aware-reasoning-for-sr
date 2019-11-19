'''
Situation inference code using pretrained CAQ based role model and Resoning-Enhanced VGG based verb model
'''

import torch
import torch.nn as nn

class Top_Down_Baseline(nn.Module):
    def __init__(self, vgg_model, tda_model, caq_model):
        super(Top_Down_Baseline, self).__init__()
        self.vgg_model = vgg_model
        self.tda_model = tda_model
        self.caq_model = caq_model

    def forward(self, v_org, topk=5):

        verb_pred = self.vgg_model(v_org) + self.tda_model(v_org)

        role_pred_topk = None

        sorted_idx = torch.sort(verb_pred, 1, True)[1]
        verbs = sorted_idx[:,:topk]

        for k in range(0,topk):
            role_pred = self.caq_model(v_org, verbs[:,k])

            if k == 0:
                idx = torch.max(role_pred,-1)[1]
                role_pred_topk = idx
            else:
                idx = torch.max(role_pred,-1)[1]
                role_pred_topk = torch.cat((role_pred_topk.clone(), idx), 1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return verbs, role_pred_topk

def build_vgg_caq_joint(vgg_model, tda_model, caq_model):

    return Top_Down_Baseline(vgg_model, tda_model, caq_model)


