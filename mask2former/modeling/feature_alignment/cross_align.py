# CrossAT : cross alignment block in MM-Former
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.mask2former_transformer_decoder import CrossAttentionLayer, FFNLayer


class CrossAT(nn.Module):
    def __init__(self, dim_feedforward, pre_norm, hidden_dim = 256, nhead = 8, num_layer = 2):
        super().__init__()

        self.num_layer = num_layer
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layer):
            self.transformer_cross_attention_layers.append(CrossAttentionLayer(d_model=hidden_dim,nhead=nhead,
                                                                                dropout=0.0,normalize_before=pre_norm,))
            self.transformer_ffn_layers.append(FFNLayer(d_model=hidden_dim,dim_feedforward=dim_feedforward,
                                                        dropout=0.0,normalize_before=pre_norm,))


    def forward(self, out_que, out_sup, pos_que, pos_sup, label = None, minsize = 15):
        not_scale = True
        bs, c, h, w = out_que.shape
        if h > w:
            ow = minsize
            oh = int(1.0 * h * ow / w)
        else:
            oh = minsize
            ow = int(1.0 * w * oh / h)

        sh, sw = out_sup.shape[-2:]
        if sh > sw:
            osw = minsize
            osh = int(1.0 * sh * osw / sw)
        else:
            osh = minsize
            osw = int(1.0 * sw * osh / sh)

        if not (h == minsize or w == minsize):
            ori_out_que = F.interpolate(out_que, size=(oh, ow), mode='bilinear').flatten(2).permute(2, 0, 1)
            ori_out_sup = F.interpolate(out_sup, size=(osh, osw), mode='bilinear').flatten(2).permute(2, 0, 1)
            ori_pos_que = F.interpolate(pos_que, size=(oh, ow), mode='bilinear').flatten(2).permute(2, 0, 1)
            ori_pos_sup = F.interpolate(pos_sup, size=(osh, osw), mode='bilinear').flatten(2).permute(2, 0, 1)
            not_scale = False
        out_que = out_que.flatten(2).permute(2, 0, 1)
        out_sup = out_sup.flatten(2).permute(2, 0, 1)
        pos_sup = pos_sup.flatten(2).permute(2, 0, 1)
        pos_que = pos_que.flatten(2).permute(2, 0, 1)
        if not_scale:
            ori_out_que = out_que
            ori_out_sup = out_sup
            ori_pos_que = pos_que
            ori_pos_sup = pos_sup
        # print("pos_sup:", pos_sup.shape, out_sup.shape)

        for i in range(self.num_layer):
            out_que = self.transformer_cross_attention_layers[i](
                out_que, ori_out_sup,
                pos=ori_pos_sup, query_pos=pos_que
                # pos=pos_que, query_pos=pos_sup
            )
            # FFN
            out_que = self.transformer_ffn_layers[i](out_que)

        for i in range(self.num_layer):
            # print(out_sup.shape, ori_out_que.shape, pos_sup.shape, ori_pos_que.shape)
            out_sup = self.transformer_cross_attention_layers[i](
                out_sup, ori_out_que,
                pos=ori_pos_que, query_pos=pos_sup
                # pos=pos_sup, query_pos=ori_pos_que
            )
            # FFN
            out_sup = self.transformer_ffn_layers[i](out_sup)

        return out_que.permute(1, 2, 0).view(bs, c, h, w), out_sup.permute(1, 2, 0).view(bs, c, sh, sw)

