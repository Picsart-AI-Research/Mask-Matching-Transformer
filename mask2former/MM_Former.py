# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch, os, cv2, math
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.fewshot_loss import WeightedDiceLoss
from .modeling.feature_alignment.self_align import MySelfAlignLayer
from .modeling.feature_alignment.cross_align import CrossAT
from .modeling.transformer_decoder.position_encoding import PositionEmbeddingSine

@META_ARCH_REGISTRY.register()
class MMFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # few shot
        shot: int,
        fewshot_weight: float,
        pre_norm: bool,
        conv_dim:int

    ):

        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.criterion_for_fewshot = WeightedDiceLoss()   #####
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.shot = shot
        self.fewshot_weight = fewshot_weight

        i = 0
        for dim, head, inc in [(512, 4, 2048),(512, 4, 1024),(512, 4, 512)]:  
            k = CrossAT(dim, nhead = head, pre_norm = pre_norm, num_layer = 2)
            self.add_module(f'crossat_{i}', k)
            self.add_module(f'conv_{i}', nn.Conv2d(inc, 256, kernel_size=1, stride=1, padding=0, bias=False))
            i = i + 1

        self.add_module(f'crossat_{i}', CrossAT(256, nhead = 2, pre_norm = pre_norm))

        self.fc1 = nn.Linear(num_queries, num_queries*5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_queries*5, num_queries)

        self.pe_layer = PositionEmbeddingSine(conv_dim//2, normalize=True)
        self.myat = MySelfAlignLayer()
        self.ranking = nn.BCELoss()

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # few shot
            "shot": cfg.DATASETS.SHOT,
            'fewshot_weight': cfg.MODEL.MASK_FORMER.FEWSHOT_WEIGHT,
            "pre_norm": cfg.MODEL.MASK_FORMER.PRE_NORM,
            "conv_dim": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        with torch.no_grad():
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)   # , labels
            mask_features = outputs["pred_masks"].sigmoid()


        # trans sup_label from bs*shot*h*w to shot*bs*h*w
        sup_label = [x["support_label"].to(self.device) for x in batched_inputs]   # bs*shot*h*w
        bs = len(sup_label)
        for i in range(bs):
            sup_label[i] = sup_label[i].unsqueeze(0)
        sup_label = torch.cat(sup_label, dim = 0).permute(1, 0, 2, 3)
        sup_label[sup_label==255] = 0



        supfeatures = []
        supfeatures_for_propotype = []
        supoutputs = []
        # outputs_for_sup = []
        for i in range(self.shot):
            sup_images = [x["support_img"][i].to(self.device) for x in batched_inputs]
            sup_images = [(x - self.pixel_mean) / self.pixel_std for x in sup_images]
            sup_images = ImageList.from_tensors(sup_images, self.size_divisibility)

            with torch.no_grad():
                supfeature = self.backbone(sup_images.tensor)  # sup_images.tensor
                supfeatures.append(supfeature)


        
        out_que = [features['res5'].float(), features['res4'].float(), features['res3'].float()]
        
        pos_que = []
        for i in range(len(out_que)):
            out_que[i] = self.myat(out_que[i])

        out_que = self.channel_change(out_que)
        for i in range(len(out_que)):
            pos_i = self.pe_layer(out_que[i])
            pos_que.append(pos_i)

        que_mask_embeds, sup_mask_embeds_best, que_mask_embeds_bg = None, None, None
        for i in range(self.shot):
            out_sup = [supfeatures[i]['res5'].float(), supfeatures[i]['res4'].float(), supfeatures[i]['res3'].float()]

            out_que_i = []
            for ii in range(3):
                out_sup[ii] = self.myat(out_sup[ii])

            out_sup = self.channel_change(out_sup)
            out_sup, pos_sup = self.addmask(out_sup, sup_label[i])
            for ii in range(3):
                crossat = getattr(self, f'crossat_{ii}')
                out_i, out_sup[ii] = crossat(out_que[ii], out_sup[ii], pos_que[ii], pos_sup[ii])
                out_que_i.append(out_i)

            # out_que[-1], out_sup[-1] = self.crossat_3(out_que[-1], out_sup[-1], pos_que[-1], pos_sup[-1], minsize=30)

            sup_i = []
            que_i = []
            que_i_bg = []
            
            for ii in range(3):
                sup_i.append(self.get_sup_propotype(out_sup[ii], sup_label[i]))
                que_i.append(self.get_que_propotype(out_que_i[ii], mask_features))
                # que_i_bg.append(self.get_que_propotype(out_que[ii], 1.0-mask_features))

            if que_mask_embeds is None:
                que_mask_embeds = torch.cat(que_i, dim = 1).transpose(1, 2)
                sup_mask_embeds_best = torch.cat(sup_i, dim = 1).transpose(1, 2)
                # que_mask_embeds_bg = torch.cat(que_i_bg, dim = 1).transpose(1, 2)

            else:
                que_mask_embeds = que_mask_embeds + torch.cat(que_i, dim = 1).transpose(1, 2)
                sup_mask_embeds_best = sup_mask_embeds_best + torch.cat(sup_i, dim = 1).transpose(1, 2)
                # que_mask_embeds_bg = que_mask_embeds_bg + torch.cat(que_i_bg, dim = 1).transpose(1, 2)
        
        que_mask_embeds = que_mask_embeds/self.shot
        sup_mask_embeds_best = sup_mask_embeds_best/self.shot
        # que_mask_embeds_bg = que_mask_embeds_bg/self.shot
        # print(que_mask_embeds.shape, sup_mask_embeds_best.shape)

        embeding = F.cosine_similarity(que_mask_embeds, sup_mask_embeds_best.repeat(1,100,1), dim = 2).unsqueeze(1)
        all_similar = self.fc2(self.relu(self.fc1(embeding))).squeeze(1)
        # all_similar = F.softmax(all_similar,dim =1)

        mask_features = F.interpolate(
            mask_features,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        out = torch.einsum("bq,bqhw->bhw", all_similar, mask_features).unsqueeze(1)
        dout_bg = 1 - out
        out_all = torch.cat([dout_bg, out],1)

        if self.training:

            # label
            label = [x["label"].to(self.device) for x in batched_inputs]   # bs*h*w
            bs = len(label)
            for i in range(bs):
                label[i] = label[i].unsqueeze(0)
            label = torch.cat(label, dim = 0)
            labels = label.clone()
            labels[label == 255] = 0

            bs_idx = torch.range(0, labels.shape[0]-1).long()
            ious = self.get_iou(mask_features, labels).detach()  # bs*100
            _, min_iou = torch.min(ious, dim = 1)
            _, max_iou = torch.max(ious, dim = 1)
            
            embeding = embeding.squeeze(1)
            min_em, _ = torch.min(embeding, dim = 1)
            max_em, _ = torch.max(embeding, dim = 1)
            embeding = (embeding-min_em.unsqueeze(-1))/((max_em-min_em+0.00000001).unsqueeze(-1))

            bg = embeding[(bs_idx, min_iou)]
            fg = embeding[(bs_idx, max_iou)]
            iou_label = torch.cat([torch.zeros(bs, 1), torch.ones(bs, 1)], dim = -1).to(ious.device)

            em = torch.cat([bg.unsqueeze(-1), fg.unsqueeze(-1)], dim = -1)
            losses_for_co =  self.ranking(em, iou_label)


            losses = {}
            losses_for_fewshot = self.criterion_for_fewshot(out_all, labels) 
            losses["fewshot_loss"] = losses_for_fewshot * self.fewshot_weight
            losses["co_loss"] = losses_for_co * self.fewshot_weight * 0.6

            return losses
        else:
            processed_results = {"few_shot_result": out_all}

            return processed_results


    def get_iou(self, pred, target):
        # pred = pred.sigmoid() 
        b, c, h, w = pred.shape
        target = target.unsqueeze(1)
        # print(pred.shape, target.shape)
        # assert pred.shape == target.shape
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
            pred,
            size=(target.shape[-2], target.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )


        pred = pred.reshape(b, c,-1)
        target = target.reshape(b, 1, -1)
        
        #compute the IoU of the foreground
        Iand1 = torch.sum(target*pred, dim = -1)
        Ior1 = torch.sum(target, dim = -1) + torch.sum(pred, dim = -1)-Iand1 + 0.0000001
        IoU1 = Iand1/Ior1

        return IoU1


    def addmask(self,out_sup, suplabels_ori):
        suplabels = suplabels_ori.clone().unsqueeze(1).float()
        # b,c,h,w = suplabels.shape
        # suplabels = suplabels.unsqueeze(1).repeat(2,1,1,1,1).view(-1,c,h,w)

        bs = out_sup[0].shape[0]
        
        pos_sup = []

        for i in range(len(out_sup)):
            suplabel = F.interpolate(suplabels, size=out_sup[i].shape[-2:], mode='bilinear')
            pos_i = self.pe_layer(out_sup[i])
            out_sup[i] = out_sup[i] * suplabel
            pos_sup.append(pos_i * suplabel)
        
        return out_sup, pos_sup

    def get_sup_propotype(self, features_for_propotype, label):

        label = F.interpolate(label.unsqueeze(1), size=features_for_propotype.shape[-2:], mode='bilinear', align_corners=True)
        weight = torch.sum(label, dim = (2,3))  # bs* 1

        propotype = features_for_propotype * label  # bs* 256*h*w
        propotype = torch.sum(propotype, dim = (2,3))/ (weight + 0.0000001)

        sup_propotype = propotype.unsqueeze(-1)

        return sup_propotype

    def get_que_propotype(self, features_for_propotypes, label):
        label = F.interpolate(label, size=features_for_propotypes.shape[-2:], mode='bilinear', align_corners=True)
        weight = torch.sum(label, dim = (2,3))  # bs* 100

        features_for_propotypes = features_for_propotypes.unsqueeze(2) # bs * 256 * 1 * h*w
        label = label.unsqueeze(1)                                     # bs * 1 * 100 * h*w
        propotype = features_for_propotypes * label
        propotype = torch.sum(propotype, dim = (3,4))/ weight.unsqueeze(1)

        return propotype

    def channel_change(self, out_feature):
        for i in range(len(out_feature)):
            conv = getattr(self, f'conv_{i}')
            out_feature[i] = conv(out_feature[i])
        return out_feature
