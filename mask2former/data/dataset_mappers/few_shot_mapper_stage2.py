# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import random

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

from mask2former.data.utils import *
from torchvision import transforms
# import mask2former.data.transforms as tr
from PIL import Image

__all__ = ["FewShotDatasetMapper_stage2"]



class FewShotDatasetMapper_stage2:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        split,
        shot,
        ignore_bg,
        dataname,
        MYSIZE,
        root,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value

        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.split = split
        self.class_list, self.sub_list, self.sub_val_list = shot_generate(split, dataname)
        self.shot = shot
        self.ignore_bg = ignore_bg
        self.MYSIZE = MYSIZE
        self.root = root

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}"
        )

        _ , self.sub_class_file_list = make_dataset(split, dataname, is_train)


    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if is_train:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            ]
            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    )
                )
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs.append(T.RandomFlip())

            # Assume always applies to the training set.
            dataset_names = cfg.DATASETS.TRAIN
            augs2 = augs

        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            augs = [T.Resize(min_size)]
            dataset_names = cfg.DATASETS.TEST

        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = 255

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY if is_train else -1,
            "split": cfg.DATASETS.SPLIT,
            "shot": cfg.DATASETS.SHOT,
            "ignore_bg": cfg.DATASETS.IGNORE_BG,
            "dataname": cfg.DATASETS.dataname,
            "MYSIZE": cfg.INPUT.MYSIZE,
            "root": cfg.DATASETS.IMGPATHROOT
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        assert os.path.isfile(dataset_dict["file_name"]), dataset_dict["file_name"]
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        sem_seg_gt = utils.read_image(dataset_dict["sem_seg_file_name"]).astype("double")

        label_class = []
        ii = 0
        while len(label_class) == 0:     # 防止support mask 为空
            ii = ii+1
            if ii > 100:
                raise ValueError( 'place 1, iter more than 100!!!')
            image_, sem_seg_gt_ = self.aug(image, sem_seg_gt)
            label = sem_seg_gt_.clone().numpy()
            label_class = np.unique(label).tolist()
            if 0 in label_class:
                label_class.remove(0)
            if 255 in label_class:
                label_class.remove(255) 
            new_label_class = []       
            # print(label_class, 'x'*100, label_path)
            # label_classlabel_class = label_class ###
            for c in label_class:
                # assert self.is_train
                if c in self.sub_val_list:
                    if not self.is_train:
                        new_label_class.append(c)
                    else: 
                        sem_seg_gt_[sem_seg_gt_ == c] = 255   # mask unseen class when training
                if c in self.sub_list:
                    if self.is_train:
                        new_label_class.append(c)
            label_class = new_label_class    
        image, sem_seg_gt = image_, sem_seg_gt_
        
        # subcls_list
        if self.is_train:
            class_chosen = label_class[random.randint(1,len(label_class))-1]
            subcls_list = [self.sub_list.index(class_chosen)]
            support_image_list, support_label_list, support_label_for_seg_list = self.get_sup_img_pairs(class_chosen, dataset_dict["file_name"])
        else:
            class_chosen = dataset_dict["class_chosen"]
            subcls_list = [self.sub_val_list.index(class_chosen)]
            support_image_list, support_label_list, support_label_for_seg_list = self.get_sup_img_pairs_val(dataset_dict)

        label = self.get_label_pairs(label, class_chosen)



        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image  # 3*h*w
        dataset_dict['label'] = torch.from_numpy(label).float()
        dataset_dict['subcls_list'] = subcls_list


        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)
        s_xs = s_x
        s_ys = s_y

        dataset_dict['support_img'] =  s_xs   # shot*3*h*w
        dataset_dict['support_label'] =  s_ys # shot*h*w


        return dataset_dict

    def get_label_pairs(self, label, class_chosen):
        target_pix = np.where(label == class_chosen)
        ignore_pix = np.where(label == 255)
        label_ = np.zeros_like(label)
        if target_pix[0].shape[0] > 0:
            label_[target_pix[0],target_pix[1]] = 1 
        label_[ignore_pix[0],ignore_pix[1]] = 255         
        return label_

    def aug(self, image, sem_seg_gt):
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image_ = aug_input.image
        sem_seg_gt_ = aug_input.sem_seg
 
        # Pad image and segmentation label here!
        image_ = torch.as_tensor(np.ascontiguousarray(image_.transpose(2, 0, 1)))
        sem_seg_gt_ = torch.as_tensor(sem_seg_gt_.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image_.shape[-2], image_.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image_ = F.pad(image_, padding_size, value=128).contiguous()
            if sem_seg_gt_ is not None:
                sem_seg_gt_ = F.pad(
                    sem_seg_gt_, padding_size, value=self.ignore_label
                ).contiguous()
        
        return image_, sem_seg_gt_

    def get_sup_img_pairs(self, class_chosen, image_path):

        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)
        check_num = True
        if len(file_class_chosen)<=self.shot:
            check_num = False

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1,num_file)-1
            support_image_path = image_path
            # support_label_path = label_path
            while((support_image_path == image_path) or (check_num and support_idx in support_idx_list)):
                support_idx = random.randint(1,num_file)-1
                support_image_path, support_label_path = file_class_chosen[support_idx] 
                support_image_path = os.path.join(self.root, support_image_path)
                support_label_path = os.path.join(self.root, support_label_path)                   
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)


        support_image_list = []
        support_label_list = []
        support_label_for_seg_list = []

        for k in range(self.shot):  

            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k] 
            support_image = utils.read_image(support_image_path, format=self.img_format)
            support_label = utils.read_image(support_label_path).astype("double")


            label_class_sup_exit = []
            while len(label_class_sup_exit) == 0:     # 防止support mask 为空
                support_image_, support_label_ = self.aug(support_image, support_label)
                label_sup = support_label_.clone().numpy()
                label_class_sup = np.unique(label_sup).tolist()
                for class_sup in label_class_sup:
                    if class_sup == class_chosen:
                        tmp_label = np.zeros_like(label_sup)
                        target_pix = np.where(label_sup == class_sup)
                        tmp_label[target_pix[0],target_pix[1]] = 1 
                        if (tmp_label.sum() >= 1.8 * 32 * 32)  or (not self.is_train) :  
                            label_class_sup_exit.append(1)
            support_image, support_label_for_seg = support_image_, support_label_
            support_label = support_label_for_seg.clone().numpy()


            # support_image, support_label_for_seg = self.aug(support_image, support_label)
            # support_label = support_label_for_seg.clone().numpy()

            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:,:] = 0
            support_label[target_pix[0],target_pix[1]] = 1 
            support_label[ignore_pix[0],ignore_pix[1]] = 255
            if support_image.shape[1] != support_label.shape[0] or support_image.shape[2] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))            
            support_image_list.append(support_image)
            support_label_list.append(torch.from_numpy(support_label).float())
            support_label_for_seg_list.append(support_label_for_seg)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot  

        return support_image_list, support_label_list, support_label_for_seg_list

    def get_sup_img_pairs_val(self, record):

        support_image_path_list = record["support_image_path_list"]
        support_label_path_list = record["support_label_path_list"]
        class_chosen = record["class_chosen"]


        support_image_list = []
        support_label_list = []
        support_label_for_seg_list = []

        for k in range(self.shot):  

            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k] 
            support_image = utils.read_image(support_image_path, format=self.img_format)
            support_label = utils.read_image(support_label_path).astype("double")


            support_image, support_label_for_seg = self.aug(support_image, support_label)
            support_label = support_label_for_seg.clone().numpy()

            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:,:] = 0
            support_label[target_pix[0],target_pix[1]] = 1 
            support_label[ignore_pix[0],ignore_pix[1]] = 255
            if support_image.shape[1] != support_label.shape[0] or support_image.shape[2] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))            
            support_image_list.append(support_image)
            support_label_list.append(torch.from_numpy(support_label).float())
            support_label_for_seg_list.append(support_label_for_seg)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot  
        
        return support_image_list, support_label_list, support_label_for_seg_list