# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets import load_sem_seg
from mask2former.data.utils import *

CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

def _get_voc_meta(cat_list):
    ret = {
        "stuff_classes": cat_list,
    }
    return ret


def register_all_pascal(root, split, BASE_CLASS_NAMES, NOVEL_CLASS_NAMES):
    meta = _get_voc_meta(CLASS_NAMES)
    base_meta = _get_voc_meta(CLASS_NAMES)

    novel_meta = _get_voc_meta(NOVEL_CLASS_NAMES)

    for name, image_dirname, sem_seg_dirname in [
        ("train", "JPEGImages", "SegmentationClassAug"),
        ("val", "JPEGImages", "SegmentationClassAug")
    ]:
        image_dir = os.path.join(root, 'JPEGImages')
        gt_dir = os.path.join(root, 'SegmentationClassAug')
        
        # print(all_name, 'x'*500)
        d = 'list/pascal/voc_' + name + '_list_split' + str(split) + '.pth'
        if name=="train":
            all_name = f"voc_fewshot_sem_seg_{name}_{str(split)}"
            DatasetCatalog.register(
                all_name,
                lambda d=d: load_fewshot_voc_seg(d, root),
                # lambda split=split, name=name: load_fewshot_voc_seg(split, name),
            )
            MetadataCatalog.get(all_name).set(
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="sem_seg",
                ignore_label=255,
                **base_meta,
            )
        elif name=="val":
            shot = 1
            all_name = f"voc_fewshot_sem_seg_{name}_{str(split)}_1shot"
            DatasetCatalog.register(
                all_name,
                lambda d=d: load_fewshot_val1000(d, name, split = split, shot = shot, dataname = 'pascal', root = root),
                # lambda split=split, name=name: load_fewshot_voc_seg(split, name),
            )

            MetadataCatalog.get(all_name).set(
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="sem_seg",
                ignore_label=255,
                **base_meta,
            )

            shot = 5
            all_name = f"voc_fewshot_sem_seg_{name}_{str(split)}_5shot"
            DatasetCatalog.register(
                all_name,
                lambda d=d: load_fewshot_val1000(d, name, split = split, shot = shot, dataname = 'pascal', root = root),
                # lambda split=split, name=name: load_fewshot_voc_seg(split, name),
            )

        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **base_meta,
        )


_root = '/home/siyujiao/data/pascal/VOCdevkit/VOC2012'

for split in range(4):
    class_list, sub_list, sub_val_list = shot_generate(split, dataname='pascal')
    BASE_CLASS_NAMES = [c for i, c in enumerate(CLASS_NAMES) if i in sub_list]
    NOVEL_CLASS_NAMES = [c for i, c in enumerate(CLASS_NAMES) if i in sub_val_list]

    register_all_pascal(_root, split, BASE_CLASS_NAMES, NOVEL_CLASS_NAMES)

