import copy, random
import logging

from detectron2.data import detection_utils as utils

import numpy as np
import torch, tqdm, os, cv2
from torch.nn import functional as F


def shot_generate(split, dataname):  # use_coco
    if dataname in ['pascal', 'p2o']:
        class_list = list(range(1, 21)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        if split == 3: 
            sub_list = list(range(1, 16)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            sub_val_list = list(range(16, 21)) #[16,17,18,19,20]
        elif split == 2:
            sub_list = list(range(1, 11)) + list(range(16, 21)) #[1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
            sub_val_list = list(range(11, 16)) #[11,12,13,14,15]
        elif split == 1:
            sub_list = list(range(1, 6)) + list(range(11, 21)) #[1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(6, 11)) #[6,7,8,9,10]
        elif split == 0:
            sub_list = list(range(6, 21)) #[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(1, 6)) #[1,2,3,4,5]
    elif dataname == 'coco':
        class_list = list(range(1, 81))
        if split == 3:
            sub_val_list = list(range(4, 81, 4))
            sub_list = list(set(class_list) - set(sub_val_list))                    
        elif split == 2:
            sub_val_list = list(range(3, 80, 4))
            sub_list = list(set(class_list) - set(sub_val_list))    
        elif split == 1:
            sub_val_list = list(range(2, 79, 4))
            sub_list = list(set(class_list) - set(sub_val_list))    
        elif split == 0:
            sub_val_list = list(range(1, 78, 4))
            sub_list = list(set(class_list) - set(sub_val_list))    
    return class_list, sub_list, sub_val_list

def make_dataset(split, dataname, istrain = True):   
    class_list, sub_list, sub_val_list = shot_generate(split, dataname)
    if istrain:
        tv = 'train'
    else:
        tv = 'val'
        sub_list = sub_val_list 
    if dataname == 'coco':
        split_data_list="list/coco/" + tv + "_list_split" + str(split) + ".pth"
    elif dataname in ['pascal', 'p2o']:
        split_data_list="list/pascal/voc_" + tv+ "_list_split" + str(split) + ".pth"
    
    assert split in [0, 1, 2, 3, 10, 11, 999]

    if not os.path.isfile(split_data_list):
        raise (RuntimeError("Image list file do not exist: " + split_data_list + "\n"))

    # split_data_list = data_list.split('.')[0] + '_split{}'.format(split) + '.pth'
    if os.path.isfile(split_data_list):
        image_label_list, sub_class_file_list = torch.load(split_data_list)
        return image_label_list, sub_class_file_list
    else:
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))


def load_fewshot_voc_seg(list_dir, root = None):
    assert os.path.isfile(list_dir), list_dir
    image_label_list, _ = torch.load(list_dir)
    dataset_dicts = []
    for img_path, gt_path in image_label_list:
        record = {}
        record["file_name"] = os.path.join(root, img_path)
        record["sem_seg_file_name"] = os.path.join(root, gt_path)
        record["class_chosen"] = None
        dataset_dicts.append(record)
    return dataset_dicts

def load_fewshot_val1000(list_dir, name, split, shot, dataname = None, root = None):
    assert os.path.isfile(list_dir), list_dir
    image_label_list, _ = torch.load(list_dir)
    dataset_dicts = []
    for img_path, gt_path in image_label_list:
        record = {}
        record["file_name"] = os.path.join(root, img_path)
        record["sem_seg_file_name"] = os.path.join(root, gt_path)
        dataset_dicts.append(record)
    while len(dataset_dicts) <1000:
        for img_path, gt_path in image_label_list:
            record = {}
            record["file_name"] = os.path.join(root, img_path)
            record["sem_seg_file_name"] = os.path.join(root, gt_path)
            dataset_dicts.append(record)
    random.shuffle(dataset_dicts)    

    dataset_sup = []
    _, _, sub_val_list = shot_generate(split, dataname = dataname)
    for record in dataset_dicts[:1000]:
        img_path = record["file_name"]
        gt_path = record["sem_seg_file_name"]

        sem_seg_gt = utils.read_image(gt_path).astype("double")
        label_class = np.unique(sem_seg_gt).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255) 

        # generate label_classes
        new_label_class = []  
        for c in label_class:
            # assert self.is_train
            if c in sub_val_list:
                new_label_class.append(c)
        label_class = new_label_class    
        assert len(label_class)>0, (img_path, gt_path)

        # generate support set
        class_chosen = label_class[random.randint(1,len(label_class))-1]
        _, sub_class_file_list = make_dataset(split, dataname = dataname, istrain = False)
        file_class_chosen = sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)
        check_num = True
        if len(file_class_chosen)<=shot:
            check_num = False

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        subcls_list = []
        for k in range(shot):
            support_idx = random.randint(1,num_file)-1
            support_image_path = img_path
            support_label_path = gt_path
            while((support_image_path == img_path and support_label_path == gt_path) or (check_num and support_idx in support_idx_list)):
                support_idx = random.randint(1,num_file)-1
                support_image_path, support_label_path = file_class_chosen[support_idx]
                support_image_path = os.path.join(root, support_image_path)
                support_label_path = os.path.join(root, support_label_path)                
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        record["class_chosen"] = class_chosen
        record["support_image_path_list"] = support_image_path_list
        record["support_label_path_list"] = support_label_path_list

        dataset_sup.append(record)

    return dataset_sup

