
import os

def add_seed(cfg):
    if cfg.DATASETS.dataname == 'pascal':
        if cfg.DATASETS.SPLIT == 0:
            seed = 4604572
        elif  cfg.DATASETS.SPLIT == 1:
            seed = 7743485
        elif  cfg.DATASETS.SPLIT == 2:
            seed = 5448843
        elif  cfg.DATASETS.SPLIT == 3:
            seed = 2534673
    if cfg.DATASETS.dataname == 'coco':
        if cfg.DATASETS.SPLIT == 0:
            seed = 8420323
        elif  cfg.DATASETS.SPLIT == 1:
            seed = 27163933
        elif  cfg.DATASETS.SPLIT == 2:
            seed = 8162312
        elif  cfg.DATASETS.SPLIT == 3:
            seed = 3391510
    if cfg.DATASETS.dataname == 'c2pv':
        seed = 321
    return ['SEED', seed]

def add_step1dir(cfg):
    OUTPUT_DIR = os.path.join('out', cfg.DATASETS.dataname,'step1', cfg.MODEL.META_ARCHITECTURE, str(cfg.DATASETS.SPLIT))
    return ['OUTPUT_DIR', OUTPUT_DIR]

def add_step2dir(cfg):
    OUTPUT_DIR = os.path.join('out', cfg.DATASETS.dataname,'step2', cfg.MODEL.META_ARCHITECTURE, str(cfg.DATASETS.SPLIT))
    return ['OUTPUT_DIR', OUTPUT_DIR]

def add_dataset(cfg):
    DATASETS_TRAIN = (cfg.DATASETS.TRAIN[0] + str(cfg.DATASETS.SPLIT), )
    DATASETS_TEST = (cfg.DATASETS.TEST[0] + str(cfg.DATASETS.SPLIT) +'_'+ str(cfg.DATASETS.SHOT) + 'shot',)
    return ['DATASETS.TRAIN', DATASETS_TRAIN, 'DATASETS.TEST', DATASETS_TEST]