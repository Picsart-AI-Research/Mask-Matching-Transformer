# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

from .data import (
    build_detection_test_loader,
    build_detection_train_loader,
    dataset_sample_per_class,
)
from .evaluation.fewshot_sem_seg_evaluation import (
    FewShotSemSegEvaluator,
)

# models

from .Potential_Objects_Segmenter import POS
from .MM_Former import MMFormer 


from .test_time_augmentation import SemanticSegmentorWithTTA
# evaluation
# from .evaluation.instance_evaluation import InstanceSegEvaluator
from .evaluation.fewshot_sem_seg_evaluation import FewShotSemSegEvaluator