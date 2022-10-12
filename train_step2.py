import logging
import os, json
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from collections import OrderedDict
import copy
import itertools
from typing import Any, Dict, List, Set
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    # build_detection_test_loader,
    # build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
# from detectron2.solver import build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import EventStorage
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.utils.logger import setup_logger

from mask2former.utils.addcfg import *
from mask2former import (
    FewShotSemSegEvaluator,  ###
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)
from mask2former.data import (
    FewShotDatasetMapper_stage1,
    FewShotDatasetMapper_stage2,
    # FewShotDatasetMapper_ori_v2,
    build_detection_train_loader,
    build_detection_test_loader,
)


logger = logging.getLogger("detectron2")

def build_optimizer(cfg, model):
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

    defaults = {}
    defaults["lr"] = cfg.SOLVER.BASE_LR
    defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if "backbone" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
            if (
                "relative_position_bias_table" in module_param_name
                or "absolute_pos_embed" in module_param_name
            ):
                print(module_param_name)
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed
            params.append({"params": [value], **hyperparams})

    def maybe_add_full_model_gradient_clipping(optim):
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    elif optimizer_type == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg.SOLVER.BASE_LR
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    # semantic segmentation
    if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
        evaluator_list.append(
            FewShotSemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
                post_process_func=dense_crf_post_process
                if cfg.TEST.DENSE_CRF
                else None,
                dataname = cfg.DATASETS.dataname,
            )
        )

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model, data_loaders, evaluators):
    results = OrderedDict()
    for dataset_name, data_loader, evaluator in zip(cfg.DATASETS.TEST, data_loaders, evaluators):
        # if cfg.INPUT.DATASET_MAPPER_NAME == "fewshot_sem":
        #     mapper = FewShotDatasetMapper(cfg, False)
        # elif cfg.INPUT.DATASET_MAPPER_NAME == "fewshot_sem_ori":
        #     mapper = FewShotDatasetMapper_ori(cfg, False)
        # else:
        #     mapper = None
        # data_loader = build_detection_test_loader(cfg, dataset_name, mapper = mapper)
        # evaluator = get_evaluator(
        #     cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        # )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def build_train_loader(cfg):
    if cfg.INPUT.DATASET_MAPPER_NAME == "fewshot_sem":
        mapper = FewShotDatasetMapper_stage2(cfg, True)
        # print(build_detection_train_loader(cfg, mapper=mapper))
    elif cfg.INPUT.DATASET_MAPPER_NAME == "fewshot_sem_ori":
        mapper = FewShotDatasetMapper_stage1(cfg, True)
    else:
        mapper = None
    return build_detection_train_loader(cfg, mapper=mapper)
# return build_detection_train_loader(cfg, mapper=mapper)

def do_train(cfg, model, resume=False, data_loaders = None, evaluators = None):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    # data_loader = build_detection_train_loader(cfg)
    data_loader = build_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))

    best_mIoU = 0

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                # and iteration != max_iter - 1
            ):
                results = do_test(cfg, model, data_loaders, evaluators)

                results_dict = dict(results)['sem_seg']
                storage.put_scalars(**results_dict, smoothing_hint=False)
                comm.synchronize()
                
                if results:   
                    cur_mIoU = results['sem_seg']['mIoU'] if 'mIoU' in results['sem_seg'] else results['sem_seg']['IoU']
                    if cur_mIoU > best_mIoU:
                        best_mIoU = cur_mIoU
                        periodic_checkpointer.save('model_best')

                        results_dict = json.dumps(results_dict)
                        f = open(os.path.join(cfg.OUTPUT_DIR, 'best.json'), 'w')
                        f.write(results_dict)
                        f.close()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            # periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts) # ['SEED', k]
    # 
    cfg.merge_from_list(add_seed(cfg))
    cfg.merge_from_list(add_step2dir(cfg)) 
    cfg.merge_from_list(add_dataset(cfg)) 

    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)

    if cfg.MODEL.WEIGHTS_ is not None:
        saved_state_dict = torch.load(cfg.MODEL.WEIGHTS_)['model']
        new_params = model.state_dict()

        for i in saved_state_dict:
            if i in new_params.keys():
                # print('\t' + i)
                new_params[i] = saved_state_dict[i]

        model.load_state_dict(new_params)

    # build test set first
    data_loaders, evaluators = [], []
    for dataset_name in cfg.DATASETS.TEST:
        if cfg.INPUT.DATASET_MAPPER_NAME == "fewshot_sem":
            mapper = FewShotDatasetMapper_stage2(cfg, False)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "fewshot_sem_ori":
            mapper = FewShotDatasetMapper_stage1(cfg, False)
        else:
            mapper = None
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper = mapper)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        data_loaders.append(data_loader)
        evaluators.append(evaluator)

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model, data_loaders, evaluators)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
        )

    do_train(cfg, model, resume=args.resume, data_loaders=data_loaders, evaluators=evaluators)
    # return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )