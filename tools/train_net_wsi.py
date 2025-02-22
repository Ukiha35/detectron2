#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict,  abc

import datetime
import time
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_wsi_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.data.datasets.coco import load_coco_json
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    DatasetEvaluator,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_wsi_dataset,
    verify_results,
    print_csv_format,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.visualizer import Visualizer
import cv2, mrcfile
import numpy as np
from detectron2.data.datasets.pascal_voc import register_pascal_voc



#声明类别，尽量保持
CLASS_NAMES = ["__background__","particle"]
# 数据集路径
DATASET_ROOT = '/media/ps/passport2/ltc/T20S/data_patch_1024_1024_bbox_240.0_overlap_0.5_iou_0.5_shrink_True/'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')

TRAIN_PATH = os.path.join(DATASET_ROOT, 'images', 'train')
VAL_PATH = os.path.join(DATASET_ROOT, 'images', 'test')
TEST_PATH = os.path.join(DATASET_ROOT, 'images', 'raw_test')

TRAIN_JSON = os.path.join(ANN_ROOT, 'instances_train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'instances_test.json')
TEST_JSON = os.path.join(ANN_ROOT, 'instances_raw_test.json')

# 声明数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "T20S_train": (TRAIN_PATH, TRAIN_JSON),
    "T20S_val": (VAL_PATH, VAL_JSON),
    "T20S_test": (TEST_PATH, TEST_JSON),
}


def mapping2jpg(img):
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

#===========以下有两种注册数据集的方法，本人直接用的第二个plain_register_dataset的方式 也可以用register_dataset的形式==================
#注册数据集（这一步就是将自定义数据集注册进Detectron2）
def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   json_file=json_file,
                                   image_root=image_root)


#注册数据集实例，加载数据集中的对象实例
def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")


# 查看数据集标注，可视化检查数据集标注是否正确，
#这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
#可选择使用此方法
def checkout_dataset_annotation(name="coco_my_val"):
    #dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
    dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH)
    print(len(dataset_dicts))
    for i, d in enumerate(dataset_dicts,0):
        #print(d)
        with mrcfile.open(d["file_name"]) as mrc:
            img = mrc.data
        # img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img, metadata=MetadataCatalog.get(name), scale=1.5)
        vis = visualizer.draw_dataset_dict(d)
        #cv2.imshow('show', vis.get_image()[:, :, ::-1])
        if not os.path.exists('out/'):
            os.makedirs('out/')
        cv2.imwrite('out/'+(os.path.basename(d['file_name'])).split('.')[0]+ '.jpg',vis.get_image()[:, :])
        #cv2.waitKey(0)
        if i == 200:
            break


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        # return PascalVOCDetectionEvaluator(dataset_name)
        return PascalVOCDetectionEvaluator(dataset_name, cfg.DATASETS.POST_PROCESS)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


class WSITrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_wsi_test_loader(cfg, dataset_name)
    
    @classmethod
    def test_wsi(cls, cfg, model, save_dir, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_wsi_dataset(model, data_loader, evaluator, save_dir)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        
        if 'wsi' in cfg.DATASETS.TEST[0]:
            os.system(f"python /home/ps/ltc/monuseg18/data/vis_wsi.py --pred_label_path {save_dir}")
        else:
            os.system(f"python /home/ps/ltc/monuseg18/data/vis.py --pred_label_path {save_dir}")

        if 'assign' in cfg.DATASETS.CLUSTER_PARAMETER.PRIOR:
            if 'wsi' in cfg.DATASETS.TEST[0]:
                os.system(f"python /home/ps/ltc/monuseg18/data/vis_canvas_wsi.py --json_path {cfg.DATASETS.CLUSTER_PARAMETER.PRIOR} --pred_label_path {save_dir}")
            else:
                os.system(f"python /home/ps/ltc/monuseg18/data/vis_canvas.py --json_path {cfg.DATASETS.CLUSTER_PARAMETER.PRIOR} --pred_label_path {save_dir}")
        return results
    

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):        
    cfg = setup(args)

    if 'RetinaNet' in cfg.MODEL.META_ARCHITECTURE:
        args.save_dir = os.path.join(cfg.SAVE_DIR, f"conf_thr_{cfg.MODEL.RETINANET.SCORE_THRESH_TEST}_SCALE_{cfg.DATASETS.CLUSTER_PARAMETER.SCALE}_step_{cfg.DATASETS.CLUSTER_PARAMETER.STEP}_axis_{cfg.DATASETS.POST_PROCESS.AXIS_THR}_area_{cfg.DATASETS.POST_PROCESS.AREA_THR}_patch_size_{cfg.DATASETS.CLUSTER_PARAMETER.PATCH_SIZE[0]}_{cfg.DATASETS.CLUSTER_PARAMETER.PATCH_SIZE[1]}_{cfg.MODEL.WEIGHTS.split('/')[-1].split('.pth')[0]}")
    else:
        args.save_dir = os.path.join(cfg.SAVE_DIR, f"conf_thr_{cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}_SCALE_{cfg.DATASETS.CLUSTER_PARAMETER.SCALE}_step_{cfg.DATASETS.CLUSTER_PARAMETER.STEP}_axis_{cfg.DATASETS.POST_PROCESS.AXIS_THR}_area_{cfg.DATASETS.POST_PROCESS.AREA_THR}_patch_size_{cfg.DATASETS.CLUSTER_PARAMETER.PATCH_SIZE[0]}_{cfg.DATASETS.CLUSTER_PARAMETER.PATCH_SIZE[1]}_{cfg.MODEL.WEIGHTS.split('/')[-1].split('.pth')[0]}")

    if cfg.DATASETS.CLUSTER_PARAMETER.PRIOR is not None and "assign" in cfg.DATASETS.CLUSTER_PARAMETER.PRIOR:
        additional_dir = cfg.DATASETS.CLUSTER_PARAMETER.PRIOR.split('/')[-2:]
        additional_dir[-1] = additional_dir[-1].split(".")[0]
        args.save_dir = os.path.join(args.save_dir, *additional_dir)

    


    # register_dataset()
    if args.eval_only:
        model = WSITrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = WSITrainer.test_wsi(cfg, model, args.save_dir)
        if cfg.TEST.AUG.ENABLED:
            res.update(WSITrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = WSITrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()

# 不使用tta的原因是使用tta来resize图像会把resize的时间算到compute time中，另外实现则可以算在data time中。
# 而由于patch大小不固定，导致resize时间不固定，故必须排除。
def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    args.GPU = '1'
    
    # args.config_file = "/home/ps/ltc/detectron2/configs/monuseg18_PascalVOC-Detection/faster_rcnn_X_101_FPN_lowres_test.yaml"
    # args.config_file = "/home/ps/ltc/detectron2/configs/monuseg18_PascalVOC-Detection/faster_rcnn_X_101_FPN_test.yaml"
    # args.config_file = "/home/ps/ltc/detectron2/configs/monuseg18_PascalVOC-Detection/faster_rcnn_X_101_FPN_lowres_test.yaml"
    # args.config_file = "/home/ps/ltc/detectron2/configs/monuseg18_PascalVOC-Detection/faster_rcnn_X_101_FPN_test_stage2.yaml"
    # args.config_file = "/home/ps/ltc/detectron2/configs/monuseg18_PascalVOC-Detection/faster_rcnn_X_101_FPN_test_wsi.yaml"
    # args.config_file = "/home/ps/ltc/detectron2/configs/monuseg18_PascalVOC-Detection/faster_rcnn_X_101_FPN_lowres_test_wsi.yaml"
    # args.config_file = "/home/ps/ltc/detectron2/configs/monuseg18_PascalVOC-Detection/retinanet_R_50_FPN_test.yaml"
    # args.config_file = "/home/ps/ltc/detectron2/configs/monuseg18_PascalVOC-Detection/retinanet_R_50_FPN_lowres_test.yaml"
    args.config_file = "/home/ps/ltc/detectron2/configs/monuseg18_PascalVOC-Detection/faster_rcnn_X_101_FPN_test_assign.yaml"
    # args.config_file = "/home/ps/ltc/detectron2/configs/monuseg18_PascalVOC-Detection/faster_rcnn_X_101_FPN_test_assign_wsi.yaml"

    args.eval_only = True


    print("Command Line Args:", args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

def register_all_pascal_voc(root):
    # dir = "/media/ps/passport1/ltc/DHB/VOC2007_pos_based_scale_20/"
    dir = "/media/ps/passport1/ltc/monuseg18/VOC2007_trainsize_1000/"
    
    SPLITS = [
        ("voc_2007_trainval", dir, "trainval"),
        ("voc_2007_train", dir, "train"),
        ("voc_2007_val", dir, "val"),
        ("voc_2007_test", dir, "test"),
        ("voc_2007_wsi", dir, "WSI"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


if __name__ == "__main__":
    register_all_pascal_voc(root="/media/ps/passport1/ltc/monuseg18/")
    invoke_main()  # pragma: no cover
