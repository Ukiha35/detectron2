import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_test_loader
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from collections import OrderedDict
import cv2
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
    inference_on_dataset,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets.coco import load_coco_json
import pycocotools
from detectron2.data.datasets import register_coco_instances
import mrcfile
from detectron2.utils.img_utils import *

#声明类别，尽量保持
CLASS_NAMES = ["__background__","particle"]
# 数据集路径
DATASET_ROOT = "/media/ps/passport2/ltc/T20S/data_patch_1024_1024_bbox_240.0_overlap_0.5_iou_0.5_shrink_True/"
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')

TEST_PATH = os.path.join(DATASET_ROOT, 'images', 'raw_test')
TEST_JSON = os.path.join(ANN_ROOT, 'instances_raw_test.json')

# 声明数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "T20S_raw_test": (TEST_PATH, TEST_JSON),
}
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


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    
    args.config_file = "/home/ps/ltc/detectron2/configs/T20S-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    args.test_data_file = "/media/ps/passport2/ltc/T20S/data_patch_1024_1024_bbox_240.0_overlap_0.5_iou_0.5_shrink_True/images/raw_test/"
    args.eval_only = True
    # args.resume = False
    args.opts = ['DATASETS.TEST',("T20S_test",)]
    
    cfg = setup(args)
    register_dataset()
    predictor = DefaultPredictor(cfg)

    for file in os.listdir(args.test_data_file):
        with mrcfile.open(os.path.join(args.test_data_file, file)) as mrc:
            img = cv2.equalizeHist(quantize(mrc.data.copy()))
            outputs = predictor(img)
            v = Visualizer(img, metadata=MetadataCatalog.get("T20S_raw_test"), scale=1.5)

            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            img = v.get_image()[:,:,[2,1,0]]
            
            
            # model = Trainer.build_model(cfg)
            # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            #     cfg.MODEL.WEIGHTS, resume=args.resume
            # )
            # res = Trainer.test(cfg, model)
            # if cfg.TEST.AUG.ENABLED:
            #     res.update(Trainer.test_with_TTA(cfg, model))
            # if comm.is_main_process():
            #     verify_results(cfg, res)
            # return res