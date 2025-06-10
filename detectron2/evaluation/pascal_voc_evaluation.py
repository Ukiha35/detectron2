# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.structures import Boxes
import json
from torchvision.ops import nms
from .evaluator import DatasetEvaluator
from tqdm import tqdm
from detectron2.structures import Instances
import openslide


def bbox_iou(bbox_patch, bbox_roi):
    """
    计算两个bbox的IoU值
    bbox格式：[xmin, ymin, xmax, ymax]
    """
    x1, y1, x2, y2 = bbox_patch
    x1_t, y1_t, x2_t, y2_t = bbox_roi
    
    # 计算交集
    inter_x1 = max(x1, x1_t)
    inter_y1 = max(y1, y1_t)
    inter_x2 = min(x2, x2_t)
    inter_y2 = min(y2, y2_t)
    
    # 如果没有交集
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    patch_area = (x2 - x1) * (y2 - y1)
        
    # 返回IoU
    return inter_area / patch_area

def filter_bboxes_by_roi(splitlines, rois, imagenames, threshold=0.5):
    """
    根据给定的ROI和阈值过滤掉bbox中不满足条件的
    Args:
        splitlines: 包含图片名称和bbox的列表，每一行是一个列表，格式为[imagename, xmin, ymin, xmax, ymax, score]
        rois: 一个字典，键是图片名称，值是该图片对应的roi的bbox列表
        imagenames: 图片名称的列表，只有这些图片会被处理
        threshold: IoU阈值，只有IoU大于阈值的bbox才会被保留
    
    Returns:
        filtered_splitlines: 过滤后的splitlines
    """
    filtered_splitlines = []
    
    for imagename in imagenames:
        # 只处理指定的图片
        lines_for_imagename = [line for line in splitlines if line[0] == imagename]
        
        # 如果该图片有对应的roi
        if imagename in rois:
            rois_list = rois[imagename]
            
            for line in lines_for_imagename:
                bbox = [float(l) for l in line[2:]]  # 获取bbox：[xmin, ymin, xmax, ymax]
                score = float(line[1])  # 获取得分
                
                # 检查bbox与所有roi的IoU
                keep_bbox = False
                for roi in rois_list:
                    iou = bbox_iou(bbox, roi)
                    if iou >= threshold:
                        keep_bbox = True
                        break
                
                # 如果bbox与任一roi的IoU大于阈值，则保留该bbox
                if keep_bbox:
                    filtered_splitlines.append(line)
        else:
            # 如果没有roi，保留该bbox（根据需求可做调整）
            filtered_splitlines.extend(lines_for_imagename)
    
    return filtered_splitlines




def fast_nms(instances, thresh, mode):
    """
    Apply Non-Maximum Suppression (NMS) manually using PyTorch operations.
    
    Args:
        instances (Instances): an Instances object containing fields such as `pred_boxes`, `scores`, and `pred_classes`.
        thresh (float): NMS threshold, only detections with IoU < threshold will be kept.
        
    Returns:
        Instances: a new Instances object containing the detections after NMS.
    """
    if thresh < 0 or thresh > 1:
        return instances

    # Extract fields
    dets = instances.pred_boxes.tensor  # Tensor of shape (N, 4) -> (xmin, ymin, xmax, ymax)
    scores = instances.scores  # Tensor of shape (N, ) -> detection scores

    # Sort by scores in descending order
    order = scores.argsort(descending=True)
    dets = dets[order]
    scores = scores[order]

    keep = []  # Indices of boxes to keep

    while len(dets) > 0:
        # Always keep the box with the highest score
        keep.append(order[0].item())

        if len(dets) == 1:
            break  # Only one box left, stop

        # Compute IoU of the top box with the rest
        xx1 = torch.maximum(dets[0, 0], dets[1:, 0])
        yy1 = torch.maximum(dets[0, 1], dets[1:, 1])
        xx2 = torch.minimum(dets[0, 2], dets[1:, 2])
        yy2 = torch.minimum(dets[0, 3], dets[1:, 3])

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        intersection = w * h

        area1 = (dets[0, 2] - dets[0, 0]) * (dets[0, 3] - dets[0, 1])
        area2 = (dets[1:, 2] - dets[1:, 0]) * (dets[1:, 3] - dets[1:, 1])
        union = area1 + area2 - intersection

        if mode=='union':
            iou = intersection / union
        elif mode=='min':
            iou = intersection / torch.min(area1, area2)

        # Select boxes with IoU less than the threshold
        keep_idxs = (iou <= thresh).nonzero(as_tuple=True)[0]
        dets = dets[1:][keep_idxs]
        order = order[1:][keep_idxs]

    # Create a new Instances object with the kept detections
    new_instances = instances[torch.tensor(keep, dtype=torch.long)]
    return new_instances


def nms_on_patches(instances, image_size, thresh, mode='union', patch_size=[2500, 2500], overlap=0.5):
    """
    Apply NMS on overlapping patches of an image.
    
    Args:
        instances (Instances): an Instances object containing fields such as pred_boxes, scores, and pred_classes.
        image_size (tuple): The size of the original image (height, width).
        patch_size (tuple): The size of each patch (height, width).
        overlap_size (tuple): The size of the overlap region between patches (height, width).
        thresh (float): NMS threshold, only detections with IoU < threshold will be kept.
        
    Returns:
        Instances: a new Instances object containing the detections after NMS.
    """
    if thresh < 0 or thresh > 1:
        return instances
        
    # Image dimensions
    img_width, img_height = image_size
    patch_width, patch_height = patch_size
    overlap_height, overlap_width = [int(i*overlap) for i in patch_size]
    
    # Split the image into overlapping patches
    # print(f'performing nms on overlapping patches...')
    
    for y in range(0, img_height, patch_height - overlap_height):  # Step by (patch_height - overlap_height)
        for x in range(0, img_width, patch_width - overlap_width):  # Step by (patch_width - overlap_width)
            # Calculate patch boundaries
            patch_xmin = x
            patch_ymin = y
            patch_xmax = min(x + patch_width, img_width)
            patch_ymax = min(y + patch_height, img_height)
            
            # Select boxes inside this patch
            dets = instances.pred_boxes.tensor  # (N, 4) -> (xmin, ymin, xmax, ymax)
            patch_indices = ((dets[:, 0] >= patch_xmin) & (dets[:, 1] >= patch_ymin) & 
                             (dets[:, 0] <= patch_xmax) & (dets[:, 1 ] <= patch_ymax))

            # If there are any boxes in this patch, apply NMS
            if patch_indices.sum() > 0:
                # Create Instances for this patch
                patch_instances = instances[patch_indices]
                instances = instances[~patch_indices]
                # Apply NMS to the current patch
                patch_instances = fast_nms(patch_instances, thresh, mode)
                
                # Store the result
                instances = Instances.cat([patch_instances,instances])

    # Combine all patch results into one Instances object
    return instances

def py_cpu_nms(instances, thresh):
    """
    Apply Non-Maximum Suppression (NMS) to the detections in the `instances`.
    
    Args:
        instances (Instances): an Instances object containing fields such as `pred_boxes`, `scores`, and `pred_classes`.
        thresh (float): NMS threshold, only detections with IoU < threshold will be kept.
        
    Returns:
        Instances: a new Instances object containing the detections after NMS.
    """
    if thresh<0 or thresh>1:
        return instances
    
    # Extract fields from Instances object
    dets = instances.pred_boxes.tensor  # Tensor of shape (N, 4) -> (xmin, ymin, xmax, ymax)
    scores = instances.scores  # Tensor of shape (N, ) -> detection scores
    classes = instances.pred_classes  # Tensor of shape (N, ) -> predicted class labels (not used in NMS directly)
    
    # Convert to numpy arrays for easier manipulation
    dets = dets.cpu().numpy()
    scores = scores.cpu().numpy()
    classes = classes.cpu().numpy()  # If needed, you can use classes in filtering later
    
    # Calculate the areas of the bounding boxes
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    
    keep = []  # List to keep the indexes of the boxes that pass NMS
    index = scores.argsort()[::-1]  # Sort by score in descending order
    
    while index.size > 0:
        i = index[0]  # Select the box with the highest score
        keep.append(i)
        
        # Calculate the overlap of the current box with the others
        x11 = np.maximum(x1[i], x1[index[1:]])  # max of xmin
        y11 = np.maximum(y1[i], y1[index[1:]])  # max of ymin
        x22 = np.minimum(x2[i], x2[index[1:]])  # min of xmax
        y22 = np.minimum(y2[i], y2[index[1:]])  # min of ymax
        
        # Compute the width and height of the overlapping area
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        
        # Compute the intersection area
        overlaps = w * h
        
        # Compute the IoU (Intersection over Union)
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        
        # Keep only the boxes that have IoU <= threshold
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # Update the index by excluding the boxes that have high IoU with the current box
    
    # Now we have the `keep` list, which contains the indices of the detections that pass NMS
    keep_dets = dets[keep]
    keep_scores = scores[keep]
    keep_classes = classes[keep]
    
    # Create a new Instances object with the NMS results
    new_instances = instances.__class__(instances.image_size)
    new_instances.set('pred_boxes', Boxes(torch.tensor(keep_dets)))
    new_instances.set('scores', torch.tensor(keep_scores))
    new_instances.set('pred_classes', torch.tensor(keep_classes))
    
    return new_instances



class PascalVOCDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC style AP for Pascal VOC dataset.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def __init__(self, dataset_name, post_process):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)

        # Too many tiny files, download all to local for speed.
        annotation_dir_local = PathManager.get_local_path(
            os.path.join(meta.dirname, "Annotations/")
        )
        self._anno_file_template = os.path.join(annotation_dir_local, "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.final_nms_thr = post_process.FINAL_NMS_THR
        self.axis_thr = post_process.AXIS_THR
        self.area_thr = post_process.AREA_THR
        self.final_nms_mode = post_process.FINAL_NMS_MODE

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):        
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = nms_on_patches(output["instances"],output['instances'].image_size, thresh=self.final_nms_thr, mode=self.final_nms_mode)
            # instances = fast_nms(output["instances"], final_nms_thr)
            # instances = py_cpu_nms(output["instances"].to(self._cpu_device),final_nms_thr)
            boxes = instances.pred_boxes.tensor.to(self._cpu_device).numpy()
            scores = instances.scores.to(self._cpu_device).tolist()
            classes = instances.pred_classes.to(self._cpu_device).tolist()
            boxes[:,[0,2]] = np.clip(boxes[:,[0,2]], 0, input['width'])
            boxes[:,[1,3]] = np.clip(boxes[:,[1,3]], 0, input['height'])
                            
            # image = openslide.open_slide(input["file_name"])
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                
                width = xmax - xmin
                height = ymax - ymin
                area = width * height
                
                if width >= self.axis_thr and height >= self.axis_thr and area >= self.area_thr:
                    self._predictions[cls].append(
                        f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                    )
                # mean_value = np.array(image.read_region((int(box[0]),int(box[1])),0,(int(box[2]-box[0]+1),int(box[3]-box[1]+1))))[:,:,:-1].mean()
                # if width >= self.axis_thr and height >= self.axis_thr and area >= self.area_thr and mean_value <= 200:
                #     self._predictions[cls].append(
                #         f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                #     )
                
            
    
    def save(self, output_dir: str):
        """
        Save the detection results for each image to a separate JSON file.

        Args:
            output_dir (str): Directory to save the JSON files.
        """        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        with PathManager.open(self._image_set_path, "r") as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        for imagename in imagenames:
            prediction_data = []
            if os.path.exists(os.path.join(output_dir, f"{imagename}.json")):
                continue
            
            # Iterate through the predictions for each class
            for class_id, predictions in self._predictions.items():
                # Iterate through the predictions for each image
                splitlines = [x.strip().split(" ") for x in predictions] 
                for pred in [p for p in splitlines if p[0]==imagename]:
                    # Prepare the prediction data to be saved for this image
                    prediction_data.append(
                            {
                                "class_id": class_id,
                                "class": 'pos',
                                "score": float(pred[1]),
                                "x": float(pred[2]),
                                "y": float(pred[3]),
                                "w": round((float(pred[4]) - float(pred[2])), 3),
                                "h": round((float(pred[5]) - float(pred[3])), 3),
                            }
                        )

            # Define the output file path for this image
            output_file = os.path.join(output_dir, f"{imagename}.json")

            # Save the new file
            if len(prediction_data)>0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(prediction_data, f, indent=4)
                print(f"Saved detection results to {output_file}.")
                    
        
    def load(self, output_dir: str):
        """
        Load the detection results from JSON files in the specified directory and classify by class_id.

        Args:
            output_dir (str): Directory containing the JSON files.
        """
        # Initialize a dictionary to store predictions by class_id
        loaded_data = defaultdict(list)

        # Get all JSON files in the output directory
        for json_file in os.listdir(output_dir):
            if json_file.endswith(".json"):
                json_path = os.path.join(output_dir, json_file)

                with open(json_path, 'r', encoding='utf-8') as f:
                    # Load the data from the JSON file
                    prediction_data = json.load(f)

                    imagename = json_file.split('.json')[0]

                    # Classify detections by class_id
                    for detection in prediction_data:
                        class_id = detection["class_id"]
                        score = detection["score"]
                        bbox = [detection[p] for p in ['x','y','w','h']]
                        bbox[2] += bbox[0]
                        bbox[3] += bbox[1]
                        # Store the detection by class_id
                        loaded_data[class_id].append(f"{imagename} {score:.3f} {bbox[0]:.1f} {bbox[1]:.1f} {bbox[2]:.1f} {bbox[3]:.1f}")
        
        self._predictions = loaded_data
        print(f"Loaded detection results from {json_path}.")

        return loaded_data

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        if "wsi" in self._dataset_name:
            ret = OrderedDict()
            ret["bbox"] = {"AP": 0, "AP50": 0, "AP75": 0, "REC75": 0, "PREC75":0, "REC50": 0, "PREC50":0, "REC30": 0, "PREC30":0}
            self._ap = ret
            return ret
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            recs = defaultdict(list)  # iou -> ap per class
            precs = defaultdict(list)  # iou -> ap per class
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)
                    recs[thresh].append(rec * 100)
                    precs[thresh].append(prec * 100)

            rec30, prec30, ap30 = voc_eval(
                res_file_template,
                self._anno_file_template,
                self._image_set_path,
                cls_name,
                ovthresh=30.0 / 100.0,
                use_07_metric=self._is_2007,
            )

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75], "REC75": recs[75][0][-1], "PREC75":precs[75][0][-1], "REC50": recs[50][0][-1], "PREC50":precs[50][0][-1], "REC30": rec30[-1]*100, "PREC30":prec30[-1]*100}
        self._ap = ret
        return ret


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    with PathManager.open(filename) as f:
        tree = ET.parse(f)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(float(bbox.find("xmin").text)),
            int(float(bbox.find("ymin").text)),
            int(float(bbox.find("xmax").text)),
            int(float(bbox.find("ymax").text)),
        ]
        objects.append(obj_struct)

    rois = []
    for roi in tree.findall(".roi/bndbox"):
        bbox = [float(roi.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
        rois.append(bbox)

    return objects, rois


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with PathManager.open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    rois = {}
    for imagename in imagenames:
        recs[imagename], rois[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(bool)
        # difficult = np.array([False for x in R]).astype(bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    
    splitlines = filter_bboxes_by_roi(splitlines,rois,imagenames,0.5)
    
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
