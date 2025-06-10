# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import numpy as np
import operator
import pickle
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import torch.utils.data as torchdata
from tabulate import tabulate
from termcolor import colored

from detectron2.config import configurable
from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import _log_api_usage, log_first_n
from detectron2.utils.img_utils import filter_bboxes_in_patch

from .catalog import DatasetCatalog, MetadataCatalog
from .common import AspectRatioGroupedDataset, DatasetFromList, MapDataset, ToIterableDataset
from .dataset_mapper import DatasetMapper
from . import detection_utils as utils
from .samplers import (
    InferenceSampler,
    RandomSubsetTrainingSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from fvcore.common.config import CfgNode
from sklearn.cluster import DBSCAN
import cv2
import os
import json
import openslide
import h5py
import random
from torchvision.ops import nms
from detectron2.data.transforms import ResizeShortestEdge, AugInput
import numpy as np
from scipy import ndimage as nd
from PIL import Image
"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "build_batch_data_loader",
    "build_detection_train_loader",
    "build_detection_test_loader",
    "build_detection_wsi_test_loader",
    "get_detection_dataset_dicts",
    "load_proposals_into_dataset",
    "print_instances_class_histogram",
    "fast_nmm",
    "fast_nms",
    "expand_bboxes",
    "clip_bboxes",
    "adjust_bboxes",
    "cluster_bboxes_with_dbscan",
    "draw_bboxes",
    
]


def compute_gaussian(tile_size: Union[Tuple[int, ...], List[int]], sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1 ):
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = nd.gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = gaussian_importance_map / (gaussian_importance_map).max() * value_scaling_factor

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = (gaussian_importance_map[gaussian_importance_map != 0]).min()

    return gaussian_importance_map




def max_pooling(input_array, pool_size=(2, 2), stride=(2,2)):
    # 获取输入的形状
    input_height, input_width = input_array.shape
    pool_height, pool_width = pool_size
    output_height = (input_height - pool_height) // stride[0] + 1
    output_width = (input_width - pool_width) // stride[1] + 1
    
    # 创建输出数组
    output_array = np.zeros((output_height, output_width))
    
    # 执行 max pooling 操作
    for i in range(output_height):
        for j in range(output_width):
            # 获取当前池化窗口
            window = input_array[i * stride[0] : i * stride[0] + pool_height, j * stride[1] : j * stride[1] + pool_width]
            # 选择窗口中的最大值
            output_array[i, j] = np.max(window)
    
    return output_array

def batch_nmm_by_size(bbox_list, iou_threshold, batch_size=1, max_size=400*400):
    """
    Apply Non-Maximum Merging (NMM) for bounding boxes.

    Args:
        bbox_list (list): A list of bounding boxes with scores. Each element is [xmin, ymin, xmax, ymax, score].
        iou_threshold (float): IoU threshold for merging boxes.

    Returns:
        torch.Tensor: The merged bounding boxes with their scores (shape: M, 5).
        List[dict]: Detailed information of merged clusters, including each retained box
                    and its associated child boxes.
    """
    # Convert input list to a PyTorch tensor
    bbox_list = torch.tensor(bbox_list, dtype=torch.float32)
    bbox_list[:, 2] += bbox_list[:, 0]  # Convert width to xmax
    bbox_list[:, 3] += bbox_list[:, 1]  # Convert height to ymax
    
    if len(bbox_list) == 0:
        return torch.empty((0, 5)), []

    # Sort boxes by scores in descending order
    _, idxs = ((bbox_list[:, 2] - bbox_list[:, 0]) * (bbox_list[:, 3] - bbox_list[:, 1])).sort(descending=False)
    bbox_list = bbox_list[idxs]
    x1 = bbox_list[:, 0]
    y1 = bbox_list[:, 1]
    x2 = bbox_list[:, 2]
    y2 = bbox_list[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Initialize output structures
    merged_boxes = []
    clusters = []
    
    while len(bbox_list) > 0:
        # Take the box with the highest score as the "retained box"
        retained_box = bbox_list[0]

        # Compute IoU with the rest of the boxes
        xx1 = torch.maximum(retained_box[0], x1)
        yy1 = torch.maximum(retained_box[1], y1)
        xx2 = torch.minimum(retained_box[2], x2)
        yy2 = torch.minimum(retained_box[3], y2)

        w = torch.maximum(torch.tensor(0.0), xx2 - xx1)
        h = torch.maximum(torch.tensor(0.0), yy2 - yy1)
        intersection = w * h
        
        # iou = intersection / (areas + areas[0] - intersection)
        iou = intersection / torch.minimum(areas, areas[0])
        
        # Find boxes to merge based on IoU threshold
        all_merge_idxs = (iou > iou_threshold).nonzero(as_tuple=True)[0]
        merge_idxs = torch.tensor([0])
        for idx in range(0,len(all_merge_idxs)-1,batch_size):
            tmp_merge_idxs = torch.hstack((merge_idxs, all_merge_idxs[(idx+1):(idx+1+batch_size)])).to(torch.int)
            # Compute the union box for the retained box and merged boxes
            merged_x1 = torch.min(x1[tmp_merge_idxs])
            merged_y1 = torch.min(y1[tmp_merge_idxs])
            merged_x2 = torch.max(x2[tmp_merge_idxs])
            merged_y2 = torch.max(y2[tmp_merge_idxs])
            if ((merged_x2-merged_x1) <= max_size[0]) and ((merged_y2-merged_y1) <= max_size[1]):
                merge_idxs = tmp_merge_idxs
            else:
                continue
            
        # Update the merged box
        merged_x1 = torch.min(x1[merge_idxs])
        merged_y1 = torch.min(y1[merge_idxs])
        merged_x2 = torch.max(x2[merge_idxs])
        merged_y2 = torch.max(y2[merge_idxs])
        merged_box = torch.tensor([merged_x1, merged_y1, merged_x2-merged_x1, merged_y2-merged_y1, retained_box[4]])
            
        merged_boxes.append(merged_box)

        # Store cluster information
        tmp_bbox_list = bbox_list[merge_idxs]
        tmp_bbox_list[:,2:4] = tmp_bbox_list[:,2:4] - tmp_bbox_list[:,0:2]

        cluster_info = {
            "cluster": [merged_x1.item(), merged_y1.item(), merged_x2.item()-merged_x1.item(), merged_y2.item()-merged_y1.item(), retained_box[4].item()],
            "children": tmp_bbox_list.tolist()
        }
        clusters.append(cluster_info)
        
        # Remove merged boxes from the list
        bbox_list = bbox_list[[i for i in list(range(len(bbox_list))) if i not in merge_idxs]]
        x1 = bbox_list[:, 0]
        y1 = bbox_list[:, 1]
        x2 = bbox_list[:, 2]
        y2 = bbox_list[:, 3]
        areas = (x2 - x1) * (y2 - y1)

    # Convert merged boxes to a tensor
    merged_boxes = (torch.stack(merged_boxes) if len(merged_boxes) > 0 else torch.empty((0, 5))).numpy()
    return merged_boxes, clusters

def resize_patch_to_target_size(patch, target_size):
    """
    根据目标输出大小动态调整图像大小，按输入大小的倍数进行处理。

    Args:
        patch (np.ndarray): 输入图像，格式为 (H, W, C)。
        target_size (tuple): 目标输出大小 (target_height, target_width)。

    Returns:
        np.ndarray: 缩放后的图像，大小为目标输出大小。
    """
    # 获取输入图像的原始大小
    original_height, original_width = patch.shape[:2]

    # 获取目标大小
    target_width, target_height = target_size

    # 计算目标与输入大小的缩放比例（取短边为基准）
    height_ratio = target_height / original_height
    width_ratio = target_width / original_width

    # 使用最小缩放倍数，以确保不会超出目标范围
    scale_factor = min(height_ratio, width_ratio)

    # 根据缩放倍数计算 ResizeShortestEdge 的参数
    new_min_size = round(min(original_width, original_height) * scale_factor)
    new_max_size = round(max(original_width, original_height) * scale_factor)

    # 确保 new_min_size 和 new_max_size 不小于 1
    new_min_size = max(1, new_min_size)
    new_max_size = max(1, new_max_size)

    # 创建 ResizeShortestEdge 工具
    resize_tool = ResizeShortestEdge([new_min_size, new_min_size], new_max_size)

    # 使用 AugInput 进行图像变换
    aug_input = AugInput(patch)
    resize_tool(aug_input)

    # 获取缩放后的图像
    resized_patch = aug_input.image

    # 如果目标大小与缩放后的大小完全一致，则直接返回
    if resized_patch.shape[:2] == (target_height, target_width):
        return resized_patch

    # 如果缩放结果仍有细微差异，用 Padding 或 Cropping 处理到目标大小
    # 这里采用 Padding 方式，将图像填充到目标大小
    top_padding = (target_height - resized_patch.shape[0]) // 2
    bottom_padding = target_height - resized_patch.shape[0] - top_padding
    left_padding = (target_width - resized_patch.shape[1]) // 2
    right_padding = target_width - resized_patch.shape[1] - left_padding

    padded_patch = np.pad(resized_patch,
                          ((top_padding, bottom_padding), (left_padding, right_padding), (0, 0)),
                          mode='constant', constant_values=255)

    return padded_patch

def pil_resize(img, target_size):
    if len(img.shape) > 2 and img.shape[2] == 1:
        pil_image = Image.fromarray(img[:, :, 0], mode="L")
    else:
        pil_image = Image.fromarray(img)
    pil_image = pil_image.resize(list(target_size), 2)
    ret = np.asarray(pil_image)
    if len(img.shape) > 2 and img.shape[2] == 1:
        ret = np.expand_dims(ret, -1)
    return ret

def fast_nmm(bbox_list, iou_threshold):
    """
    Apply Non-Maximum Merging (NMM) for bounding boxes.

    Args:
        bbox_list (list): A list of bounding boxes with scores. Each element is [xmin, ymin, xmax, ymax, score].
        iou_threshold (float): IoU threshold for merging boxes.

    Returns:
        torch.Tensor: The merged bounding boxes with their scores (shape: M, 5).
        List[dict]: Detailed information of merged clusters, including each retained box
                    and its associated child boxes.
    """
    # Convert input list to a PyTorch tensor
    bbox_list = torch.tensor(bbox_list, dtype=torch.float32)
    bbox_list[:, 2] += bbox_list[:, 0]  # Convert width to xmax
    bbox_list[:, 3] += bbox_list[:, 1]  # Convert height to ymax
    
    if len(bbox_list) == 0:
        return torch.empty((0, 5)), []

    # Sort boxes by scores in descending order
    scores = bbox_list[:, 4]
    _, idxs = scores.sort(descending=True)
    bbox_list = bbox_list[idxs]

    # Extract box coordinates and areas
    x1 = bbox_list[:, 0]
    y1 = bbox_list[:, 1]
    x2 = bbox_list[:, 2]
    y2 = bbox_list[:, 3]
    scores = bbox_list[:, 4]
    areas = (x2 - x1) * (y2 - y1)

    # Initialize output structures
    merged_boxes = []
    clusters = []
    
    while len(bbox_list) > 0:
        # Take the box with the highest score as the "retained box"
        retained_box = bbox_list[0]

        # Compute IoU with the rest of the boxes
        xx1 = torch.maximum(retained_box[0], x1)
        yy1 = torch.maximum(retained_box[1], y1)
        xx2 = torch.minimum(retained_box[2], x2)
        yy2 = torch.minimum(retained_box[3], y2)

        w = torch.maximum(torch.tensor(0.0), xx2 - xx1)
        h = torch.maximum(torch.tensor(0.0), yy2 - yy1)
        intersection = w * h
        
        # iou = intersection / (areas + areas[0] - intersection)
        iou = intersection / torch.minimum(areas, areas[0])
        
        # Find boxes to merge based on IoU threshold
        merge_idxs = (iou > iou_threshold).nonzero(as_tuple=True)[0]

        # Compute the union box for the retained box and merged boxes
        merged_x1 = torch.min(x1[merge_idxs])
        merged_y1 = torch.min(y1[merge_idxs])
        merged_x2 = torch.max(x2[merge_idxs])
        merged_y2 = torch.max(y2[merge_idxs])

        # Update the merged box
        merged_box = torch.tensor([merged_x1, merged_y1, merged_x2-merged_x1, merged_y2-merged_y1, retained_box[4]])
        merged_boxes.append(merged_box)

        # Store cluster information
        tmp_bbox_list = bbox_list[merge_idxs]
        tmp_bbox_list[:,2:4] = tmp_bbox_list[:,2:4] - tmp_bbox_list[:,0:2]
        cluster_info = {
            "cluster": [merged_x1.item(), merged_y1.item(), merged_x2.item()-merged_x1.item(), merged_y2.item()-merged_y1.item()],
            "children": tmp_bbox_list.tolist()
        }
        clusters.append(cluster_info)

        # Remove merged boxes from the list
        bbox_list = bbox_list[(iou <= iou_threshold)]
        x1 = bbox_list[:, 0]
        y1 = bbox_list[:, 1]
        x2 = bbox_list[:, 2]
        y2 = bbox_list[:, 3]
        areas = (x2 - x1) * (y2 - y1)

    # Convert merged boxes to a tensor
    merged_boxes = (torch.stack(merged_boxes) if len(merged_boxes) > 0 else torch.empty((0, 5))).numpy()
    return merged_boxes, clusters

def fast_nms(bbox_list, thresh):
    """
    Apply Non-Maximum Suppression (NMS) using PyTorch's GPU-accelerated implementation.
    
    Args:
        instances (Instances): an Instances object containing fields such as `pred_boxes`, `scores`, and `pred_classes`.
        thresh (float): NMS threshold, only detections with IoU < threshold will be kept.
        
    Returns:
        Instances: a new Instances object containing the detections after NMS.
    """
    if thresh < 0 or thresh > 1:
        return bbox_list
    
    # Extract fields
    dets = bbox_list[:,:4]  # Tensor of shape (N, 4) -> (xmin, ymin, xmax, ymax)
    scores = bbox_list[:,4]  # Tensor of shape (N, ) -> detection scores
    
    # Use PyTorch's built-in NMS function
    keep = nms(torch.tensor(dets), torch.tensor(scores), thresh)  # Returns the indices of boxes to keep
    
    # Filter instances to keep only the selected detections
    new_bbox_list = bbox_list[keep]
    
    return new_bbox_list

def expand_bboxes(coordinates,expand=5, size_limit=None):
    coordinates = np.array([[bbox[0] - expand,  bbox[1] - expand, 
                        (bbox[2]+2*expand), (bbox[3]+2*expand), bbox[4]] for bbox in coordinates])
    if size_limit is None:
        return coordinates
    else:
        return clip_bboxes(coordinates,size_limit=size_limit)

def clip_bboxes(coordinates, size_limit=[0,0,1000,1000]):
    coordinates[:, 2:4] = coordinates[:, 2:4] + coordinates[:, 0:2]
    
    coordinates[:, 0:2] = np.clip(coordinates[:, 0:2], [size_limit[0], size_limit[1]], [size_limit[0]+size_limit[2], size_limit[1]+size_limit[3]])
    coordinates[:, 2:4] = np.clip(coordinates[:, 2:4], [size_limit[0], size_limit[1]], [size_limit[0]+size_limit[2], size_limit[1]+size_limit[3]])
    
    coordinates[:, 2:4] = coordinates[:, 2:4] - coordinates[:, 0:2]
    
    return coordinates[np.bitwise_and(coordinates[:,3]>0, coordinates[:,2]>0)]

def adjust_patches(coordinates, max_size=(500,500), min_size=(100,100), target_size=(300,300),size_limit=(0,0,1000,1000)):
        """
        Adjust patch sizes by splitting large patches, expanding small patches, 
        and resizing moderate-sized patches to a target size.

        Args:
            coordinates (np.array): Array of patch coordinates (x, y, w, h).
            max_size (tuple): Maximum width and height (w, h) of a patch.
            min_size (tuple): Minimum width and height (w, h) of a patch.
            target_size (tuple): Target width and height (w, h) to resize patches to.

        Returns:
            np.array: Adjusted patch coordinates.
        """
        final_patches = []
        for patch in coordinates:
            # 1. Expand small patches
            x, y, w, h, score = patch
            if w < min_size[0] or h < min_size[1]:
                new_w = max(w, min_size[0])
                new_h = max(h, min_size[1])
                new_x = x - (new_w - w) / 2
                new_y = y - (new_h - h) / 2
                expanded_patch = clip_bboxes(np.array([[new_x, new_y, new_w, new_h, score]]),size_limit=size_limit)[0]
            else:
                expanded_patch = [x, y, w, h, score]
                
            # 2. Split large patches
            x, y, w, h, score = expanded_patch
            if w > max_size[0] or h > max_size[1]:
                num_x = int(np.ceil(w / target_size[0]))
                num_y = int(np.ceil(h / target_size[1]))
                sub_w = w / num_x
                sub_h = h / num_y
                for i in range(num_x):
                    for j in range(num_y):
                        sub_x = x + i * sub_w
                        sub_y = y + j * sub_h
                        final_patches.append([sub_x, sub_y, sub_w, sub_h, score]) 
            else:
                final_patches.append([x, y, w, h, score])
            

        return np.array(final_patches)

def cluster_bboxes_with_dbscan(coordinates, distance_threshold=50, min_samples=1):
    """
    Clusters bounding boxes using DBSCAN based on spatial proximity.
    
    Args:
        coordinates (np.array): Array of shape (N, 4), where each row is [x, y, w, h].
        distance_threshold (float): Maximum distance between centers of boxes to form a cluster.
    
    Returns:
        parent_boxes (np.array): Array of parent bounding boxes (x, y, w, h) after clustering.
    """
    # Step 1: 计算边界框的中心
    centers = np.array([
        [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]  # Center (x, y)
        for bbox in coordinates
    ])
    
    # Step 2: 使用 DBSCAN 对中心点进行聚类
    clustering = DBSCAN(eps=distance_threshold, min_samples=min_samples, metric='euclidean').fit(centers)
    labels = clustering.labels_  # 每个框的簇标签
    
    # Step 3: 根据聚类结果计算父类框
    parent_boxes = []
    for label in np.unique(labels):
        cluster_indices = np.where(labels == label)[0]
        cluster_boxes = coordinates[cluster_indices]
        
        # 合并当前簇中的所有框，计算父类框的边界
        x_min = np.min(cluster_boxes[:, 0])
        y_min = np.min(cluster_boxes[:, 1])
        x_max = np.max(cluster_boxes[:, 0] + cluster_boxes[:, 2])
        y_max = np.max(cluster_boxes[:, 1] + cluster_boxes[:, 3])
        score = np.mean(cluster_boxes[:,4])
        
        # 父类框：[x, y, w, h]
        parent_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min, score])
    
    return np.array(parent_boxes)




def filter_images_with_only_crowd_annotations(dataset_dicts):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if ann.get("iscrowd", 0) == 0:
                return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def filter_images_with_few_keypoints(dataset_dicts, min_keypoints_per_image):
    """
    Filter out images with too few number of keypoints.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format as dataset_dicts, but filtered.
    """
    num_before = len(dataset_dicts)

    def visible_keypoints_in_image(dic):
        # Each keypoints field has the format [x1, y1, v1, ...], where v is visibility
        annotations = dic["annotations"]
        return sum(
            (np.array(ann["keypoints"][2::3]) > 0).sum()
            for ann in annotations
            if "keypoints" in ann
        )

    dataset_dicts = [
        x for x in dataset_dicts if visible_keypoints_in_image(x) >= min_keypoints_per_image
    ]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with fewer than {} keypoints.".format(
            num_before - num_after, min_keypoints_per_image
        )
    )
    return dataset_dicts


def load_proposals_into_dataset(dataset_dicts, proposal_file):
    """
    Load precomputed object proposals into the dataset.

    The proposal file should be a pickled dict with the following keys:

    - "ids": list[int] or list[str], the image ids
    - "boxes": list[np.ndarray], each is an Nx4 array of boxes corresponding to the image id
    - "objectness_logits": list[np.ndarray], each is an N sized array of objectness scores
      corresponding to the boxes.
    - "bbox_mode": the BoxMode of the boxes array. Defaults to ``BoxMode.XYXY_ABS``.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
        proposal_file (str): file path of pre-computed proposals, in pkl format.

    Returns:
        list[dict]: the same format as dataset_dicts, but added proposal field.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading proposals from: {}".format(proposal_file))

    with PathManager.open(proposal_file, "rb") as f:
        proposals = pickle.load(f, encoding="latin1")

    # Rename the key names in D1 proposal files
    rename_keys = {"indexes": "ids", "scores": "objectness_logits"}
    for key in rename_keys:
        if key in proposals:
            proposals[rename_keys[key]] = proposals.pop(key)

    # Fetch the indexes of all proposals that are in the dataset
    # Convert image_id to str since they could be int.
    img_ids = set({str(record["image_id"]) for record in dataset_dicts})
    id_to_index = {str(id): i for i, id in enumerate(proposals["ids"]) if str(id) in img_ids}

    # Assuming default bbox_mode of precomputed proposals are 'XYXY_ABS'
    bbox_mode = BoxMode(proposals["bbox_mode"]) if "bbox_mode" in proposals else BoxMode.XYXY_ABS

    for record in dataset_dicts:
        # Get the index of the proposal
        i = id_to_index[str(record["image_id"])]

        boxes = proposals["boxes"][i]
        objectness_logits = proposals["objectness_logits"][i]
        # Sort the proposals in descending order of the scores
        inds = objectness_logits.argsort()[::-1]
        record["proposal_boxes"] = boxes[inds]
        record["proposal_objectness_logits"] = objectness_logits[inds]
        record["proposal_bbox_mode"] = bbox_mode

    return dataset_dicts


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = np.asarray(
            [x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=int
        )
        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            assert (
                classes.max() < num_classes
            ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan"),
        key="message",
    )


def get_detection_dataset_dicts(
    names,
    filter_empty=True,
    min_keypoints=0,
    proposal_files=None,
    check_consistency=True,
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.
        check_consistency (bool): whether to check if datasets have consistent metadata.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names

    available_datasets = DatasetCatalog.keys()
    names_set = set(names)
    if not names_set.issubset(available_datasets):
        logger = logging.getLogger(__name__)
        logger.warning(
            "The following dataset names are not registered in the DatasetCatalog: "
            f"{names_set - available_datasets}. "
            f"Available datasets are {available_datasets}"
        )

    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]

    if isinstance(dataset_dicts[0], torchdata.Dataset):
        if len(dataset_dicts) > 1:
            # ConcatDataset does not work for iterable style dataset.
            # We could support concat for iterable as well, but it's often
            # not a good idea to concat iterables anyway.
            return torchdata.ConcatDataset(dataset_dicts)
        return dataset_dicts[0]

    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    if check_consistency and has_instances:
        try:
            class_names = MetadataCatalog.get(names[0]).thing_classes
            utils.check_metadata_consistency("thing_classes", names)
            print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts


def build_batch_data_loader(
    dataset,
    sampler,
    total_batch_size,
    *,
    aspect_ratio_grouping=False,
    num_workers=0,
    collate_fn=None,
    drop_last: bool = False,
    single_gpu_batch_size=None,
    prefetch_factor=2,
    persistent_workers=False,
    pin_memory=False,
    seed=None,
    **kwargs,
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces indices.
            Must be provided iff. ``dataset`` is a map-style dataset.
        total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
            :func:`build_detection_train_loader`.
        single_gpu_batch_size: You can specify either `single_gpu_batch_size` or `total_batch_size`.
            `single_gpu_batch_size` specifies the batch size that will be used for each gpu/process.
            `total_batch_size` allows you to specify the total aggregate batch size across gpus.
            It is an error to supply a value for both.
        drop_last (bool): if ``True``, the dataloader will drop incomplete batches.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    if single_gpu_batch_size:
        if total_batch_size:
            raise ValueError(
                """total_batch_size and single_gpu_batch_size are mutually incompatible.
                Please specify only one. """
            )
        batch_size = single_gpu_batch_size
    else:
        world_size = get_world_size()
        assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
        ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
            total_batch_size, world_size
        )
        batch_size = total_batch_size // world_size
    logger = logging.getLogger(__name__)
    logger.info("Making batched data loader with batch_size=%d", batch_size)

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        dataset = ToIterableDataset(dataset, sampler, shard_chunk_size=batch_size)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    if aspect_ratio_grouping:
        # assert drop_last, "Aspect ratio grouping will drop incomplete batches."
        assert not drop_last, "Aspect ratio grouping will not drop any incomplete batches."
        data_loader = torchdata.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            generator=generator,
            **kwargs,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, batch_size)
        if collate_fn is None:
            return data_loader
        return MapDataset(data_loader, collate_fn)
    else:
        return torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
            worker_init_fn=worker_init_reset_seed,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            generator=generator,
            **kwargs,
        )


def _get_train_datasets_repeat_factors(cfg) -> Dict[str, float]:
    repeat_factors = cfg.DATASETS.TRAIN_REPEAT_FACTOR
    assert all(len(tup) == 2 for tup in repeat_factors)
    name_to_weight = defaultdict(lambda: 1, dict(repeat_factors))
    # The sampling weights map should only contain datasets in train config
    unrecognized = set(name_to_weight.keys()) - set(cfg.DATASETS.TRAIN)
    assert not unrecognized, f"unrecognized datasets: {unrecognized}"
    logger = logging.getLogger(__name__)
    logger.info(f"Found repeat factors: {list(name_to_weight.items())}")

    # pyre-fixme[7]: Expected `Dict[str, float]` but got `DefaultDict[typing.Any, int]`.
    return name_to_weight


def _build_weighted_sampler(cfg, enable_category_balance=False):
    dataset_repeat_factors = _get_train_datasets_repeat_factors(cfg)
    # OrderedDict to guarantee order of values() consistent with repeat factors
    dataset_name_to_dicts = OrderedDict(
        {
            name: get_detection_dataset_dicts(
                [name],
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=(
                    cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                    if cfg.MODEL.KEYPOINT_ON
                    else 0
                ),
                proposal_files=(
                    cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None
                ),
            )
            for name in cfg.DATASETS.TRAIN
        }
    )
    # Repeat factor for every sample in the dataset
    repeat_factors = [
        [dataset_repeat_factors[dsname]] * len(dataset_name_to_dicts[dsname])
        for dsname in cfg.DATASETS.TRAIN
    ]

    repeat_factors = list(itertools.chain.from_iterable(repeat_factors))

    repeat_factors = torch.tensor(repeat_factors)
    logger = logging.getLogger(__name__)
    if enable_category_balance:
        """
        1. Calculate repeat factors using category frequency for each dataset and then merge them.
        2. Element wise dot producting the dataset frequency repeat factors with
            the category frequency repeat factors gives the final repeat factors.
        """
        category_repeat_factors = [
            RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset_dict, cfg.DATALOADER.REPEAT_THRESHOLD, sqrt=cfg.DATALOADER.REPEAT_SQRT
            )
            for dataset_dict in dataset_name_to_dicts.values()
        ]
        # flatten the category repeat factors from all datasets
        category_repeat_factors = list(itertools.chain.from_iterable(category_repeat_factors))
        category_repeat_factors = torch.tensor(category_repeat_factors)
        repeat_factors = torch.mul(category_repeat_factors, repeat_factors)
        repeat_factors = repeat_factors / torch.min(repeat_factors)
        logger.info(
            "Using WeightedCategoryTrainingSampler with repeat_factors={}".format(
                cfg.DATASETS.TRAIN_REPEAT_FACTOR
            )
        )
    else:
        logger.info(
            "Using WeightedTrainingSampler with repeat_factors={}".format(
                cfg.DATASETS.TRAIN_REPEAT_FACTOR
            )
        )

    sampler = RepeatFactorTrainingSampler(repeat_factors)
    return sampler


def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=(
                cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0
            ),
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        if isinstance(dataset, torchdata.IterableDataset):
            logger.info("Not using any sampler since the dataset is IterableDataset.")
            sampler = None
        else:
            logger.info("Using training sampler {}".format(sampler_name))
            if sampler_name == "TrainingSampler":
                sampler = TrainingSampler(len(dataset))
            elif sampler_name == "RepeatFactorTrainingSampler":
                repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                    dataset, cfg.DATALOADER.REPEAT_THRESHOLD, sqrt=cfg.DATALOADER.REPEAT_SQRT
                )
                sampler = RepeatFactorTrainingSampler(repeat_factors, seed=cfg.SEED)
            elif sampler_name == "RandomSubsetTrainingSampler":
                sampler = RandomSubsetTrainingSampler(
                    len(dataset), cfg.DATALOADER.RANDOM_SUBSET_RATIO
                )
            elif sampler_name == "WeightedTrainingSampler":
                sampler = _build_weighted_sampler(cfg)
            elif sampler_name == "WeightedCategoryTrainingSampler":
                sampler = _build_weighted_sampler(cfg, enable_category_balance=True)
            else:
                raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
    **kwargs,
):
    """
    Build a dataloader for object detection with some default features.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). It can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``.
            If ``dataset`` is map-style, the default sampler is a :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
            Sampler must be None if ``dataset`` is iterable.
        total_batch_size (int): total batch size across all workers.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
        collate_fn: a function that determines how to do batching, same as the argument of
            `torch.utils.data.DataLoader`. Defaults to do no collation and return a list of
            data. No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = TrainingSampler(len(dataset))
        assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs,
    )


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=(
            [
                cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)]
                for x in dataset_name
            ]
            if cfg.MODEL.LOAD_PROPOSALS
            else None
        ),
    )
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": (
            InferenceSampler(len(dataset))
            if not isinstance(dataset, torchdata.IterableDataset)
            else None
        ),
    }

def _wsi_test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=(
            [
                cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)]
                for x in dataset_name
            ]
            if cfg.MODEL.LOAD_PROPOSALS
            else None
        ),
    )
    return {
        "dataset": dataset,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": (
            InferenceSampler(len(dataset))
            if not isinstance(dataset, torchdata.IterableDataset)
            else None
        ),
        "cluster_parameter": cfg.DATASETS.CLUSTER_PARAMETER,
        # "bg_value": cfg.MODEL.PIXEL_MEAN,
        "bg_value": [0.0,0.0,0.0],
    }


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    *,
    mapper: Callable[[Dict[str, Any]], Any],
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> torchdata.DataLoader:
    """
    Similar to `build_detection_train_loader`, with default batch size = 1,
    and sampler = :class:`InferenceSampler`. This sampler coordinates all workers
    to produce the exact set of all samples.

    Args:
        dataset: a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). They can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper: a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler: a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers. Sampler must be None
            if `dataset` is iterable.
        batch_size: the batch size of the data loader to be created.
            Default to 1 image per worker since this is the standard when reporting
            inference time in papers.
        num_workers: number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


@configurable(from_config=_wsi_test_loader_from_config)
def build_detection_wsi_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    *,
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    cluster_parameter: CfgNode,
    bg_value: list = [0.0, 0.0, 0.0],
) -> torchdata.DataLoader:
    """
    Similar to `build_detection_train_loader`, with default batch size = 1,
    and sampler = :class:`InferenceSampler`. This sampler coordinates all workers
    to produce the exact set of all samples.

    Args:
        dataset: a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). They can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper: a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler: a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers. Sampler must be None
            if `dataset` is iterable.
        batch_size: the batch size of the data loader to be created.
            Default to 1 image per worker since this is the standard when reporting
            inference time in papers.
        num_workers: number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=(lambda batch: wsi_trivial_batch_collator(batch, cluster_parameter=cluster_parameter, bg_value=bg_value)) if collate_fn is None else collate_fn,
    )

    # patch_size=[1000,1000], step=0.5, dbscan_thr=30, min_sample=1, expand=10

class WSIPatchDataset(torchdata.Dataset):
    def __init__(self, item, cluster_parameter, bg_value):
        self.item = item.copy()
        self.item.pop("annotations",None)
        self.patch_size = cluster_parameter.PATCH_SIZE
        self.region_size = cluster_parameter.REGION_SIZE
        self.step = cluster_parameter.STEP
        self.prior = cluster_parameter.PRIOR
        self.prior_mode = cluster_parameter.PRIOR_MODE
        self.prior_level = cluster_parameter.PRIOR_LEVEL
        self.roi_threshold = cluster_parameter.ROI_THRESHOLD
        self.wsi_prior = cluster_parameter.WSI_PRIOR
        self.pack_mode = cluster_parameter.PACK_MODE
        self.scale = cluster_parameter.SCALE
        # self.dbscan_thr = cluster_parameter.DBSCAN_THR
        # self.min_sample = cluster_parameter.MIN_SAMPLE
        self.expand = cluster_parameter.EXPAND
        self.adjust_parameter = cluster_parameter.ADJUST_PARAMETER
        self.nmm_thr = cluster_parameter.NMM_THR
        self.border = cluster_parameter.BORDER
        self.max_child_size = cluster_parameter.MAX_CHILD_SIZE
        self.config_canvas_size = cluster_parameter.CANVAS_SIZE
        self.canvas_size = None
        self.bg_value = bg_value
        self.region_mode = cluster_parameter.REGION_MODE
        self.region = cluster_parameter.REGION
        self.region_level = cluster_parameter.REGION_LEVEL
        self.region_file = cluster_parameter.REGION_FILE
        self.selection_ratio = cluster_parameter.SELECTION_RATIO
        self.cluster_nmm_thr = cluster_parameter.NMM_CLUSTER_THR
        self.max_cluster_size = cluster_parameter.MAX_CLUSTER_SIZE
        # self.preprocess()
        if self.region_mode == 'region':
            self.divide_region_by_prior()
        if self.prior is None:
            if self.wsi_prior is None:
                self.coordinates, self.coordinates_in_patch, self.canvas_id, self.padding = self.sliding_window_operation([0, 0, self.item['width'], self.item['height']])
            else:
                self.divide_by_prior()
        else:
            if self.prior_mode == 'bbox':
                self.divide_by_prior()
            elif self.prior_mode == 'assign':
                self.divide_by_canvas()
                
        if self.canvas_size is None:
            if self.config_canvas_size is not None:
                self.canvas_size = np.array([self.config_canvas_size]*self.coordinates.shape[0])

    def __len__(self):
        if len(self.canvas_id) > 0:
            return self.canvas_id.max()+1
        else:
            return 0

    def preprocess(self):
        self.item['image'] = self.item['image'][0].numpy()

        per995 = np.percentile(self.item['image'], 99.5)
        per005 = np.percentile(self.item['image'], 0.5)

        self.item['image'][self.item['image'] > per995] = per995
        self.item['image'][self.item['image'] < per005] = per005

        ma = self.item['image'].max()
        mi = self.item['image'].min()
        r = ma - mi
        self.item['image'] = 255.0 * (self.item['image'] - mi) / r + 0.5

        preprocessed = cv2.equalizeHist(self.item['image'].astype(np.uint8))
        self.item['image'] = torch.tensor(preprocessed[None]).repeat(3,1,1)
    
    def sliding_window_operation(self, region):
        patch_size = self.patch_size
        step = self.step

        x_coords = np.arange(0, region[2], int(patch_size[0]*step))
        x_coords = x_coords[(x_coords+patch_size[0])<region[2]]
        x_coords = np.append(x_coords,region[2] - patch_size[0]).astype(float)
        
        y_coords = np.arange(0, region[3], int(patch_size[1]*step))
        y_coords = y_coords[(y_coords+patch_size[1])<region[3]]
        y_coords = np.append(y_coords,region[3] - patch_size[1]).astype(float)
        
        x, y = np.meshgrid(x_coords, y_coords, indexing='ij')

        coordinates = np.stack((x.reshape(-1), y.reshape(-1))).transpose((1,0))
        coordinates = np.hstack((coordinates, 
                                      np.full((coordinates.shape[0], 2), patch_size)))
        coordinates[:,:2] += region[:2]
        
        coordinates_in_patch = coordinates.copy()
        coordinates_in_patch[:,:2] = 0
        coordinates_in_patch[:,2:] /= self.scale
        
        canvas_id = np.arange(0,coordinates.shape[0])
        padding = np.zeros((coordinates.shape[0], 2))
        
        return coordinates, coordinates_in_patch, canvas_id, padding
        
    def get_coord_from_heatmap(self,region,all_prior):
        tmp_region = np.array(region)
        tmp_region[2:] += tmp_region[:2]
        prior = all_prior[filter_bboxes_in_patch(all_prior, tmp_region, 0.0)].copy()
        prior[:,2:4] -= prior[:,:2]
        
        first_stage_map = np.zeros(region[2:])
        counter = np.zeros(region[2:])
        for bbox in prior:
            bbox_in_region = (bbox[:-1] - np.array(list(region[:2])+[0,0])).astype(int)
            gaussion = compute_gaussian(bbox_in_region[2:], sigma_scale=1. / 8, value_scaling_factor=10)
            first_stage_map[bbox_in_region[0]:(bbox_in_region[0]+bbox_in_region[2]),bbox_in_region[1]:(bbox_in_region[1]+bbox_in_region[3])] += gaussion * bbox[-1]
            counter[bbox_in_region[0]:(bbox_in_region[0]+bbox_in_region[2]),bbox_in_region[1]:(bbox_in_region[1]+bbox_in_region[3])] += gaussion
        
        zero_mask = counter == 0
        first_stage_map[~zero_mask] = first_stage_map[~zero_mask] / counter[~zero_mask]
        probs_map = max_pooling(first_stage_map, pool_size=self.prior_level, stride=self.prior_level)
        POI = probs_map >= self.roi_threshold
        
        tmp_coordinates = []
        x_idxs, y_idxs = np.where(POI)
        if POI.sum() == 0:
            return [], []
        for idx in range(len(y_idxs)):
            x_mask = x_idxs[idx]
            y_mask = y_idxs[idx]
            x_center = np.clip(int((x_mask + 0.5) * self.prior_level[0] + region[0]), self.patch_size[0] // 2, region[0]+region[2] - self.patch_size[0] // 2)
            y_center = np.clip(int((y_mask + 0.5) * self.prior_level[1] + region[1]), self.patch_size[1] // 2, region[1]+region[3] - self.patch_size[1] // 2)
            x, w = x_center - self.patch_size[0] // 2, self.patch_size[0]
            y, h = y_center - self.patch_size[1] // 2, self.patch_size[1]
            pos_idx = np.where(first_stage_map[x:(x+w), y:(y+h)] > self.roi_threshold)
            scr = first_stage_map[x:(x+w), y:(y+h)][pos_idx].mean() if len(pos_idx[0]) > 0 else 0
            tmp_coordinates.append([x, y, w, h, scr])
            
        tmp_coordinates = expand_bboxes(tmp_coordinates,self.expand,region)
        cluster_dict = []
        for thr in self.nmm_thr:
            tmp_coordinates, new_cluster_dict = batch_nmm_by_size(tmp_coordinates, iou_threshold=thr,max_size=[m-self.border*2 for m in self.max_child_size])
        tmp_coordinates = expand_bboxes(tmp_coordinates,self.border,region)
    
        tmp_coordinates = tmp_coordinates[:,:-1]
        tmp_coordinates_in_patch = tmp_coordinates.copy()
        tmp_coordinates_in_patch[:,:2] = 0
        tmp_coordinates_in_patch[:,2:] /= self.scale
        return tmp_coordinates, tmp_coordinates_in_patch
            
    def get_coord(self,region,all_prior):
        tmp_region = np.array(region)
        tmp_region[2:] += tmp_region[:2]
        prior = all_prior[filter_bboxes_in_patch(all_prior, tmp_region, 0.0)].copy()
        prior[:,2:4] -= prior[:,:2]
        
        if len(prior) == 0:
            return np.zeros([0,4]), np.zeros([0,4]), np.zeros([0,2])
        
        tmp_coordinates = expand_bboxes(prior,self.expand,region)
        for thr in self.nmm_thr:
            tmp_coordinates, _ = batch_nmm_by_size(tmp_coordinates, iou_threshold=thr,max_size=[m-self.border*2 for m in self.max_child_size])
        
        if len(self.cluster_nmm_thr) > 0:
            tmp_cluster_coordinates = tmp_coordinates.copy()
            for thr in self.cluster_nmm_thr:
                tmp_cluster_coordinates, _ = batch_nmm_by_size(tmp_coordinates, iou_threshold=thr,max_size=[m-self.border*2 for m in self.max_cluster_size])
            tmp_coordinates = np.vstack((tmp_coordinates, tmp_cluster_coordinates))
            # tmp_coordinates = tmp_cluster_coordinates
        
        tmp_padding = []
        for idx in range(len(tmp_coordinates)):
            bbox = tmp_coordinates[idx].copy()
            x_mask = bbox[0]+bbox[2]/2
            y_mask = bbox[1]+bbox[3]/2
            
            w, h = max(bbox[2], self.patch_size[0]), max(bbox[3], self.patch_size[1])
            # w, h = bbox[2] + 20, bbox[3] + 20
            
            x_center = np.clip(int((x_mask + 0.5)), region[0] + w // 2, region[0] + region[2] - w // 2)
            y_center = np.clip(int((y_mask + 0.5)), region[1] + h // 2, region[1] + region[3] - h // 2)
            x, y = x_center - w // 2, y_center - h // 2
            
            tmp_coordinates[idx] = np.array([x, y, w, h, bbox[-1]])
            tmp_padding.append([w-bbox[2], h-bbox[3]])
        tmp_coordinates = tmp_coordinates[:,:-1]
        tmp_coordinates_in_patch = tmp_coordinates.copy()
        tmp_coordinates_in_patch[:,:2] = 0
        tmp_coordinates_in_patch[:,2:] /= self.scale
        return tmp_coordinates, tmp_coordinates_in_patch, tmp_padding
    
    def divide_region_by_prior(self):
        slide = openslide.open_slide(self.item["file_name"])
        scale_rate = slide.level_dimensions[0][0] / slide.level_dimensions[self.region_level][0]
        regions = []
        for f_id, f in enumerate(self.region_file):
            if f == self.item["file_name"]:
                regions.append([int(r * scale_rate) for r in self.region[f_id*4:(f_id+1)*4]])
        
        self.coordinates = []
        self.coordinates_in_patch = []
        self.canvas_id = []
        self.padding = []
        
        if len(regions) == 0:
            return
        
        if self.prior is None:
            for region in regions:
                tmp_coordinates, tmp_coordinates_in_patch, _, padding = self.sliding_window_operation(region)
                self.coordinates.append(tmp_coordinates)
                self.coordinates_in_patch.append(tmp_coordinates_in_patch)
                self.padding.append(padding)
        else:
            with open(os.path.join(self.prior,os.path.basename(self.item['file_name']).split('.')[0]+'.json'),'r') as f:
                all_prior = json.load(f)
            all_prior = np.array([[bbox['x'], bbox['y'], bbox['w']+bbox['x'], bbox['h']+bbox['y'], bbox['score']] for bbox in all_prior])
            for region in regions:
                # tmp_coordinates, tmp_coordinates_in_patch = self.get_coord_from_heatmap(region, all_prior)
                tmp_coordinates, tmp_coordinates_in_patch, tmp_padding = self.get_coord(region, all_prior)
                
                self.coordinates.append(tmp_coordinates)
                self.coordinates_in_patch.append(tmp_coordinates_in_patch)
                self.padding.append(tmp_padding)
        
        self.coordinates = np.vstack(self.coordinates)
        self.coordinates_in_patch = np.vstack(self.coordinates_in_patch)
        self.padding = np.vstack(self.padding)
        
        if self.selection_ratio < 1.0:
            num_patches = len(self.coordinates)
            num_selected = int(num_patches * self.selection_ratio)
            selected_indices = np.random.choice(num_patches, num_selected, replace=False)
            
            self.coordinates = self.coordinates[selected_indices]
            self.coordinates_in_patch = self.coordinates_in_patch[selected_indices]
            self.padding = self.padding[selected_indices]
            
        self.canvas_id = np.arange(0,self.coordinates.shape[0])
        # draw_bboxes(self.item["file_name"], "/media/ps/passport1/ltc/monuseg18/assign_prior", self.coordinates, self.canvas_id)

        if self.coordinates != []:
            print(f"max patch size:{int(self.coordinates[:,2].max()), int(self.coordinates[:,3].max())}")
            print(f"mean patch size:{int(self.coordinates[:,2].mean()), int(self.coordinates[:,3].mean())}")
            print(f"min patch size:{int(self.coordinates[:,2].min()), int(self.coordinates[:,3].min())}")
            print(f"patch number: {len(self.coordinates)}")
            print()  
        
    def divide_by_prior(self):
        if self.wsi_prior is not None:
            with h5py.File(os.path.join(self.wsi_prior,os.path.basename(self.item['file_name']).split('.')[0]+'.h5'), 'r') as h5_file:
                regions = h5_file['coords'][...]
            regions = np.hstack((regions, np.full((regions.shape[0], 2), self.region_size)))
        else:
            regions = [[0, 0, self.item['width'], self.item['height']]]
            
        self.coordinates = []
        self.coordinates_in_patch = []
        self.padding = []
        
        if self.prior is None:
            for region in regions:
                tmp_coordinates, tmp_coordinates_in_patch, _, padding = self.sliding_window_operation(region)
                self.coordinates.append(tmp_coordinates)
                self.coordinates_in_patch.append(tmp_coordinates_in_patch)
                self.padding.append(padding)
        else:
            with open(os.path.join(self.prior,os.path.basename(self.item['file_name']).split('.')[0]+'.json'),'r') as f:
                all_prior = json.load(f)
            all_prior = np.array([[bbox['x'], bbox['y'], bbox['w']+bbox['x'], bbox['h']+bbox['y'], bbox['score']] for bbox in all_prior])
            for region in regions:
                # tmp_coordinates, tmp_coordinates_in_patch = self.get_coord_from_heatmap(region, all_prior)
                tmp_coordinates, tmp_coordinates_in_patch, tmp_padding = self.get_coord(region, all_prior)
                
                self.coordinates.append(tmp_coordinates)
                self.coordinates_in_patch.append(tmp_coordinates_in_patch)
                self.padding.append(tmp_padding)
       
        self.coordinates = np.vstack(self.coordinates)
        self.coordinates_in_patch = np.vstack(self.coordinates_in_patch)
        self.padding = np.vstack(self.padding)
       
        if self.selection_ratio < 1.0:
            num_patches = len(self.coordinates)
            num_selected = int(num_patches * self.selection_ratio)
            selected_indices = np.random.choice(num_patches, num_selected, replace=False)
            
            self.coordinates = self.coordinates[selected_indices]
            self.coordinates_in_patch = self.coordinates_in_patch[selected_indices]
            self.padding = self.padding[selected_indices]
        
        self.canvas_id = np.arange(0,self.coordinates.shape[0])
        # draw_bboxes(self.item["file_name"], "/media/ps/passport1/ltc/monuseg18/assign_prior", self.coordinates, self.canvas_id)

        if self.coordinates is not []:
            print(f"max patch size:{int(self.coordinates[:,2].max()), int(self.coordinates[:,3].max())}")
            print(f"mean patch size:{int(self.coordinates[:,2].mean()), int(self.coordinates[:,3].mean())}")
            print(f"min patch size:{int(self.coordinates[:,2].min()), int(self.coordinates[:,3].min())}")
            print(f"patch number: {len(self.coordinates)}")
            print()  
        
    def divide_by_canvas(self):
        with open(self.prior,'r') as f:
            prior = json.load(f)
        prior = [p for p in prior if os.path.basename(self.item['file_name']).split('.')[0] in p['file_name']]
        
        coordinates = []
        coordinates_in_patch = []
        canvas_id = []
        canvas_size = []

        for id, p in enumerate(prior):
            canvas_size.append([p['bin_width'],p['bin_height']])
            for coord, coord_in_patch in zip(p['origin_cluster_box'],p['moved_cluster_box']):
                if (np.array(coord[2:])<=1).all():
                    continue
                if self.pack_mode == 'resize':
                    coordinates.append([coord[0],coord[1],coord[2]-coord[0],coord[3]-coord[1]])
                    coordinates_in_patch.append([coord_in_patch[0],coord_in_patch[1],max(coord_in_patch[2]-coord_in_patch[0],1),max(coord_in_patch[3]-coord_in_patch[1],1)])
                elif self.pack_mode == 'crop':
                    if coord[2]-coord[0] > coord_in_patch[2]-coord_in_patch[0] + 0.1:
                        x_center = (coord[0]+coord[2])/2
                        half_cropped_width = (coord_in_patch[2]-coord_in_patch[0])/2
                        coord[0] = x_center - half_cropped_width
                        coord[2] = x_center + half_cropped_width
                    elif coord[3]-coord[1] > coord_in_patch[3]-coord_in_patch[1] + 0.1:
                        y_center = (coord[1]+coord[3])/2
                        half_cropped_height = (coord_in_patch[3]-coord_in_patch[1])/2
                        coord[1] = y_center - half_cropped_height
                        coord[3] = y_center + half_cropped_height
                    coordinates.append([coord[0],coord[1],coord[2]-coord[0],coord[3]-coord[1]])
                    coordinates_in_patch.append([coord_in_patch[0],coord_in_patch[1],coord_in_patch[2]-coord_in_patch[0],coord_in_patch[3]-coord_in_patch[1]])
                
                canvas_id.append(id)

        self.coordinates = np.array(coordinates)
        self.coordinates_in_patch = np.array(coordinates_in_patch)
        self.canvas_id = np.array(canvas_id)
        self.canvas_size = np.array(canvas_size)
        self.padding = np.zeros((len(self.coordinates), 2))
        
        if len(self.coordinates) > 0:
            print(f"max patch size:{int(self.coordinates[:,2].max()), int(self.coordinates[:,3].max())}")
            print(f"mean patch size:{int(self.coordinates[:,2].mean()), int(self.coordinates[:,3].mean())}")
            print(f"min patch size:{int(self.coordinates[:,2].min()), int(self.coordinates[:,3].min())}")
            print(f"patch number: {len(self.coordinates)}")
            print()   
        
    def __getitem__(self, idx):   
        ids = np.where(self.canvas_id==idx)[0]
        
        coord = (self.coordinates[ids]).astype(int)
        coord_in_patch = (self.coordinates_in_patch[ids]).astype(int)

        
        if (self.canvas_size[idx] < 0).any():
            current_canvas_size = [max(coord_in_patch[:,0]+coord_in_patch[:,2]),max(coord_in_patch[:,1]+coord_in_patch[:,3])]
        else:
            current_canvas_size = self.canvas_size[idx]
            
        canvas = np.ones((current_canvas_size[1], current_canvas_size[0], 3)) * self.bg_value
        
        if self.item["file_name"].endswith(".jpg"):
            image = cv2.imread(self.item["file_name"])
            for cp, c in zip(coord_in_patch, coord):
                patch = image[c[1]:(c[1]+c[3]),c[0]:(c[0]+c[2])] #opencv默认读进来是BGR格式
                canvas[cp[1]:(cp[1]+cp[3]),cp[0]:(cp[0]+cp[2])] = resize_patch_to_target_size(patch, cp[2:])
                
                
        else:
            image = openslide.open_slide(self.item["file_name"])
            for cp, c in zip(coord_in_patch, coord):
                patch = np.array(image.read_region((c[0],c[1]),0,(c[2],c[3])))[:,:,-2::-1] # openslide默认读进来是RGBA格式
                canvas[cp[1]:(cp[1]+cp[3]),cp[0]:(cp[0]+cp[2])] = resize_patch_to_target_size(patch, cp[2:])
            
        canvas = torch.as_tensor(canvas.astype("float32").transpose(2, 0, 1))
        
        
        ret = self.item.copy()
        
        ret['image'] = canvas
        ret["width"] = current_canvas_size[0]
        ret["height"] = current_canvas_size[1]
        ret['padding'] = self.padding[idx]
        ret['coord'] = coord
        ret['coord_in_patch'] = coord_in_patch
        ret['image_id'] = self.item['image_id']+'_'+str(idx)
        # ret['border'] = self.border

        return ret



def wsi_trivial_batch_collator(batch, cluster_parameter, bg_value):
    for id in range(len(batch)):  
        dataset = WSIPatchDataset(batch[id], cluster_parameter=cluster_parameter, bg_value=bg_value)
        dataloader = torchdata.DataLoader(
            dataset,
            batch_size=len(batch),
            drop_last=False,
            num_workers=1,
            collate_fn=trivial_batch_collator,
        )
        batch[id]['dataloader'] = dataloader

    return batch


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2**31
    seed_all_rng(initial_seed + worker_id)



def draw_bboxes(image_path, output_path, bboxes, canvas_ids):
    """
    Draw bounding boxes on an image with unique colors based on canvas IDs and save the result.

    Args:
        image_path (str): Path to the input image (jpg format).
        output_path (str): Path to save the output image.
        bboxes (numpy.ndarray): [x, 4] array of bounding boxes. Each box is [x, y, w, h].
        canvas_ids (list): List of canvas IDs corresponding to each bounding box (length x).
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    
    if len(bboxes) != len(canvas_ids):
        raise ValueError("The length of bboxes and canvas_ids must be the same.")
    
    # Generate unique colors for each canvas_id
    unique_ids = list(set(canvas_ids))
    id_to_color = {
        cid: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
        for cid in unique_ids
    }
    
    # Draw bounding boxes
    for bbox, cid in zip(bboxes, canvas_ids):
        x, y, w, h = map(int, bbox)
        x_min, y_min, x_max, y_max = x, y, x + w, y + h  # Convert [x, y, w, h] to [xmin, ymin, xmax, ymax]
        color = id_to_color[cid]  # Use the color associated with the canvas_id
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)  # Thickness=2
    
    # Save the output image
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_file, image)
    print(f"Saved output image with bounding boxes to {output_file}")