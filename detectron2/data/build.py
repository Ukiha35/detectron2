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
from sklearn.cluster import DBSCAN
import cv2
import os
import json
import openslide
import h5py

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
]


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
        
        # 父类框：[x, y, w, h]
        parent_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
    
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
    drop_last: bool = True,
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
        assert drop_last, "Aspect ratio grouping will drop incomplete batches."
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
        "patch_size": cfg.DATASETS.PATCH_SIZE,
        "step": cfg.DATASETS.STEP,
        "scale": cfg.DATASETS.SCALE,
        "prior": cfg.INPUT.PRIOR,
        "dbscan_thr": cfg.DATASETS.DBSCAN_THR,
        "min_sample": cfg.DATASETS.MIN_SAMPLE,
        "expand": cfg.DATASETS.EXPAND,
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
    mapper: Callable[[Dict[str, Any]], Any],
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    patch_size: Tuple[int, int],
    step: float = 0.5,
    scale: float = 1.0,
    prior: str = None,
    dbscan_thr: int = 30,
    min_sample: int = 1,
    expand: int = 10,
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
    # if mapper is not None:
    #     dataset = MapDataset(dataset, mapper)

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
        collate_fn=(lambda batch: wsi_trivial_batch_collator(batch, patch_size, step, scale, prior, dbscan_thr, min_sample, expand)) if collate_fn is None else collate_fn,
    )

class WSIPatchDataset(torchdata.Dataset):
    def __init__(self, item,patch_size,step,scale,prior,dbscan_thr, min_sample, expand):
        self.item = item.copy()
        self.item.pop("annotations",None)
        self.patch_size = patch_size
        self.step = step
        self.prior = prior
        self.scale = scale
        self.dbscan_thr = dbscan_thr
        self.min_sample = min_sample
        self.expand = expand
        # self.preprocess()
        if self.prior is None:
            self.sliding_window_operation()
        else:
            self.divide_by_prior()

    def __len__(self):
        return len(self.coordinates)

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
    
    def sliding_window_operation(self):
        patch_size = self.patch_size
        step = self.step
        shape = [self.item['width'], self.item['height']]

        x_coords = np.arange(0, shape[0], int(patch_size[0]*step))
        x_coords = x_coords[(x_coords+patch_size[0])<shape[0]]
        x_coords = np.append(x_coords,shape[0] - patch_size[0])
        
        y_coords = np.arange(0, shape[1], int(patch_size[1]*step))
        y_coords = y_coords[(y_coords+patch_size[1])<shape[1]]
        y_coords = np.append(y_coords,shape[1] - patch_size[1])
        
        x, y = np.meshgrid(x_coords, y_coords, indexing='ij')

        self.coordinates = np.stack((x.reshape(-1), y.reshape(-1))).transpose((1,0))
        self.coordinates = np.hstack((self.coordinates, 
                                      np.full((self.coordinates.shape[0], 2), self.patch_size),
                                      np.full((self.coordinates.shape[0], 1), self.scale)))
    def divide_by_prior(self):
        try:
            with open(os.path.join(self.prior,os.path.basename(self.item['file_name']).split('.')[0]+'.json'),'r') as f:
                prior = json.load(f)
            self.coordinates = np.array([[bbox['x'], bbox['y'], bbox['w'], bbox['h']] for bbox in prior])
            self.coordinates = cluster_bboxes_with_dbscan(self.coordinates,self.dbscan_thr,self.min_sample)
            
            self.coordinates = np.array([[(bbox[0] + bbox[2] / 2) - (bbox[2]+self.expand) / 2, 
                             (bbox[1] + bbox[3] / 2) - (bbox[3]+self.expand) / 2, 
                              (bbox[2]+self.expand), (bbox[3]+self.expand)] for bbox in self.coordinates])
            
        except:
            with h5py.File(os.path.join(self.prior,os.path.basename(self.item['file_name']).split('.')[0]+'.h5'), 'r') as h5_file:
                self.coordinates = h5_file['coords'][...]
            self.coordinates = np.hstack((self.coordinates, np.full((self.coordinates.shape[0], 2), self.patch_size)))

        low_limit = [[0, 0]]*len(self.coordinates)
        high_limit = np.array([self.item['width'] - self.coordinates[:, 2], self.item['height'] - self.coordinates[:, 3]]).transpose((1,0))
        self.coordinates[:, :2] = np.clip(self.coordinates[:, :2], low_limit, high_limit)
        
        self.coordinates = np.hstack((self.coordinates, np.full((self.coordinates.shape[0], 1), self.scale)))
        
        print(f"max patch size:{int(self.coordinates[:,2].max()), int(self.coordinates[:,3].max())}")
        print(f"min patch size:{int(self.coordinates[:,2].min()), int(self.coordinates[:,3].min())}")
        print(f"patch number: {len(self.coordinates)}")

    def __getitem__(self, idx):
        coord = self.coordinates[idx][:-1]
        patch_scale = self.coordinates[idx][-1]
        
        if self.item["file_name"].endswith(".jpg"):
            image = cv2.imread(self.item["file_name"])
            patch = image[int(coord[1]):(int(coord[1]+coord[3])),int(coord[0]):(int(coord[0]+coord[2]))]
        else:
            image = openslide.open_slide(self.item["file_name"])
            patch = np.array(image.read_region((int(coord[0]),int(coord[1])),0,(int(coord[2]),int(coord[3]))))[:,:,:-1]
            
        scaled_coord = [int(p/patch_scale) for p in coord]
        patch = cv2.resize(patch, scaled_coord[2:])
        patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
        
        
        ret = self.item.copy()
        
        ret['image'] = patch
        ret["wsi_height_level_0"] = self.item["height"]
        ret["wsi_width_level_0"] = self.item["width"]
        ret["width"] = scaled_coord[2]
        ret["height"] = scaled_coord[3]
        ret['patch_offset'] = scaled_coord[:2]
        ret['patch_scale'] = coord[3]/scaled_coord[3]
        ret['image_id'] = self.item['image_id']+'_'+str(idx)

        return ret



def wsi_trivial_batch_collator(batch, patch_size=[1000,1000], step=0.5, scale=1.0, prior=None, dbscan_thr=30, min_sample=1, expand=10):#1000
    for id in range(len(batch)):  
        dataset = WSIPatchDataset(batch[id],patch_size=patch_size,step=step,scale=scale, prior=prior, dbscan_thr=dbscan_thr, min_sample=min_sample, expand=expand)
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
