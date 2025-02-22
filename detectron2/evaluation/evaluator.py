# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.structures import Instances,Boxes
from tqdm import tqdm
import os,json
import numpy as np

def filter_bboxes_in_patch(bboxes, patch, ratio=0.5):
    """
    筛选出在 patch 内部至少有一定比例端点在 patch 中的 bbox 的索引。

    Args:
        bboxes (numpy.ndarray): 二维数组，shape 为 (N, 4)，每一行是 [xmin, ymin, xmax, ymax]。
        patch (tuple): patch 的范围 (xmin, ymin, xmax, ymax)。
        ratio (float): 扩展比例，用于动态调整 patch 的边界。

    Returns:
        numpy.ndarray: 符合条件的 bbox 索引。
    """
    
    # 解包 patch 范围
    patch_xmin, patch_ymin, patch_xmax, patch_ymax = patch

    # 计算每个 bbox 的宽度和高度
    bbox_widths = bboxes[:, 2] - bboxes[:, 0]
    bbox_heights = bboxes[:, 3] - bboxes[:, 1]

    # 根据比例扩展 patch 的边界
    expanded_patch_xmin = patch_xmin - bbox_widths * ratio
    expanded_patch_ymin = patch_ymin - bbox_heights * ratio
    expanded_patch_xmax = patch_xmax + bbox_widths * ratio
    expanded_patch_ymax = patch_ymax + bbox_heights * ratio

    # 检查 bbox 的四个端点是否都在扩展后的 patch 中
    inside_xmin = bboxes[:, 0] >= expanded_patch_xmin
    inside_ymin = bboxes[:, 1] >= expanded_patch_ymin
    inside_xmax = bboxes[:, 2] <= expanded_patch_xmax
    inside_ymax = bboxes[:, 3] <= expanded_patch_ymax

    # 计算最终的筛选条件
    selected_mask = inside_xmin & inside_ymin & inside_xmax & inside_ymax

    # 返回符合条件的索引
    return torch.where(selected_mask)[0]

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
    model,
    data_loader,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    callbacks=None,
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.
        callbacks (dict of callables): a dictionary of callback functions which can be
            called at each stage of inference.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        dict.get(callbacks or {}, "on_start", lambda: None)()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            dict.get(callbacks or {}, "before_inference", lambda: None)()
            outputs = model(inputs)
            dict.get(callbacks or {}, "after_inference", lambda: None)()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()
        dict.get(callbacks or {}, "on_end", lambda: None)()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


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

def inference_on_wsi_dataset(
    model,
    data_loader,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    save_dir,
    callbacks=None,
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.
        callbacks (dict of callables): a dictionary of callback functions which can be
            called at each stage of inference.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = np.array([[len(d['dataloader']) for d in input] for input in data_loader]).sum()  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    output_file = os.path.join(save_dir, 'log', f"result_log.json")
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            ori_time = json.load(f)
        total_compute_time += ori_time['compute_time']
        total_data_time += ori_time['data_time']
        total_eval_time += ori_time['eval_time']
    except:
        if not os.path.exists(os.path.join(save_dir, 'log')):
            os.makedirs(os.path.join(save_dir, 'log'))
        with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=4)
                
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        idx = 0
        dict.get(callbacks or {}, "on_start", lambda: None)()
        for inputs in data_loader:
            total_data_time += time.perf_counter() - start_data_time

            total_additional_outputs = []
            total_outputs = []
            for wsi in inputs:
                print(f"processing {wsi['file_name']}...")
                if os.path.exists(os.path.join(save_dir,os.path.basename(wsi['file_name'].split('.')[0]+'.json'))):
                    continue
                wsi_outputs = []
                patch_start_data_time = time.perf_counter()
                for patch in tqdm(wsi['dataloader']):
                    total_data_time += time.perf_counter() - patch_start_data_time
                    
                    start_compute_time = time.perf_counter()
                    dict.get(callbacks or {}, "before_inference", lambda: None)()
                    
                    output = model(patch)
                    
                    dict.get(callbacks or {}, "after_inference", lambda: None)()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    total_compute_time += time.perf_counter() - start_compute_time
                    
                    ########################################################################
                    # patch postprocessing
                    ########################################################################
                    
                    for i in range(len(output)):
                        coord_in_patch = patch[i]['coord_in_patch']
                        coord = patch[i]['coord']
                        image_size = (wsi['width'],wsi['height'])
                        device = output[i]["instances"].pred_boxes.tensor.device
                        # border = patch[i]['border']
                        total_hit_bbox = []
                        total_hit_scores = []
                        total_hit_classes = []
                        for cp,c in zip(coord_in_patch, coord):
                            interval = np.array([0,0,0,0])
                            # interval = np.array([(0 if c[id]==lim else 1) for id, lim in enumerate([0,0,image_size[0],image_size[1]])])
                            
                            patch_border = cp.copy()
                            patch_border[2:] += patch_border[:2]
                            patch_border[2:] -= interval[2:]
                            patch_border[:2] += interval[:2]
                            
                            hit_idx = filter_bboxes_in_patch(output[i]["instances"].pred_boxes.tensor,patch_border,0.1)

                            scale = [c[2]/cp[2], c[3]/cp[3]]
                            hit_bbox = output[i]["instances"][hit_idx].pred_boxes.tensor
                            hit_bbox -= torch.tensor(np.tile(cp[0:2], 2), device=device)
                            hit_bbox *= torch.tensor(scale*2).to(device)
                            hit_bbox += torch.tensor(np.tile(c[0:2], 2), device=device)

                            total_hit_bbox.extend(hit_bbox)
                            total_hit_scores.extend(output[i]["instances"][hit_idx].scores)
                            total_hit_classes.extend(output[i]["instances"][hit_idx].pred_classes)
                            
                    ########################################################################
                    # patch postprocessing
                    ########################################################################
                        if len(total_hit_bbox)>0:
                            wsi_outputs.append(Instances(image_size, pred_boxes=Boxes(torch.stack(total_hit_bbox)),
                                scores=torch.stack(total_hit_scores),
                                pred_classes=torch.stack(total_hit_classes)))                            

                    if idx == num_warmup:
                        start_time = time.perf_counter()
                        total_data_time = 0
                        total_compute_time = 0
                        total_eval_time = 0   

                    idx += 1     
                    patch_start_data_time = time.perf_counter()

                    '''
                    with torch.no_grad():  
                        for p in patch:
                            p['image'].to(torch.device('cuda'))
                            
                        start_compute_time = time.perf_counter()
                        dict.get(callbacks or {}, "before_inference", lambda: None)()
                        
                        output = model(patch)
                        
                        dict.get(callbacks or {}, "after_inference", lambda: None)()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        total_compute_time += time.perf_counter() - start_compute_time
                        
                        for o in output:
                            o['instances'].to(torch.device("cpu"))
                        wsi_outputs.extend(output)
                
                    '''
                if len(wsi_outputs) > 0:
                    total_outputs.append({"instances":Instances.cat(wsi_outputs)})
            
            
            

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, total_outputs)
            
            total_eval_time += time.perf_counter() - start_eval_time
            evaluator.save(save_dir)
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            results['compute_time'] = total_compute_time
            results['data_time'] = total_data_time
            results['eval_time'] = total_eval_time
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
                
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()
        dict.get(callbacks or {}, "on_end", lambda: None)()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    # 根据.json判断是否已经推理过了，但需要最后把推理过的结果load上来
    evaluator.load(save_dir)
    
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
        
    if results is not None:
        with open(output_file, 'r', encoding='utf-8') as f:
            final_results = json.load(f)
        final_results.update({k: (v if isinstance(v, dict) else dict(v)) for k, v in results.items()})
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4)
        
    
    return results