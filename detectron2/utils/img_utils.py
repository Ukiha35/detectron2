import starfile
import os
import mrcfile
import sys
from tqdm import tqdm
import numpy as np
import cv2
import json

def downsample_with_size(x, size1, size2):
    F = np.fft.rfft2(x)
    A = F[..., 0:size1 // 2, 0:size2 // 2 + 1]
    B = F[..., -size1 // 2:, 0:size2 // 2 + 1]
    F = np.concatenate([A, B], axis=0)

    a = size1 * size2
    b = x.shape[-2] * x.shape[-1]
    F *= (a / b)

    f = np.fft.irfft2(F, s=(size1, size2))

    return f.astype(x.dtype)

def quantize(x, dtype=np.uint8):
    x = clip_percentile(x)
    ma = x.max()
    mi = x.min()
    r = ma - mi
    x = 255.0 * (x - mi) / r + 0.5
    
    x = x.astype(dtype)
    return x
    # buckets = np.linspace(mi, ma, 255)
    # return np.digitize(x, buckets).astype(dtype)
    
def clip_percentile(x):
    x[x > np.percentile(x, 99.5)] = np.percentile(x, 99.5)
    x[x < np.percentile(x, 0.5)] = np.percentile(x, 0.5)
    return x

# def draw_rect(slide, l, t, height, weight, color=1, thickness=1):
#     r = l + height
#     b = t + weight
#     slide[l-thickness:l+thickness, t:b] = color
#     slide[r-thickness:r+thickness, t:b] = color
#     slide[l:r, t-thickness:t+thickness] = color
#     slide[l:r, b-thickness:b+thickness] = color
#     return slide

def draw_rect(slide, l, t, height, weight, color=1, thickness=1):
    r = l + height
    b = t + weight
    slide[np.clip(l-thickness,0,slide.shape[0]):np.clip(l+thickness,0,slide.shape[0]), np.clip(t,0,slide.shape[1]):np.clip(b,0,slide.shape[1])] = color
    slide[np.clip(r-thickness,0,slide.shape[0]):np.clip(r+thickness,0,slide.shape[0]), np.clip(t,0,slide.shape[1]):np.clip(b,0,slide.shape[1])] = color
    slide[np.clip(l,0,slide.shape[0]):np.clip(r,0,slide.shape[0]), np.clip(t-thickness,0,slide.shape[1]):np.clip(t+thickness,0,slide.shape[1])] = color
    slide[np.clip(l,0,slide.shape[0]):np.clip(r,0,slide.shape[0]), np.clip(b-thickness,0,slide.shape[1]):np.clip(b+thickness,0,slide.shape[1])] = color
    return slide


def bias_calculate_iou(bboxA, bboxB):
    # 解包边界框
    xA, yA, wA, hA = bboxA
    xB, yB, wB, hB = bboxB

    # 计算右下角坐标
    boxA = (xA, yA, xA + wA, yA + hA)
    boxB = (xB, yB, xB + wB, yB + hB)

    # 计算重叠区域的坐标
    x_overlap_A = max(boxA[0], boxB[0])
    y_overlap_A = max(boxA[1], boxB[1])
    x_overlap_B = min(boxA[2], boxB[2])
    y_overlap_B = min(boxA[3], boxB[3])

    # 计算重叠区域的宽度和高度
    inter_width = max(0, x_overlap_B - x_overlap_A)
    inter_height = max(0, y_overlap_B - y_overlap_A)

    # 计算重叠区域的面积
    intersection_area = inter_width * inter_height

    # 计算两个框的面积
    boxA_area = wA * hA
    boxB_area = wB * hB

    # 计算IoU
    union_area = boxA_area + boxB_area - intersection_area
    iou = intersection_area / boxA_area if boxA_area > 0 else 0
    
    return iou


def bbox_intersection(bboxA, bboxB):
    # 解包边界框
    xA, yA, wA, hA = bboxA
    xB, yB, wB, hB = bboxB

    # 计算右下角坐标
    xA_max = xA + wA
    yA_max = yA + hA
    xB_max = xB + wB
    yB_max = yB + hB

    # 计算交集的左上角和右下角坐标
    x_overlap_A = max(xA, xB)
    y_overlap_A = max(yA, yB)
    x_overlap_B = min(xA_max, xB_max)
    y_overlap_B = min(yA_max, yB_max)

    # 计算交集的宽度和高度
    inter_width = max(0, x_overlap_B - x_overlap_A)
    inter_height = max(0, y_overlap_B - y_overlap_A)

    # 如果交集存在，返回交集框；否则返回None
    if inter_width > 0 and inter_height > 0:
        return [x_overlap_A, y_overlap_A, inter_width, inter_height]
    else:
        return None  # 没有交集

