#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum, EnumMeta

import numpy as np
from skimage.measure import label


class ModalityEnumMeta(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class ModalityEnum(str, Enum, metaclass=ModalityEnumMeta):
    T2 = "MRI T2"
    CT = "CT"


def normalization_imgs(imgs):
    """centering and reducing data structures"""
    imgs = imgs.astype(np.float32, copy=False)
    mean = np.mean(imgs)  # mean for data centering
    std = np.std(imgs)  # std for data normalization
    if np.int32(std) != 0:
        imgs -= mean
        imgs /= std
    return imgs


def get_array_affine_header(test_dataset):
    array = np.zeros(test_dataset.exam.volume.shape, dtype=np.uint16)
    affine, header = test_dataset.exam.volume.affine, test_dataset.exam.volume.header
    return array, affine, header


def prob2mask(prob):
    mask = prob.squeeze().cpu().numpy()
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    return mask.swapaxes(0, 1).astype(np.uint8)


def getLargestConnectedArea(segmentation):
    if len(np.unique(segmentation)) == 1:
        return segmentation
    else:
        labels = label(segmentation, connectivity=1)
        unique, counts = np.unique(labels, return_counts=True)
        list_seg = list(zip(unique, counts))[1:]  # the 0 label is by default background so take the rest
        largest = max(list_seg, key=lambda x: x[1])[0]
        labels_max = (labels == largest).astype(int)
        return labels_max
