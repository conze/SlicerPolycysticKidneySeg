#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data.dataset import Dataset

from ..exams.exam_genkyst_prod import exam_genkyst_prod
from ..manage.manage_genkyst import extract_genkyst_slice_prod
from ..utils.utils import normalization_imgs


class tiny_dataset_genkyst_prod(Dataset):
    def __init__(self, inputPath, outputDir, size, modality, vgg: bool = False):
        self.inputPath = inputPath
        self.outputDir = outputDir
        self.size = size
        self.vgg = vgg
        self.modality = modality
        self.exam = exam_genkyst_prod(self.inputPath, self.outputDir, self.modality)
        self.exam.normalize()

    def __len__(self):
        return self.exam.volume.shape[1]

    def __getitem__(self, idx: int):
        img = extract_genkyst_slice_prod(self.exam, idx, self.size)
        if self.vgg:
            img_ = np.zeros(shape=(img.shape[0], img.shape[1], 3), dtype=np.float32)
        else:
            img_ = np.zeros(shape=(img.shape[0], img.shape[1], 1), dtype=np.float32)
        img_[:, :, 0] = img
        if self.vgg:
            img_[:, :, 1], img_[:, :, 2] = img, img
        img_ = normalization_imgs(img_)
        return img_.swapaxes(2, 0)
