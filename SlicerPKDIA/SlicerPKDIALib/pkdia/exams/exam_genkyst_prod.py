#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import nibabel
import numpy as np

from ..utils.modality import ModalityEnum
from ..utils.utils import normalization_imgs


class exam_genkyst_prod:  # PKDIAv2
    def __init__(self, inputPath, outputPath, modality):

        self.inputPath = inputPath
        self.outputPath = outputPath
        self.modality = modality
        self.volume = None
        self.exam_upload()

    def exam_upload(self):
        self.volume = nibabel.as_closest_canonical(nibabel.load(self.inputPath))

        _, inputFile = os.path.split(self.inputPath)
        inputFileName, ext = inputFile.split(os.extsep, 1)
        outputFile = inputFileName + "-prod." + ext

        if self.modality == ModalityEnum.CT:
            data = self.volume.get_fdata()
            data = np.transpose(data, [0, 2, 1])
            affine = self.volume.affine
            affine[:, [1, 2]] = affine[:, [2, 1]]
            self.volume = nibabel.Nifti1Image(data, affine=affine)

        nibabel.save(self.volume, os.path.join(self.outputPath, outputFile))

    def normalize(self):
        self.volume.get_fdata()[:, :, :] = normalization_imgs(self.volume.get_fdata())[:, :, :]
