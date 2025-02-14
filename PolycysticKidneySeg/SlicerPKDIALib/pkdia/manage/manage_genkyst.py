#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.transform import resize, rotate


def extract_genkyst_slice_prod(exam, idx, size):
    img = rotate(
        resize(np.squeeze(exam.volume.get_fdata()[:, idx, :])[::-1, :], output_shape=(size, size), preserve_range=True),
        90,
        preserve_range=True,
    )
    min_greyscale, max_greyscale = np.percentile(img, (1, 99))
    img = rescale_intensity(img, in_range=(min_greyscale, max_greyscale), out_range=(0, 1))
    return img_as_ubyte(img)
