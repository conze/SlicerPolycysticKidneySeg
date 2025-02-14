import logging
import os

import nibabel
import numpy as np
import torch
from skimage.transform import resize, rotate
from torch.utils.data import DataLoader

from .datasets.dataset_genkyst import tiny_dataset_genkyst_prod
from .nets import swinv2Unet
from .utils.utils import get_array_affine_header, getLargestConnectedArea, prob2mask


def applyPKDIA(inputPath, outputDir, modality, weightsPath, verbose=False):
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    img_size = 256
    n_classes = 1
    net = swinv2Unet.SwinV2TwoDecoder(
        model_name="swinv2_cr_tiny_ns_224",
        pretrained=True,
        img_size=(img_size, img_size),
        in_chans=1,
        n_classes_1dec=n_classes,
        n_classes_2dec=n_classes,
    )
    vgg = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        logging.info(f"using device {device}")

    net.to(device=device)
    net.load_state_dict(torch.load(weightsPath, map_location=device, weights_only=True))

    if verbose:
        logging.info("model loaded !")

    test_dataset = tiny_dataset_genkyst_prod(inputPath, outputDir, img_size, modality, vgg)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    array_LK, affine, header = get_array_affine_header(test_dataset)
    array_RK = array_LK.copy()
    array = array_LK.copy()

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            image = data
            image = image.to(device=device, dtype=torch.float32)

            mask_LK = prob2mask(torch.sigmoid(net(image)[0]))
            mask_RK = prob2mask(torch.sigmoid(net(image)[1]))

            # == LK
            mask_LK = rotate(mask_LK, -90, preserve_range=True)
            mask_LK = resize(
                mask_LK,
                output_shape=(test_dataset.exam.volume.shape[0], test_dataset.exam.volume.shape[2]),
                preserve_range=True,
            )
            mask_LK[np.where(mask_LK > 0.95)] = 1
            mask_LK[np.where(mask_LK != 1)] = 0
            array_LK[0 : test_dataset.exam.volume.shape[0], idx, 0 : test_dataset.exam.volume.shape[2]] = mask_LK[
                ::-1, ::
            ]

            # == RK
            mask_RK = rotate(mask_RK, -90, preserve_range=True)
            mask_RK = resize(
                mask_RK,
                output_shape=(test_dataset.exam.volume.shape[0], test_dataset.exam.volume.shape[2]),
                preserve_range=True,
            )
            mask_RK[np.where(mask_RK > 0.95)] = 1
            mask_RK[np.where(mask_RK != 1)] = 0
            array_RK[0 : test_dataset.exam.volume.shape[0], idx, 0 : test_dataset.exam.volume.shape[2]] = mask_RK[
                ::-1, ::
            ]

        array_nopp = array_LK + array_RK
        array_nopp[np.where(array_nopp > 0.0)] = 1
        prediction_nopp = nibabel.Nifti1Image(
            array_nopp.astype(np.uint16), affine=affine, header=header
        )  # no post-processing

        array_LK = getLargestConnectedArea(array_LK)
        array_RK = getLargestConnectedArea(array_RK)
        array = array_LK + array_RK
        array[np.where(array > 0.0)] = 1

        prediction_LK = nibabel.Nifti1Image(array_LK.astype(np.uint16), affine=affine, header=header)
        prediction_RK = nibabel.Nifti1Image(array_RK.astype(np.uint16), affine=affine, header=header)

        prediction = nibabel.Nifti1Image(array.astype(np.uint16), affine=affine, header=header)

        _, inputFile = os.path.split(inputPath)
        inputFileName, ext = inputFile.split(os.extsep, 1)

        predLKPath = os.path.join(outputDir, inputFileName + "-prediction-LK." + ext)
        preRKPath = os.path.join(outputDir, inputFileName + "-prediction-RK." + ext)
        predPath = os.path.join(outputDir, inputFileName + "-prediction." + ext)
        predNoppPath = os.path.join(outputDir, inputFileName + "-prediction-nopp." + ext)
        nibabel.save(prediction_LK, predLKPath)
        nibabel.save(prediction_RK, preRKPath)
        nibabel.save(prediction, predPath)
        nibabel.save(prediction_nopp, predNoppPath)
        return predLKPath, preRKPath, predPath, predNoppPath
