#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# BSD 3-Clause License
#
# Copyright (c) 2025, Tristan Dagobert  tristan.dagobert@ens-paris-saclay.fr
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""
...
"""

import os
from os.path import exists, join, basename, dirname
import argparse
import timeit
import zipfile
from math import gcd

import numpy as np
from numpy.linalg import norm
from scipy import ndimage

from matplotlib import cm
import matplotlib.pyplot as plt

from numba import njit
#import imageio as iio
import iio
def convert_to_gray_image(cfg, img):
    """
    Convert an RGB image into a gray level one. If the image contains 4
    channels, we assume it is a Sentinel-2 image with the B04, B03, B02, B08
    channels storage in this order.
    """
    img = img[:, :, 0:3]
    img = np.mean(img, axis=-1)

    return img


def convert_to_rainbow_image(img, apply_log=True):
    """
    Make a jetcolor image map.
    """
    epsilon=1e-10
    if apply_log:
        img = np.log(img+epsilon)
        mini = np.min(img)
        maxi = np.max(img)

        img = 1.0 * (img - mini) / (maxi - mini)
        img = img.squeeze()
        img = np.uint8(255.0 * cm.jet(img)) #  pylint: disable=E1101
        img = img[:, :, 0:3]                #  pylint: disable=E1136

    return img


def normalize_image(img, sat=None):
    """
    …
    """
    img = img[:, :, 0:3]
    # convertir en float
    if sat is None:
        mini = np.min(img)
        maxi = np.max(img)
    else:
        val = np.sort(img.flatten())
        mini = val[int(sat*val.size)]
        maxi = val[int((1-sat)*val.size)]
        # remplacer les valeurs < mini ou > maxi par mini et maxi ... np.clip
    img = 255 * (img - mini) / (maxi - mini)
    img[img>255.0] = 255.0
    img[img<0.0] = 0.0

    img = np.array(img, dtype=np.uint8)
    return img


def perturbate_image(img):
    """
    Add small noise to image to avoid tie values during the Kolmogorov-Smirnov
    hypothesis test.
    """
    noise = np.random.rand(img.shape[0], img.shape[1])
    img = img + 1e-10 * noise
    return img


def saturate_image(img, sat=0.000):
    """
    ...
    """
    val = np.sort(img.flatten())
    maxi = val[int((1-sat)*val.size)-1]
    print(int((1-sat)*val.size), maxi)
    img[img > maxi] = maxi
    return img


def load_parameters():
    """
    …
    """

    desc = "Compute the changes between two images."
    parser = argparse.ArgumentParser(description=desc)
    subparsers = parser.add_subparsers(dest="action")


    f_parser = subparsers.add_parser(
        "series", help="N image series.")
    f_parser.add_argument(
        "--zip", type=str, required=True, help="Zip contenant 2N images."
    )
    f_parser.add_argument(
        "--epsilon", type=float, required=False, default=1.0,
        help="NFA threshold."
    )
    f_parser.add_argument(
        "--b", type=int, required=True,
        help="Side of the square neighborhood of x."
    )
    f_parser.add_argument(
        "--dirout", type=str, required=True, help="Output directory."
    )

    cfg = parser.parse_args()

    return cfg


@njit
def kolmogorov_smirnov(data1, data2):
    """
    ...
    """
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    assert data1.shape[0] == data2.shape[0]
    n_1 = data1.shape[0]

    if min(n_1, n_1) == 0:
        raise ValueError('Data passed to ks_2samp must not be empty')

    data_all = np.zeros(n_1 + n_1)
    data_all[0:n_1] = data1[:]
    data_all[n_1:] = data2[:]

    weight1 = np.ones(data1.shape)
    weight2 = np.ones(data1.shape)
    cwei1 = np.zeros(weight1.size + 1)
    cwei1[1: ] = np.cumsum(weight1) / np.sum(weight1)
    cwei2 = np.zeros(weight2.size + 1)
    cwei2[1: ] = np.cumsum(weight2) / np.sum(weight2)
    cdf1 = cwei1[np.searchsorted(data1, data_all, side='right')]
    cdf2 = cwei2[np.searchsorted(data2, data_all, side='right')]

#    print("maxi cdf1", np.max(cdf1))
    cddiffs = cdf1 - cdf2
    diff = np.max(cddiffs)

    valg = gcd(n_1, n_1)
    prob = -np.inf

    lcm = (n_1 // valg) * n_1
    valh = int(np.round(diff * lcm))
    diff = valh * 1.0 / lcm
    if valh == 0:
        return True, diff, 1.0
    # prob = binom(2n, n-h) / binom(2n, n)
    # Evaluating in that form incurs roundoff errors
    # from special.binom. Instead calculate directly
    jrange = np.arange(valh)
    prob = np.prod((n_1 - jrange) / (n_1 + jrange + 1.0))
    return True, diff, prob


@njit
def handle_boundaries(img):
    """
    Replacement of NaN values located on the edges,
    by the values located on the boundaries.
    Parameters
    ----------
    img : np.array ndim=(nrow, ncol, ncan)
    """
    nrow, ncol = img.shape

    # replacement of columns
    for i in np.arange(nrow):
        j = 0
        while j < ncol and np.isnan(img[i, j]):
            j += 1
        # entire line is NaN
        if j == ncol:
            continue
        # replacement of left columns
        img[i, 0:j] = img[i, j]

        while not np.isnan(img[i, j]):
            j += 1
        # replacement of right columns
        img[i, j:ncol] = img[i, j-1]

    # replacement of lines
    for j in np.arange(ncol):
        i = 0
        while i < nrow and np.isnan(img[i, j]):
            i += 1
        # entire column is NaN
        if i == nrow:
            continue
        # replacement of top lines
        img[0:i, j] = img[i, j]

        while not np.isnan(img[i, j]):
            i += 1
        # replacement of right colums
        img[i:nrow, j] = img[i-1, j]

    return img


def compute_theta(im1, im2):
    """
    Module gradient differences between the pixels of both images.
    """
    img = im1 - im2
    gh_img = ndimage.sobel(img, 0)  # horizontal gradient
    gv_img = ndimage.sobel(img, 1)  # vertical gradient
    grad_img = np.stack((gh_img, gv_img), axis=-1)

    module = norm(grad_img, axis=2)
    return module

@njit
def compute_pvalues(angle0, angle1, side):
    """
    Parameters
    ----------
    theta0 :
        The image to test.
    theta1 :
        The image of reference.
    side : int
        Side of the square neighborhood of x.
    """
    nrow, ncol = angle0.shape
    h_b = side // 2

    # initialization
    pval = np.nan * np.ones((nrow, ncol))
    # computation per pixel
    for x_i in np.arange(nrow):
        for x_j in np.arange(ncol):
            # limits tests
            if (x_i-h_b < 0 or nrow <= x_i+h_b
                or x_j-h_b < 0 or ncol <= x_j+h_b):
                continue
            # neighborhood of x
            tile0 = angle0[x_i-h_b:x_i+h_b+1, x_j-h_b:x_j+h_b+1].flatten()
            tile1 = angle1[x_i-h_b:x_i+h_b+1, x_j-h_b:x_j+h_b+1].flatten()
            _, _, pvalue = kolmogorov_smirnov(tile1, tile0)

            pval[x_i, x_j] = pvalue
    pval = handle_boundaries(pval)

    return pval


def load_images(cfg):
    """
    ...
    """
    with zipfile.ZipFile(cfg.zip, 'r') as monzip:
        pfxrep = [dirname(f) for f in monzip.namelist()][0]
        print("prefixe", pfxrep)
        monzip.extractall(path=cfg.dirout)
    files = sorted(os.listdir(join(cfg.dirout, pfxrep)))
#    files = [join(cfg.dirout, fic) for fic in files]

    # (u_n)
    files_u_n = files[0::2]
    files_v_n = files[1::2]
    print(files_u_n)
    print(files_v_n)
    im_n = []
    for u_n in files:
        print(f"{u_n}")
        im_n += [normalize_image(iio.read(join(cfg.dirout,u_n)), sat=0.01)]
    for i, u_n in enumerate(im_n):
        iio.write(join(cfg.dirout, f"input_{i}.png"), u_n)
    
    imu_n = [saturate_image(
        convert_to_gray_image(cfg, iio.read(join(cfg.dirout,u_n)))
    ) for u_n in files_u_n]
    imv_n = [saturate_image(
        convert_to_gray_image(cfg, iio.read(join(cfg.dirout,v_n)))
    ) for v_n in files_v_n]

    return files_u_n, files_v_n, imu_n, imv_n


def traiter(cfg):
    """
    ...
    """

    files_u_n, files_v_n, imu_n, imv_n = load_images(cfg)
    imu_0 = imu_n[-1]
    imv_0 = imv_n[-1]
    imu_n = imu_n[:-1]
    imv_n = imv_n[:-1]

    print(f"{len(imu_n)} {len(imv_n)}")
    assert len(imu_n) == len(imv_n)
    nlig, ncol = imu_0.shape

    # strategy 6
    nsample = 0
    theta_u0_v0 = compute_theta(imu_0, imv_0)

    mappes = []
    for i, imu_i in enumerate(imu_n):
        for j, imv_j in enumerate(imv_n):
            print(f"paire u{i} v{j} = {files_u_n[i]} {files_v_n[j]}")
            nsample += 1
            theta_ui_vj = compute_theta(imu_i, imv_j)
            # compute Boolean map
            mappe = compute_pvalues(theta_u0_v0, theta_ui_vj, cfg.b)
            mappes += [mappe]

    # computation of the NFA according to the median p-value
    median = np.stack(mappes, axis=2)
    median = np.median(median, axis=2)
    nfa = nsample * nlig * ncol * median
    iio.write(join(cfg.dirout, "median_strat6.tif"), nfa)

    img = convert_to_rainbow_image(median)
    iio.write(join(cfg.dirout, "median_strat6.png"), img)

    nfa = 255 * (nfa < cfg.epsilon)
    iio.write(join(cfg.dirout, "nfa_strat6.png"), nfa)
    return 0


def main():
    """
    ...
    """
    print("VERSION 1")
    cfg = load_parameters()
    print(cfg)
    if not exists(cfg.dirout):
        os.mkdir(cfg.dirout)

    if cfg.action == "series":
        traiter(cfg)

    return 0


if __name__ == "__main__":
    execution_time = timeit.timeit(main, number=1)
    print(f"Execution time: {execution_time:6.3f} seconds")
