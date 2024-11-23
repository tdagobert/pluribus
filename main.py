#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# BSD 3-Clause License
#
# Copyright (c) 2024, Tristan Dagobert  tristan.dagobert@ens-paris-saclay.fr
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
#import shutil
import argparse
import timeit
import zipfile
from math import gcd

import numpy as np
from numpy.linalg import norm
#from scipy.ndimage import gaussian_filter
#from scipy.special import factorial
from scipy import ndimage
#from scipy import stats

from matplotlib import cm

from numba import njit
#import imageio as iio
import iio
@njit
def kolmogorov_smirnov(data1, data2):
    """
    ...
    """
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    if min(n1, n2) == 0:
        raise ValueError('Data passed to ks_2samp must not be empty')

    data_all = np.zeros(n1+n2)
    data_all[0:n1] = data1[:]
    data_all[n1:] = data2[:]
#    data_all = np.concatenate([data1, data2])
    # using searchsorted solves equal data problem
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    cddiffs = cdf1 - cdf2

    # Identify the location of the statistic
#    argminS = np.argmin(cddiffs)
#    argmaxS = np.argmax(cddiffs)
#    loc_minS = data_all[argminS]
#    loc_maxS = data_all[argmaxS]

    # Ensure sign of minS is not negative.
#    minS = np.clip(-cddiffs[argminS], 0, 1)
#    maxS = cddiffs[argmaxS]

#    d = maxS
#     d = cddiffs[np.argmax(cddiffs)]
    d = np.max(cddiffs)
#    d_location = loc_maxS
#    d_sign = 1
    g = gcd(n1, n2)
#    n1g = n1 // g
#    n2g = n2 // g
    prob = -np.inf

    lcm = (n1 // g) * n2
    h = int(np.round(d * lcm))
    d = h * 1.0 / lcm
    if h == 0:
        return True, d, 1.0
    # prob = binom(2n, n-h) / binom(2n, n)
    # Evaluating in that form incurs roundoff errors
    # from special.binom. Instead calculate directly
    jrange = np.arange(h)
    prob = np.prod((n1 - jrange) / (n1 + jrange + 1.0))
    return True, d, prob


@njit
def handle_boundaries(img):
    """
    Replacement of NaN values located on the edges,
    by the values located on the boundaries.
    Parameters
    ----------
    img : np.array ndim=(nrow, ncol, ncan)
    """
    nrow, ncol, ncan = img.shape
    for k in np.arange(ncan):
        # replacement of columns
        for i in np.arange(nrow):
            j = 0
            while j < ncol and np.isnan(img[i, j, k]):
                j += 1
            # entire line is NaN
            if j == ncol:
                continue
            # replacement of left columns
            img[i, 0:j, k] = img[i, j, k]

            while not np.isnan(img[i, j, k]):
                j += 1
            # replacement of right columns
            img[i, j:ncol, k] = img[i, j-1, k]

        # replacement of lines
        for j in np.arange(ncol):
            i = 0
            while i < nrow and np.isnan(img[i, j, k]):
                i += 1
            # entire column is NaN
            if i == nrow:
                continue
            # replacement of top lines
            img[0:i, j, k] = img[i, j, k]

            while not np.isnan(img[i, j, k]):
                i += 1
            # replacement of right colums
            img[i:nrow, j, k] = img[i-1, j, k]

    return img


#@njit
#@jit(nopython=False)
def angular(dirout, im1, im2):
    """
    Angular differences between the pixels of both images.
    """
    gh_im1 = ndimage.sobel(im1, 0)  # horizontal gradient
    gv_im1 = ndimage.sobel(im1, 1)  # vertical gradient
    grad_im1 = np.stack((gh_im1, gv_im1), axis=-1)

    gh_im2 = ndimage.sobel(im2, 0)  # horizontal gradient
    gv_im2 = ndimage.sobel(im2, 1)  # vertical gradient
    grad_im2 = np.stack((gh_im2, gv_im2), axis=-1)
    magnitude = np.sqrt(gh_im1**2 + gv_im1**2)
    iio.write(os.path.join(dirout, "magnitude.tif"), magnitude)
    print(gh_im1.shape, grad_im1.shape)

#    w = np.tensordot(grad_im1, grad_im2, axes=(2))
#    print(w.shape)
#    exit()

    prodsca = (
        grad_im1[:,:,0] * grad_im2[:,:,0] + grad_im1[:,:,1] * grad_im2[:,:,1]
    )
    cosine = np.arccos(
        prodsca / (norm(grad_im1, axis=2)*norm(grad_im2, axis=2))
    )
#    cosine = prodsca
    cosine[np.isnan(cosine)] = 0.0
    print(cosine.shape)
    return cosine


def compute_change(imu0, imv0, imu1, imv1, cfg):
    """
    Parameters
    ----------
    imu0 : np.array ndim=(nrow, ncol)
        Reference image.
    imv0 : np.array ndim=(nrow, ncol)
        Compared image.
    imu1 : np.array ndim=(nrow, ncol)
        Reference image one year before.
    imv1 : np.array ndim=(nrow, ncol)
        Compared image one year before.
    b : int
        Side of the square neighborhood of x.
    """

    # computes the angular difference
    angle0 = angular(cfg.dirout, imu0, imv0)
    angle1 = angular(cfg.dirout, imu1, imv1)

    nrow, ncol = imu0.shape
    h_b = cfg.b // 2

    # initialization
    phi = np.nan * np.ones((nrow, ncol, 1))
#    uni1 = np.nan * np.ones((nrow, ncol, 1))
#    uni0 = np.nan * np.ones((nrow, ncol, 1))
    # computation per pixel
    for x_i in np.arange(nrow):
#        print(x_i)
        for x_j in np.arange(ncol):
            # limits tests
            if (x_i-h_b < 0 or nrow <= x_i+h_b
                or x_j-h_b < 0 or ncol <= x_j+h_b):
                continue
            # neighborhood of x
            tile0 = angle0[x_i-h_b:x_i+h_b+1, x_j-h_b:x_j+h_b+1].flatten()
            tile1 = angle1[x_i-h_b:x_i+h_b+1, x_j-h_b:x_j+h_b+1].flatten()

#            uniform = np.linspace(np.min(tile0), np.max(tile0), num=tile0.size)
#            _, pvalue = stats.ks_2samp(uniform, tile0)
#            uni0[x_i, x_j, 0] = pvalue
#            uniform = np.linspace(np.min(tile1), np.max(tile1), num=tile1.size)
#            _, pvalue = stats.ks_2samp(uniform, tile1)
#            uni1[x_i, x_j, 0] = pvalue
            # Kolmogorov-Smirnov test
#            _, pvalue = stats.ks_2samp(tile1, tile0, alternative="greater")
            _, _, pvalue = kolmogorov_smirnov(tile1, tile0)
#            print(pvalue)
            phi[x_i, x_j, 0] = pvalue
    phi = handle_boundaries(phi)
    return phi, angle0, angle1 #, uni0, uni1


def convert_to_gray_image(cfg, img):
    """
    Convert an RGB image into a gray level one. If the image contains 4
    channels, we assume it is a Sentinel-2 image with the B04, B03, B02, B08
    channels storage in this order.
    """
    if cfg.channel is None:
        img = img[:, :, 0:3]
        img = np.mean(img, axis=-1)
    else:
        img = img[:, :, cfg.channel]
    return img


def convert_to_rainbow_image(img, apply_log=True):
    """
    Make a jetcolor image map.
    """
#    epsilon=1-e20
    if apply_log:
        img = np.log(img)
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


def load_parameters():
    """
    …
    """

    desc = "Compute the changes between two images."
    parser = argparse.ArgumentParser(description=desc)
    subparsers = parser.add_subparsers(dest="action")

    a_parser = subparsers.add_parser("apparier", help="Between 4 images.")

    a_parser.add_argument(
        "--u0", type=str, required=True, help="First image."
    )
    a_parser.add_argument(
        "--v0", type=str, required=True, help="Second image."
    )
    a_parser.add_argument(
        "--u1", type=str, required=True, help="First image one year before."
    )
    a_parser.add_argument(
        "--v1", type=str, required=True, help="Second image one year before."
    )
    a_parser.add_argument(
        "--b", type=int, required=True,
        help="Side of the square neighborhood of x."
    )
    a_parser.add_argument(
        "--dirout", type=str, required=True, help="Output directory."
    )
    a_parser.add_argument(
        "--channel", type=int, required=False, help="Channel."
    )

    b_parser = subparsers.add_parser("zipper", help="Between 4 images.")
    b_parser.add_argument(
        "--zip", type=str, required=True, help="Zip contenant 4 images."
    )
    b_parser.add_argument(
        "--epsilon", type=float, required=False, default=1.0,
        help="NFA threshold."
    )
    b_parser.add_argument(
        "--b", type=int, required=True,
        help="Side of the square neighborhood of x."
    )
    b_parser.add_argument(
        "--dirout", type=str, required=True, help="Output directory."
    )
    b_parser.add_argument(
        "--channel", type=int, required=False, help="Channel."
    )

    cfg = parser.parse_args()

    return cfg


def main():
    """
    ...
    """

    cfg = load_parameters()
    if not exists(cfg.dirout):
        os.mkdir(cfg.dirout)
#    else:
#        shutil.rmtree(cfg.dirout)

    imu0, imv0, imu1, imv1 = None, None, None, None
    if cfg.action == "apparier":
        imu0 = iio.read(cfg.u0)
        imv0 = iio.read(cfg.v0)
        imu1 = iio.read(cfg.u1)
        imv1 = iio.read(cfg.v1)

    if cfg.action == "zipper":
        with zipfile.ZipFile(cfg.zip, 'r') as monzip:
            fichiers = sorted([basename(f) for f in monzip.namelist()])
            pfxrep = [dirname(f) for f in monzip.namelist()][0]
            print(pfxrep)
            monzip.extractall(path=cfg.dirout)
            print(
                "contenu du répertoire:",
                sorted(os.listdir(join(cfg.dirout, pfxrep)))
            )
            fichiers = sorted(os.listdir(join(cfg.dirout, pfxrep)))[-4:]
            print(fichiers)
            print(f"u1={fichiers[0]} v1={fichiers[1]} u0={fichiers[2]}  v0={fichiers[3]}")
            imu0 = iio.read(join(cfg.dirout, pfxrep, fichiers[2]))
            imv0 = iio.read(join(cfg.dirout, pfxrep, fichiers[3]))
            imu1 = iio.read(join(cfg.dirout, pfxrep, fichiers[0]))
            imv1 = iio.read(join(cfg.dirout, pfxrep, fichiers[1]))

    for img, name in zip([imu0, imv0, imu1, imv1],
                         ["imu0.png", "imv0.png", "imu1.png", "imv1.png"]):
        img_normalized = normalize_image(img, sat=0.05)
        iio.write(join(cfg.dirout, name), img_normalized)
        
    imu0 = convert_to_gray_image(cfg, imu0)
    imv0 = convert_to_gray_image(cfg, imv0)
    imu1 = convert_to_gray_image(cfg, imu1)
    imv1 = convert_to_gray_image(cfg, imv1)

    imu0 = perturbate_image(imu0)
    imv0 = perturbate_image(imv0)
    imu1 = perturbate_image(imu1)
    imv1 = perturbate_image(imv1)

    epsilon = 1.0
    phi, angle0, angle1 = compute_change(imu0, imv0, imu1, imv1, cfg)
    nfa1 = imu0.shape[0] * imu0.shape[1] * phi
    mappe1 = 255 * np.array(nfa1 < epsilon, dtype=np.uint8)

    jet = convert_to_rainbow_image(phi)
    for img, name in zip(
            [angle0, angle1, phi, jet, mappe1],
            ["angle_u0_v0.tif", "angle_u1_v1.tif", "phi1.tif",
             "jet1.png", "map1.png"]):
        iio.write(join(cfg.dirout, name), img)

    #iio.write(join(cfg.dirout, "uni0.tif"), uni0)
    #iio.write(join(cfg.dirout, "uni1.tif"), uni1)

#    iio.write(join(cfg.dirout, "angle_u0_v0.tif"), angle0)
#    iio.write(join(cfg.dirout, "angle_u1_v1.tif"), angle1)
#
#    iio.write(join(cfg.dirout, "phi1.tif"), phi)
#    iio.write(join(cfg.dirout, "jet1.png"), jet)
#
#    iio.write(join(cfg.dirout, "map1.png"), mappe1)

    phi, angle0, angle1 = compute_change(imv1, imv0, imu1, imu0, cfg)
    nfa2 = imu0.shape[0] * imu0.shape[1] * phi
    mappe2 = 255 * np.array(nfa2 < epsilon, dtype=np.uint8)

    jet = convert_to_rainbow_image(phi)
    for img, name in zip(
            [angle0, angle1, phi, jet, mappe2],
            ["angle_v1_v0.tif", "angle_u1_v0.tif", "phi2.tif",
             "jet2.png", "map2.png"]):
        iio.write(join(cfg.dirout, name), img)
#    iio.write(join(cfg.dirout, "angle_v1_v0.tif"), angle0)
#    iio.write(join(cfg.dirout, "angle_u1_v0.tif"), angle1)
#    iio.write(join(cfg.dirout, "phi2.tif"), phi)
#    iio.write(join(cfg.dirout, "jet2.png"), jet)
#    iio.write(join(cfg.dirout, "map2.png"), mappe2)

    mappe = (mappe1 * mappe2)
    iio.write(join(cfg.dirout, "map.png"), mappe)

    mappe3 = 255 * np.array((nfa1 + nfa2)/2.0 < epsilon, dtype=np.uint8)
    iio.write(join(cfg.dirout, "map_avgnfa.png"), mappe3)
    return 0


if __name__ == "__main__":
    execution_time = timeit.timeit(main, number=1)
    print(f"Execution time: {execution_time:.6f} seconds")

# python main.py --u0
# for c in 0 1 2; do for i in 03 05 07 09 11; do python main.py --u0 santjordi/2018-07-12_S2B# _orbit_008_tile_31TDF_L1C_band_RGBI.tif --v0 santjordi/2019-01-03_S2A_orbit_008_tile_31TDF_# L1C_band_RGBI.tif --u1 santjordi/2017-07-12_S2A_orbit_008_tile_31TDF_L1C_band_RGBI.tif --v1#santjordi/2018-01-18_S2A_orbit_008_tile_31TDF_L1C_band_RGBI.tif --b $i --dirout vois_${i}_# channel_${c} --channel $c & done; done
