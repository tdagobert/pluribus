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
import argparse
import timeit
import zipfile
from math import gcd

import numpy as np
from numpy.linalg import norm
from scipy import ndimage
from scipy import stats

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
    
    val = np.sort(img.flatten())
    maxi = val[int((1-sat)*val.size)-1]
    print(int((1-sat)*val.size), maxi)
    img[img > maxi] = maxi
    return img

def load_images(cfg):
    """
    ...
    """
    with zipfile.ZipFile(cfg.zip, 'r') as monzip:
        fichiers = sorted([basename(f) for f in monzip.namelist()])
        pfxrep = [dirname(f) for f in monzip.namelist()][0]
        print(pfxrep)
        monzip.extractall(path=cfg.dirout)
        print(
            "contenu du répertoire:",
            sorted(os.listdir(join(cfg.dirout, pfxrep)))
        )
    fichiers = sorted(os.listdir(join(cfg.dirout, pfxrep)))

    liste_img_brut = [iio.read(join(cfg.dirout, pfxrep, i)) for i in fichiers]
    for img, name in zip(
        liste_img_brut,
        ["imu2.png", "imv2.png", "imu1.png", "imv1.png", "imu0.png", "imv0.png"]
    ):
        img_normalized = normalize_image(img, sat=0.005)
        iio.write(join(cfg.dirout, name), img_normalized)


    liste_img_gray = [convert_to_gray_image(cfg, img) for img in liste_img_brut]
    liste_img = [perturbate_image(img) for img in liste_img_gray]
    return liste_img


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
    f_parser.add_argument(
        "--channel", type=int, required=False, help="Channel."
    )
    f_parser.add_argument(
        "--feature", type=str, required=False, choices=["angle", "magnitude"],
        default="angle", help="..."
    )

    cfg = parser.parse_args()

    return cfg


@njit
#@njit(nopython=False)
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
    ncan = 1
    for k in np.arange(ncan):
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


#@njit
#@jit(nopython=False)
#def angular(cfg, im1, im2, with_mag=False):
#    """
#    Angular differences between the pixels of both images.
#    """
#    gh_im1 = ndimage.sobel(im1, 0)  # horizontal gradient
#    gv_im1 = ndimage.sobel(im1, 1)  # vertical gradient
#    grad_im1 = np.stack((gh_im1, gv_im1), axis=-1)
#
#    gh_im2 = ndimage.sobel(im2, 0)  # horizontal gradient
#    gv_im2 = ndimage.sobel(im2, 1)  # vertical gradient
#    grad_im2 = np.stack((gh_im2, gv_im2), axis=-1)
#    magnitude = None
#    if with_mag:
#        magnitude = norm(grad_im1, axis=2) + norm(grad_im2, axis=2)
#
#    if cfg.feature == "magnitude":
#        magnitude = np.sqrt(gh_im1**2 + gv_im1**2)
#        iio.write(os.path.join(cfg.dirout, "magnitude.tif"), magnitude)
#        return magnitude
#    print(gh_im1.shape, grad_im1.shape)
#
##    w = np.tensordot(grad_im1, grad_im2, axes=(2))
##    print(w.shape)
##    exit()
#
#    prodsca = (
#        grad_im1[:,:,0] * grad_im2[:,:,0] + grad_im1[:,:,1] * grad_im2[:,:,1]
#    )
#    cosine = np.arccos(
#        prodsca / (norm(grad_im1, axis=2)*norm(grad_im2, axis=2))
#    )
##    cosine = prodsca
#    cosine[np.isnan(cosine)] = 0.0
#    print(cosine.shape)
#    return cosine, magnitude
#
#@njit
#@jit(nopython=False)
def compute_phi(im1, im2):
    """
    ...
    """
    return np.abs(im1 - im2)

def compute_omega(im1, im2):
    """
    Module
    """
    """
    Angular differences between the pixels of both images.
    """
    img = im1 - im2
    gh_img = ndimage.sobel(img, 0)  # horizontal gradient
    gv_img = ndimage.sobel(img, 1)  # vertical gradient
    grad_img = np.stack((gh_img, gv_img), axis=-1)

    module = norm(grad_img, axis=2)
    return module

def compute_khi(im1, im2):
    """
    ...
    """
    """
    Angular differences between the pixels of both images.
    """
    gh_im1 = ndimage.sobel(im1, 0)  # horizontal gradient
    gv_im1 = ndimage.sobel(im1, 1)  # vertical gradient
    grad_im1 = np.stack((gh_im1, gv_im1), axis=-1)

    gh_im2 = ndimage.sobel(im2, 0)  # horizontal gradient
    gv_im2 = ndimage.sobel(im2, 1)  # vertical gradient
    grad_im2 = np.stack((gh_im2, gv_im2), axis=-1)

    prodsca = (
        grad_im1[:,:,0] * grad_im2[:,:,0] + grad_im1[:,:,1] * grad_im2[:,:,1]
    )
    cosine = np.arccos(
        prodsca / (norm(grad_im1, axis=2)*norm(grad_im2, axis=2))
    )
    cosine[np.isnan(cosine)] = 0.0
    return cosine


@njit
def compute_theta(imu, imv, half_l):
    """
    Parameters
    ----------
    imu : np.array ndim=(nrow, ncol)
        Reference image.
    imv : np.array ndim=(nrow, ncol)
        Compared image.
    half_l : int
        Half side of the square patch.
    """

    nrow, ncol = imu.shape

    # initialization
    phi_uvl = np.nan * np.ones((nrow, ncol))

    # computation per pixel
    for x_i in np.arange(nrow):
        for x_j in np.arange(ncol):
            # limits tests
            if (x_i - half_l < 0 or nrow <= x_i + half_l
                or x_j - half_l < 0 or ncol <= x_j + half_l):
                continue

            # neighborhood of x
            tilu = imu[x_i-half_l:x_i+half_l+1, x_j-half_l:x_j+half_l+1]
            y_i = x_i
            y_j = x_j

            # limits tests
            if (y_i - half_l < 0 or nrow <= y_i + half_l
                or y_j - half_l < 0 or ncol <= y_j + half_l):
                continue

            # neighborhood of y
            tilv = imv[y_i-half_l:y_i+half_l+1, y_j-half_l:y_j+half_l+1]

            # calcul de la distance
            suu = np.sum(tilu*tilu)
            svv = np.sum(tilv*tilv)
            phi_uvl[x_i, x_j] = (
                max(suu, svv) * (1 - np.sum(tilu * tilv)**2 / (suu * svv))
            )

    phi_uvl = handle_boundaries(phi_uvl)
    return phi_uvl


@njit
def compute_map(angle0, angle1, b):
    """
    Parameters
    ----------
    theta0 :
        The image to test.
    theta1 :
        The image of reference.
    b : int
        Side of the square neighborhood of x.
    """
    nrow, ncol = angle0.shape
    h_b = b // 2

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


#@njit
# comdef compute_map2(angle0, angle1, b, epsilon):
# com    """
# com    Parameters
# com    ----------
# com    theta0 :
# com        The image to test.
# com    theta1 :
# com        The image of reference.
# com    b : int
# com        Side of the square neighborhood of x.
# com    """
# com    nrow, ncol = angle0.shape
# com    h_b = b // 2
# com
# com    # initialization
# com    pval = np.nan * np.ones((nrow, ncol))
# com    # computation per pixel
# com    for x_i in np.arange(nrow):
# com#        print(x_i)
# com        for x_j in np.arange(ncol):
# com            # limits tests
# com            if (x_i-h_b < 0 or nrow <= x_i+h_b
# com                or x_j-h_b < 0 or ncol <= x_j+h_b):
# com                continue
# com            # neighborhood of x
# com            tile0 = angle0[x_i-h_b:x_i+h_b+1, x_j-h_b:x_j+h_b+1].flatten()
# com            tile1 = angle1[x_i-h_b:x_i+h_b+1, x_j-h_b:x_j+h_b+1].flatten()
# com
# com#            _, pvalue = stats.ks_2samp(tile1, tile0)
# com            _, _, pvalue = kolmogorov_smirnov(tile1, tile0)
# com
# com            pval[x_i, x_j] = pvalue
# com    pval = handle_boundaries(pval)
# com    nfa = nrow * ncol * pval
# com    mappe_v = 1 * nfa < epsilon
# com#    mappe_v = np.uint8(mappe_v)
# com    return mappe_v
# com#    return pval, angle0, angle1


def traiter(cfg):
    """
    ...
    """
    with zipfile.ZipFile(cfg.zip, 'r') as monzip:
        fichiers = sorted([basename(f) for f in monzip.namelist()])
        pfxrep = [dirname(f) for f in monzip.namelist()][0]
        print("prefixe", pfxrep)
        monzip.extractall(path=cfg.dirout)
#        print(sorted(os.listdir(join(cfg.dirout, pfxrep))))
    files = sorted(os.listdir(join(cfg.dirout, pfxrep)))
    files = [join(cfg.dirout, fic) for fic in files]

    # (u_n)
    files_u_n = files[0::2]
    files_v_n = files[1::2]
    print(files_u_n)
    print(files_v_n)
#    imu_n = []
#    for u_n in files_u_n:
#        print(f"image={u_n}")
#        a = iio.read(u_n)
#        imu_n += [convert_to_gray_image(cfg, a)]
    imu_n = [saturate_image(convert_to_gray_image(cfg, iio.read(u_n))) for u_n in files_u_n]
    imv_n = [saturate_image(convert_to_gray_image(cfg, iio.read(v_n))) for v_n in files_v_n]
    imu_0, imu_1 = imu_n[-1], imu_n[-2]
    imv_0, imv_1 = imv_n[-1], imv_n[-2]
    imu_n = imu_n[:-1]
    imv_n = imv_n[:-1]
#com    theta_u1_u0 = compute_theta(imu_0, imu_1)
#com    theta_v1_v0 = compute_theta(imv_0, imv_1)
    # strategy 1

    # strategy 2
    print(f"{len(imu_n)} {len(imv_n)}")
    assert len(imu_n) == len(imv_n)
    nb = 0
    nlig, ncol = imu_1.shape
#com    avg_map = np.zeros((nlig, ncol))#, dtype=np.float64)
#com    for i in range(len(imu_n) - 1):
#com        for j in range(i+1, len(imu_n)):
#com            nb += 1
#com            print(f"paire u {files_u_n[i]} {files_u_n[j]}")
#com            theta_ui_uj = compute_theta(imu_n[i], imu_n[j])
#com            print(f"paire v {files_v_n[i]} {files_v_n[j]}")
#com            theta_vi_vj = compute_theta(imv_n[i], imv_n[j])
#com            # compute Boolean map
#com            map_u = compute_map(theta_u1_u0, theta_ui_uj, cfg.b, cfg.epsilon)
#com            map_v = compute_map(theta_v1_v0, theta_vi_vj, cfg.b, cfg.epsilon)
#com            # difference
#com            map_d = map_v - map_u * map_v
#com            avg_map += map_d
#com    avg_map /= nb
#com    iio.write(join(cfg.dirout, "avg_map_strat2.tif"), 255 * avg_map)
#com
#com    # strategy 4
#com    nb = 0
#com    avg_map = np.zeros((nlig, ncol))#, dtype=np.float)
#com    theta_u0_v0 = compute_theta(imu_0, imv_0)
#com    for i in range(len(imu_n)):
#com        for j in range(i, len(imu_n)):
#com            print(f"paire u, v {files_u_n[i]} {files_v_n[j]}")
#com            nb += 1
#com            theta_ui_vj = compute_theta(imu_n[i], imv_n[j])
#com            # compute Boolean map
#com            mappe = compute_map(theta_u0_v0, theta_ui_vj, cfg.b, cfg.epsilon)
#com            avg_map += mappe
#com    avg_map /= nb
#com    iio.write(join(cfg.dirout, "avg_map_strat4.tif"), 255 * avg_map)
#com
    # strategy 5
    nb = 0
    avg_map = np.zeros((nlig, ncol))#, dtype=np.float)
# com    scale = 3
    #theta_u0_v0 = compute_theta(imu_0, imv_0, scale)
    theta_u0_v0 = compute_omega(imu_0, imv_0)

    mappes = []
    for i in range(len(imu_n)):
        for j in range(len(imu_n)):
            print(f"paire u, v {files_u_n[i]} {files_v_n[j]}")
            nb += 1
#            theta_ui_vj = compute_theta(imu_n[i], imv_n[j], scale)
            theta_ui_vj = compute_omega(imu_n[i], imv_n[j])
            # compute Boolean map
#            mappe = compute_map(theta_u0_v0, theta_ui_vj, cfg.b, cfg.epsilon)
            mappe = compute_map(theta_u0_v0, theta_ui_vj, cfg.b)
            mappes += [mappe]

    # computation of the NFA according to the median p-value
    median = np.stack(mappes, axis=2)
    median = np.median(median, axis=2)
    nfa = nlig * ncol * nb * median
    iio.write(join(cfg.dirout, "median_strat5.tif"), nfa)

    return

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
    print(f"Execution time: {execution_time:.6f} seconds")
