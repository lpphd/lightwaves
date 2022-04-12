# -*- coding: utf-8 -*-


# This code is based on the sktime ROCKET code, which has the following license
# BSD 3-Clause License
#
# Copyright (c) 2019 - 2020 The sktime developers.
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

import numpy as np
from numba import njit
from numba import prange


@njit(
    "Tuple((float32[:],int32[:],float32[:],int32[:],int32[:],int32[:],"
    "int32[:]))(int32,float32[:,:],int32[:],optional(int32))",
    cache=True,
)
def _generate_first_phase_kernels(n_columns, candidate_kernels, candidate_dilations, seed):
    """
    Generates kernels with all dilations for each input channel, with suitable padding so that output of convolution remains the same length as input
    :param n_columns: Number of channels of dataset (slice)
    :param candidate_kernels: The set of base kernels used by LightWaveS
    :param candidate_dilations: The set of base dilations used by LightWaveS
    :param seed: Random seed
    :return: Tuple of information similar to ROCKET with the weights, dilations, paddings, etc to be used during transformation
    """
    if seed is not None:
        np.random.seed(seed)

    num_kernels = int(len(candidate_kernels) * len(candidate_dilations) * n_columns)
    lengths = np.zeros(num_kernels, dtype=np.int32)
    num_channel_indices = np.ones(num_kernels, dtype=np.int32)
    channel_indices = np.repeat(np.arange(n_columns), int(len(candidate_kernels) * candidate_dilations.size)).astype(
        np.int32)

    w_l = 0
    for k in candidate_kernels:
        w_l += len(k) * n_columns * candidate_dilations.size

    weights = np.zeros(int(w_l),
                       dtype=np.float32
                       )

    biases = np.zeros(num_kernels).astype(np.float32)
    paddings = np.zeros(num_kernels, dtype=np.int32)
    dilations = np.zeros(num_kernels, dtype=np.int32)

    a1 = 0  # for weights

    c = 0
    for ch in range(n_columns):
        for i in range(len(candidate_kernels)):
            _length = len(candidate_kernels[i])
            for j in candidate_dilations:
                dilations[c] = 2 ** j
                paddings[c] = ((_length - 1) * (2 ** j)) // 2
                b1 = a1 + _length
                weights[a1:b1] = candidate_kernels[i]
                a1 = b1
                lengths[c] = _length
                c += 1

    return (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        num_channel_indices,
        channel_indices,
    )


@njit(fastmath=True, cache=True)
def _apply_kernel(X, weights, length, bias, dilation, padding):
    """
    Perform convolution of kernel with input, return output of convolution and features
    :param X: Input vector
    :param weights: Kernel weights
    :param length: Kernel length
    :param bias: Kernel bias
    :param dilation: Kernel dilation
    :param padding: Padding to apply to input
    :return: (Convolution_output, np array of features)
    """
    n_timepoints = len(X)

    output_length = ((n_timepoints + (2 * padding)) - ((length - 1) * dilation)) // 2

    _output = np.zeros(output_length)

    _ppv = 0

    _max = np.NINF

    _min = np.PINF

    end = (n_timepoints + padding) - ((length - 1) * dilation)

    store = True
    store_idx = 0
    c = 0
    _ls = 0
    for i in range(-padding, end):

        _sum = bias

        index = i

        for j in range(length):

            if index > -1 and index < n_timepoints:
                _sum = _sum + weights[j] * X[index]

            index = index + dilation

        _sum /= (dilation * np.sqrt(dilation))

        if _sum > 0:
            c += 1
            _ppv += 1
        else:
            if c > _ls:
                _ls = c
            c = 0

        if abs(_sum) > _max:
            _max = abs(_sum)

        if abs(_sum) < _min:
            _min = abs(_sum)

        if store:
            _output[store_idx] = np.abs(_sum)
            store_idx += 1
            store = False
        else:
            store = True

    return _output, np.array(
        [_ppv / (end + padding), _max, _ls / (end + padding), _min]).reshape((1, -1))


@njit(fastmath=True, cache=True)
def _apply_kernel_features_only(X, weights, length, bias, dilation, padding):
    """
    Perform convolution of kernel with input, return only features
    :param X: Input vector
    :param weights: Kernel weights
    :param length: Kernel length
    :param bias: Kernel bias
    :param dilation: Kernel dilation
    :param padding: Padding to apply to input
    :return: Np array of features
    """
    n_timepoints = len(X)

    _ppv = 0

    _max = np.NINF
    _min = np.PINF

    end = (n_timepoints + padding) - ((length - 1) * dilation)

    _ls = 0
    c = 0

    for i in range(-padding, end):

        _sum = bias

        index = i

        for j in range(length):

            if index > -1 and index < n_timepoints:
                _sum = _sum + weights[j] * X[index]

            index = index + dilation

        _sum /= (dilation * np.sqrt(dilation))

        if _sum > 0:
            c += 1
            _ppv += 1
        else:
            if c > _ls:
                _ls = c
            c = 0

        if abs(_sum) > _max:
            _max = abs(_sum)

        if abs(_sum) < _min:
            _min = abs(_sum)

    return np.array([_ppv / (end + padding), _max, _ls / (end + padding), _min]).reshape(
        (1, -1))


@njit(
    "float32[:,:,:](float32[:,:,:],Tuple((float32[::1],int32[:],float32[:],"
    "int32[:],int32[:],int32[:],int32[:])))",
    parallel=True,
    fastmath=True,
    cache=True,
)
def _apply_2layer_kernels(X, kernels):
    """
    Apply all kernels to input
    :param X: The time series input of dimension (n_instances, channels, timesteps)
    :param kernels: The tuple with the kernel information, similar to ROCKET
    :return: Array of features, of dimension (n_instances,n_kernels,n_features)
    """
    (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        num_channel_indices,
        channel_indices,
    ) = kernels

    num_features = 4
    n_instances, n_columns, n_timepoints = X.shape
    num_kernels = len(lengths)

    _X = np.zeros(
        (n_instances, num_kernels, 2 * num_features), dtype=np.float32
    )

    for i in prange(n_instances):

        a1 = 0  # for weights
        a2 = 0  # for channel_indices
        a3 = 0  # for features

        for j in range(num_kernels):
            b1 = a1 + num_channel_indices[j] * lengths[j]
            b2 = a2 + num_channel_indices[j]
            b3 = a3 + 1

            conv_output, _X[i, a3:b3, :num_features] = _apply_kernel(
                X[i, channel_indices[a2]],
                weights[a1:b1],
                lengths[j],
                biases[j],
                dilations[j],
                paddings[j]
            )

            _X[i, a3:b3, num_features:2 * num_features] = _apply_kernel_features_only(
                conv_output,
                weights[a1:b1],
                lengths[j],
                biases[j],
                dilations[j],
                paddings[j]
            )

            a1 = b1
            a2 = b2
            a3 = b3

    return _X
