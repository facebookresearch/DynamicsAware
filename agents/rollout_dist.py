# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

def measure_distances(rollouts_1,
                      masks_1,
                      rollouts_2,
                      masks_2,
                      cutoff,
                      time_slice=0):
    """Time slice controls which part of the rollouts to use for computing distance,
    0 corresponds to entire rollout (cropped to the length of the shorter of the pair), 
    a positive interger n will use the first n frames, a negative integer -n will use the last n frames"""
    moving_objects_mask = (rollouts_1[:, 0, :, [8, 9, 10, 12]] > 0.5).any(dim=2)
    joint_time_mask = (1 - masks_1) & (1 - masks_2)
    if time_slice != 0:
        time_slice_mask = torch.ones_like(joint_time_mask).cumsum(dim=1)
        if time_slice > 0:
            time_slice_mask = time_slice_mask <= time_slice
        else:
            length_per_episode = joint_time_mask.sum(dim=1)[:, None]
            time_slice_mask = time_slice_mask > (length_per_episode +
                                                  time_slice)
        joint_time_mask = joint_time_mask.bool() & time_slice_mask.bool()

    total_mask = moving_objects_mask[:, None, :].bool() & \
                  joint_time_mask[:, :, None].bool()

    coord_1 = rollouts_1[:, :, :, :2]
    coord_2 = rollouts_2[:, :, :, :2]

    dists = ((coord_1 - coord_2)**2).sum(dim=3)
    dists = dists**0.5
    dists[dists > cutoff] = cutoff
    dists[~total_mask] = 0.0
    dists = dists.sum(dim=1).sum(dim=1)

    normalizing_factors = total_mask.float().sum(dim=1).sum(
        dim=1) * cutoff
    return dists / normalizing_factors


def measure_distances_cross_task(rollouts_1, masks_1, rollouts_2, masks_2,
                                 cutoff):
    """A version of the measure distances suitable for cross template comparison, measures distances between red
    and greed object trajectories only"""
    moving_objects_mask = (rollouts_1[:, 0, :, [8, 9]] > 0.5).any(axis=0)
    joint_time_mask = (1 - masks_1) & (1 - masks_2)

    total_mask = (moving_objects_mask[:, None, :] &
                  joint_time_mask[:, :, None]).astype(bool)

    coord_1 = rollouts_1[:, :, :, :2]
    coord_2 = rollouts_2[:, :, :, :2]

    dists = ((coord_1 - coord_2)**2).sum(axis=3)
    dists = dists**0.5
    dists[dists > cutoff] = cutoff
    dists[~total_mask] = 0.0
    dists = dists.sum(axis=1).sum(axis=1)

    normalizing_factors = total_mask.astype(float).sum(axis=1).sum(
        axis=1) * cutoff

    return dists / normalizing_factors


def cartesian_product(array):
    n = array.shape[0]
    arr_1 = array[None, :].repeat_interleave(n, 0).flatten(0, 1)
    arr_2 = array[:, None].repeat_interleave(n, 1).flatten(0, 1)
    return arr_1, arr_2


def measure_all_distances(rollouts, masks, cutoff, cross_task=False, time_slice=0):
    n = rollouts.shape[0]
    rollouts_1, rollouts_2 = cartesian_product(rollouts)
    masks_1, masks_2 = cartesian_product(masks)
    if cross_task:
        return measure_distances_cross_task(rollouts_1, masks_1, rollouts_2,
                                            masks_2, cutoff).view(n, n)
    else:
        return measure_distances(rollouts_1, masks_1, rollouts_2, masks_2,
                                 cutoff, time_slice).view(n, n)
