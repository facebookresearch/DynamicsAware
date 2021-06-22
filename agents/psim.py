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

import torch
import phyre
import numpy as np

import multiprocessing
from multiprocessing import sharedctypes
import ctypes
import math

from types import SimpleNamespace

import sys
import pdb


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

            # worker = SimulationWorker(task_ids, tier, child_con, max_len,
                                      # sim_status_bufffer, feature_buffer,
                                      # img_buffer, mask_buffer,
                                      # task_index_buffer, action_buffer,
                                      # max_batch_size, action_dim)


class DummyResult:

    def __init__(self, features=None, imgs=None):
        self.featurized_objects = SimpleNamespace()
        self.featurized_objects.features = features
        self.status = phyre.SimulationStatus.NOT_SOLVED
        self.images = imgs 


def standardize_img_rollout_shape(img_rollout_arrays, time_dim_length):
    """Give list of feature arrays, returns them bunched up in a single matrix with fixed length across T and N
    
    Also returns padding"""
    n = len(img_rollout_arrays)
    standardized_arr = np.zeros(
        (n, time_dim_length, phyre.SCENE_HEIGHT, phyre.SCENE_WIDTH),
        dtype=np.float32)
    for i, input_arr in enumerate(img_rollout_arrays):
        for t, time_step_arr in enumerate(input_arr):
            standardized_arr[i, t, :len(time_step_arr)] = time_step_arr
    
    masks = [
        gen_masks(DummyResult(imgs=img_rollout_arrays[i]),
                  time_dim_length,
                  has_images=True,
                  has_features=False)[3] for i in range(n)
    ]
    masks = np.stack(masks)
    return standardized_arr, masks

def standardize_features_shape(feature_arrays, time_dim_length):
    """Give list of feature arrays, returns them bunched up in a single matrix with fixed length across T and N
    
    Also returns padding"""
    n = len(feature_arrays)
    standardized_arr = np.zeros(
        (n, time_dim_length, MAX_N_OBJECT_IN_SCENE, FEATURIZED_OBJECT_DIM),
        dtype=np.float32)
    for i, input_arr in enumerate(feature_arrays):
        for t, time_step_arr in enumerate(input_arr):
            standardized_arr[i, t, :len(time_step_arr)] = time_step_arr
    
    masks = [
        gen_masks(DummyResult(feature_arrays[i]),
                  time_dim_length,
                  has_images=False,
                  has_features=True)[3] for i in range(n)
    ]
    masks = np.stack(masks)
    return standardized_arr, masks


class SimulationWorker(multiprocessing.Process):

    def __init__(self, task_ids, tier, pipe, max_len, status_buffer,
                 feature_buffer, scene_buffer, mask_buffer, task_ind_buffer,
                 action_buffer, max_batch_size, action_dim):
        super().__init__()
        self.task_ids = task_ids
        self.tier = tier
        self.pipe = pipe
        self.max_len = max_len

        if scene_buffer is not None:
            self.scene_array = np.frombuffer(scene_buffer, dtype=np.long,
                                             count=max_batch_size * max_len * phyre.SCENE_WIDTH * phyre.SCENE_HEIGHT)\
                                                 .reshape(max_batch_size, max_len, phyre.SCENE_WIDTH, phyre.SCENE_HEIGHT)
        else:
            self.scene_array = None

        if feature_buffer is not None:
            self.feature_array = np.frombuffer(feature_buffer, dtype=np.float32, count=max_batch_size * max_len * MAX_N_OBJECT_IN_SCENE * FEATURIZED_OBJECT_DIM)\
                                              .reshape(max_batch_size, max_len, MAX_N_OBJECT_IN_SCENE, FEATURIZED_OBJECT_DIM)
        else:
            self.feature_array = None

        self.status_array = np.frombuffer(status_buffer,
                                          dtype=np.int32,
                                          count=max_batch_size)

        if scene_buffer is not None or feature_buffer is not None:
            self.mask_array = np.frombuffer(
                mask_buffer, dtype=np.byte,
                count=max_batch_size * max_len).reshape(max_batch_size, max_len)

        self.action_buffer = np.frombuffer(action_buffer, dtype=np.float32, count=max_batch_size * action_dim)\
                                           .reshape(max_batch_size, action_dim)
        self.task_ind_buffer = np.frombuffer(task_ind_buffer,
                                             dtype=np.long,
                                             count=max_batch_size)

    def run(self):
        #TODO make work when neither images or features are wanted
        self.simulator = phyre.initialize_simulator(self.task_ids, self.tier)
        while 1:
            start_ind, n, stride, requires_images, requires_features = self.pipe.recv(
            )

            for i in range(start_ind, start_ind + n):
                task_ind = self.task_ind_buffer[i]
                action = self.action_buffer[i]
                status, imgs, features, masks = self.sim_single(
                    task_ind, action, stride, requires_images,
                    requires_features)

                if requires_images:
                    self.scene_array[i] = imgs

                if requires_features:
                    for t, obj_repr in enumerate(features):
                        self.feature_array[i, t, :len(obj_repr)] = obj_repr

                self.mask_array[i] = masks
                self.status_array[i] = int(status)
            self.pipe.send(1)

    def sim_single(self, task_ind, action, stride, requires_images,
                   requires_features):
        results = self.simulator.simulate_action(
            task_ind,
            action,
            stride=stride,
            need_featurized_objects=requires_features,
            need_images=requires_images)

        return gen_masks(results, self.max_len, requires_images,
                        requires_features)


def gen_masks(results, max_length, has_images, has_features):
    imgs = None
    features = None
    masks = None
    if results.status == phyre.SimulationStatus.INVALID_INPUT:
        if has_features or has_images:
            masks = np.ones(
                (max_length,), dtype=np.uint8
            )  #the dtype looks wrong but it actually works like this

        if has_images:
            imgs = np.zeros(
                (max_length, phyre.SCENE_WIDTH, phyre.SCENE_HEIGHT),
                dtype=np.uint8)
        if has_features:
            features = np.zeros(
                (max_length, MAX_N_OBJECT_IN_SCENE, FEATURIZED_OBJECT_DIM),
                dtype=np.float32)

    else:
        if has_features or has_images:
            masks = np.zeros((max_length,), dtype=np.uint8)
            episode_length = results.images.shape[0] if has_images\
                    else results.featurized_objects.features.shape[0]
            masks[episode_length:] = 1

        if has_images:
            padding = np.zeros((max_length - episode_length,
                                phyre.SCENE_WIDTH, phyre.SCENE_HEIGHT),
                               dtype=np.uint8)
            imgs = np.concatenate([results.images, padding], axis=0)

        if has_features:
            features = results.featurized_objects.features

    return int(results.status), imgs, features, masks


def collate(batch):
    return torch.stack([x[0] for x in batch
                       ]), torch.stack([x[1] for x in batch])


MAX_N_OBJECT_IN_SCENE = 65
FEATURIZED_OBJECT_DIM = 14


class ParallelPhyreSimulator:

    def __init__(self,
                 task_ids,
                 tier,
                 num_workers,
                 max_len,
                 max_batch_size,
                 requires_imgs=True,
                 requires_featurized=True):
        self.pipes = []
        self.workers = []

        sim_status_buffer = sharedctypes.RawArray(
            ctypes.c_int32, max_batch_size)
        self.sim_status_array = np.frombuffer(sim_status_buffer,
                                              dtype=np.int32,
                                              count=max_batch_size)

        if requires_featurized:
            img_buffer = sharedctypes.RawArray(
                ctypes.c_long, (max_batch_size * max_len * phyre.SCENE_WIDTH *
                                phyre.SCENE_HEIGHT))
            self.img_array = np.frombuffer(img_buffer, dtype=np.long,
                            count=max_batch_size * max_len * phyre.SCENE_WIDTH * phyre.SCENE_HEIGHT)\
                                .reshape(max_batch_size, max_len, phyre.SCENE_WIDTH, phyre.SCENE_HEIGHT)
        else:
            img_buffer = None
            self.img_array = None

        if requires_featurized:
            feature_buffer = sharedctypes.RawArray(
                ctypes.c_float, max_batch_size * max_len *
                MAX_N_OBJECT_IN_SCENE * FEATURIZED_OBJECT_DIM)
            self.feature_array = np.frombuffer(feature_buffer,
                                               dtype=np.float32,
                                               count=max_batch_size * max_len *
                                               MAX_N_OBJECT_IN_SCENE *
                                               FEATURIZED_OBJECT_DIM)\
                                              .reshape(max_batch_size, max_len, MAX_N_OBJECT_IN_SCENE, FEATURIZED_OBJECT_DIM)

        else:
            feature_buffer = None
            self.feature_array = None
        if requires_featurized or requires_imgs:
            mask_buffer = sharedctypes.RawArray(
                ctypes.c_uint8, max_batch_size * max_len)
            self.mask_array = np.frombuffer(mask_buffer, dtype=np.uint8, count=max_batch_size * max_len)\
                          .reshape(max_batch_size, max_len)
        else:
            mask_buffer = None
            self.mask_array = None

        action_dim = 3 if tier == "ball" else 6

        action_buffer = sharedctypes.RawArray(
            ctypes.c_float, max_batch_size * action_dim)
        task_index_buffer = sharedctypes.RawArray(
            ctypes.c_long, max_batch_size)

        self.action_array = np.frombuffer(
            action_buffer, dtype=np.float32,
            count=max_batch_size * action_dim).reshape(max_batch_size,
                                                       action_dim)
        self.task_ind_array = np.frombuffer(task_index_buffer,
                                            dtype=np.long,
                                            count=max_batch_size)
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers

        for i in range(num_workers):
            par_con, child_con = multiprocessing.Pipe()
            self.pipes.append(par_con)
            worker = SimulationWorker(task_ids, tier, child_con, max_len,
                                      sim_status_buffer, feature_buffer,
                                      img_buffer, mask_buffer,
                                      task_index_buffer, action_buffer,
                                      max_batch_size, action_dim)
            worker.start()
            self.workers.append(worker)

        # The following code segment is to make the ParallelPhyreSimulator constuctor wait for each worker to finish initializing its phyre.Simulator before returning
        self.action_array[:] = 0.0
        self.task_ind_array[:] = 0
        for pipe in self.pipes:
            pipe.send((0, 0, 0, 0, 0))
        for pipe in self.pipes:
            pipe.recv()

    def simulate_parallel(self,
                          batch_task_inds,
                          batch_actions,
                          stride=60,
                          need_images=True,
                          need_featurized_objects=False):
        full_batch_size = len(batch_task_inds)
        if full_batch_size > self.max_batch_size:
            raise ValueError(
                f"Cannot simulate more than {self.max_batch_size} runs at once"
                + "({full_batch_size} provided)")

        if not (isinstance(batch_actions, np.ndarray) and
                batch_actions.dtype == np.float32):
            batch_actions = np.array(batch_actions, dtype=np.float32)

        if not (isinstance(batch_task_inds, np.ndarray) and
                batch_task_inds.dtype == np.long):
            batch_task_inds = np.array(batch_task_inds, dtype=np.long)

        self.action_array[:full_batch_size] = batch_actions
        self.task_ind_array[:full_batch_size] = batch_task_inds

        mini_batch_size = math.ceil(full_batch_size / self.num_workers)
        if need_featurized_objects:
            self.feature_array[:] = 0.0
        for i, batch_start in enumerate(
                range(0, full_batch_size, mini_batch_size)):
            current_batch_size = min(mini_batch_size,
                                     full_batch_size - batch_start)
            message = (batch_start, current_batch_size, stride, need_images,
                       need_featurized_objects)
            self.pipes[i].send(message)

        for i, _ in enumerate(range(0, full_batch_size, mini_batch_size)):
            self.pipes[i].recv()

        if need_images:
            imgs = self.img_array[:full_batch_size]
        else:
            imgs = None

        if need_featurized_objects:
            features = self.feature_array[:full_batch_size]
        else:
            features = None

        if need_featurized_objects or need_images:
            masks = self.mask_array[:full_batch_size]
        else:
            masks = None

        return self.sim_status_array[:full_batch_size], imgs, features, masks

    def close(self):
        for worker in self.workers:
            worker.terminate()

    def __del__(self):
        self.close()


if __name__ == "__main__":

    import time, timeit
    from parallel_simulator import dump_img
    import random

    tier = ["ball", "two_balls"][0]

    train_id, dev_id, test_id = phyre.get_fold(tier + "_within_template", 7)
    train_id = list(train_id)
    random.seed(42)
    random.shuffle(train_id)
    train_id = train_id[:10]

    simulator = phyre.initialize_simulator(train_id, tier)
    cache = phyre.get_default_100k_cache(tier)

    task_ind = 1
    action = [0.8, 0.8, 0.1]
    solution = cache.load_simulation_states(train_id[1]).argmax()
    solution = cache.action_array[solution]
    print(solution, "solution")

    status, imgs = simulator.simulate_single(1, action)
    print(status, "status")

    print(
        "single sim",
        timeit.timeit(lambda: simulator.simulate_single(1, action), number=10) /
        10)

    psimulator = ParallelPhyreSimulator(train_id, tier, 78, 17, 1024)

    #imgs, masks = psimulalor.simulate_parallel([1,2,3], np.array([cache.action_array[solution], [0.8, 0.1, 0.1], [0.5, 0.99, 1.0]]))

    tasks = [1] * 1024

    actions = [action for i in range(1024)]
    actions[19] = list(solution)
    #tasks[19] = 5

    status, imgs, features, masks = psimulator.simulate_parallel(
        tasks, actions, need_images=True, need_featurized_objects=True)

    tasks = np.array(tasks, dtype=np.long)
    actions = np.array(actions, dtype=np.float32)
    #print('TIME FOR 64 SIMS ', timeit.timeit(lambda : psimulator.simulate_parallel(tasks, actions), number=10)/10)

    print("images", imgs.shape)
    # for i in range(64):
    # dump_img(imgs[i][16], "p" + str(i))

    # for i, mask in enumerate(masks):
    # print(i, list(mask), status[i])
    psimulator.close()
