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
"""This library contains actual implementation of the DQN agent."""
from typing import Optional, Sequence, Tuple
import glob
import logging
import os
import time
import weakref
import pathlib

import numpy as np
import torch
from torch import nn

import nets
import phyre
from psim import ParallelPhyreSimulator, standardize_img_rollout_shape, standardize_features_shape
import rollout_dist

AUX_LOSS_EVAL_TASKS = 30
AUX_EVAL_ACTIONS_PER_TASK = 8
AUCCESS_EVAL_TASKS = 200
XE_EVAL_SIZE = 10000
MAX_LEN = 17
TaskIds = Sequence[str]
NeuralModel = torch.nn.Module
TrainData = Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                  phyre.ActionSimulator, torch.Tensor]


def create_balanced_eval_set(cache: phyre.SimulationCache, task_ids: TaskIds,
                             size: int, tier: str) -> TrainData:
    """Prepares balanced eval set to run through a network.

    Selects (size // 2) positive (task, action) pairs and (size // 2)
    negative pairs and represents them in a compact formaer

    Returns a tuple
        (task_indices, is_solved, selected_actions, simulator, observations).

        Tensors task_indices, is_solved, selected_actions, observations, all
        have lengths size and correspond to some (task, action) pair.
        For any i the following is true:
            is_solved[i] is true iff selected_actions[i] solves task
            task_ids[task_indices[i]].
    """
    task_ids = tuple(task_ids)
    data = cache.get_sample(task_ids)
    actions = data['actions']
    simulation_statuses = data['simulation_statuses']

    flat_statuses = simulation_statuses.reshape(-1)
    [positive_indices
    ] = (flat_statuses == int(phyre.SimulationStatus.SOLVED)).nonzero()
    [negative_indices
    ] = (flat_statuses == int(phyre.SimulationStatus.NOT_SOLVED)).nonzero()

    half_size = size // 2
    rng = np.random.RandomState(42)
    positive_indices = rng.choice(positive_indices, half_size)
    negative_indices = rng.choice(negative_indices, half_size)

    all_indices = np.concatenate([positive_indices, negative_indices])
    selected_actions = torch.FloatTensor(actions[all_indices % len(actions)])
    is_solved = torch.LongTensor(flat_statuses[all_indices].astype('int')) > 0
    task_indices = torch.LongTensor(all_indices // len(actions))

    simulator = phyre.initialize_simulator(task_ids, tier)
    observations = torch.LongTensor(simulator.initial_scenes)
    return task_indices, is_solved, selected_actions, simulator, observations


def create_metric_eval_set(tasks_ids, action_tier, cache, num_tasks,
                           actions_per_task, simulator):
    rng = np.random.RandomState(42)
    task_ids = rng.choice(tasks_ids, num_tasks)
    if simulator is None:
        simulator = phyre.initialize_simulator(task_ids, action_tier)
    n_actions = cache.get_sample(task_ids[:1])["simulation_statuses"].size
    actions = []
    task_inds = []
    all_action_indices = np.arange(n_actions)

    actions_per_task = (actions_per_task // 2) * 4
    for task_id in task_ids:
        cache_for_task = cache.get_sample([task_id])
        statuses = cache_for_task["simulation_statuses"].squeeze()
        positive_actions = all_action_indices[statuses == 1]
        negative_actions = all_action_indices[statuses == -1]

        if positive_actions.size:
            positive_actions = rng.choice(positive_actions,
                                          size=actions_per_task // 2)
            negative_actions = rng.choice(negative_actions,
                                          size=actions_per_task // 2)
            actions.append(cache_for_task["actions"][positive_actions])
            actions.append(cache_for_task["actions"][negative_actions])
        else:
            negative_actions = rng.choice(negative_actions,
                                          size=actions_per_task)
            actions.append(cache_for_task["actions"][negative_actions])

        task_inds.append([simulator.task_ids.index(task_id)] * actions_per_task)

    actions = np.concatenate(actions)
    task_inds = np.concatenate(task_inds)
    observations = simulator.initial_scenes[task_inds]
    rollouts = [
        simulator.simulate_action(task_ind, action, need_images=False, need_featurized_objects=True).featurized_objects.features
        for task_ind, action in zip(task_inds, actions)
    ]
    rollouts, masks = standardize_features_shape(rollouts, MAX_LEN)
    observations = torch.from_numpy(observations)
    actions = torch.from_numpy(actions)
    return observations, actions, task_inds, np.unique(
        task_inds), torch.from_numpy(rollouts), torch.from_numpy(masks)


def compute_loss_on_eval_batch(model, n_samples_per_task, aux_loss_hp, observations, actions,
                               task_inds, unique_inds, rollouts, masks):
    with torch.no_grad():
        _, embeddings = model(observations, actions, get_embeddings=True)
    return sample_distance_loss(model, embeddings, task_inds, rollouts, masks,
                                unique_inds, n_samples_per_task, aux_loss_hp).item()


def compact_simulation_data_to_trainset(action_tier_name: str,
                                        actions: np.ndarray,
                                        simulation_statuses: Sequence[int],
                                        task_ids: TaskIds) -> TrainData:
    """Converts result of SimulationCache.get_data() to pytorch tensors.

    The format of the output is the same as in create_balanced_eval_set in addition to two
    extra return dictionaries at the end:
    positive_in_task, where positive_in_task[i] contains all flat training dataset that are in the i'th
    task and have a positive label and negative in task is smae but for negative labels. The index i is
    a sequential index of the task as contained in the data set. i starts at zero and is not the same as the 
    task index in the phyre API
    """
    invalid = int(phyre.SimulationStatus.INVALID_INPUT)
    solved = int(phyre.SimulationStatus.SOLVED)

    task_indices = np.repeat(np.arange(len(task_ids)).reshape((-1, 1)),
                             actions.shape[0],
                             axis=1).reshape(-1)
    action_indices = np.repeat(np.arange(actions.shape[0]).reshape((1, -1)),
                               len(task_ids),
                               axis=0).reshape(-1)
    simulation_statuses = simulation_statuses.reshape(-1)

    good_statuses = simulation_statuses != invalid
    is_solved = torch.LongTensor(
        simulation_statuses[good_statuses].astype('uint8')) == solved
    action_indices = action_indices[good_statuses]
    actions = torch.FloatTensor(actions[action_indices])
    task_indices = torch.LongTensor(task_indices[good_statuses])

    simulator = phyre.initialize_simulator(task_ids, action_tier_name)
    observations = torch.LongTensor(simulator.initial_scenes)

    positive_in_task = {}
    negative_in_task = {}
    prev_end = 0
    n = is_solved.numel()
    for i in range(len(task_ids)):
        lower_bound = prev_end
        for j_batch in range(prev_end, n, 4096):
            if not (task_indices[j_batch:j_batch +
                                 4096] == i).all() or j_batch >= (n - 1 - 4096):
                for j in range(j_batch, min(j_batch + 4096, n)):
                    if task_indices[j] != i or j == (n - 1):
                        upper_bound = j - 1
                        prev_end = j
                        break_out = True
                        break
                break

        is_solved_in_task = is_solved[lower_bound:upper_bound + 1]
        in_task_indices = torch.arange(lower_bound,
                                       upper_bound + 1,
                                       dtype=torch.long)
        positive_in_task[i] = in_task_indices[is_solved_in_task]
        negative_in_task[i] = in_task_indices[~is_solved_in_task]
    return task_indices, is_solved, actions, simulator, observations, positive_in_task, negative_in_task


def build_model(network_type: str, **kwargs) -> NeuralModel:
    """Builds a DQN network by name."""
    if network_type in ('resnet18', 'resnet50'):
        model = nets.ResNetFilmAction(
            backbone=network_type,
            action_size=kwargs['action_space_dim'],
            fusion_place=kwargs['fusion_place'],
            action_hidden_size=kwargs['action_hidden_size'],
            action_layers=kwargs['action_layers'],
            embedding_dim=kwargs.get('embedding_dim'),
            embeddor_type=kwargs.get('embeddor_type'),
            repr_merging_method=kwargs.get('repr_merging_method'),
            n_regressor_outputs=kwargs.get('n_regressor_outputs'))
    elif network_type == 'simple':
        model = nets.SimpleNetWithAction(kwargs['action_space_dim'])
    else:
        raise ValueError('Unknown network type: %s' % network_type)
    return model


def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    known_checkpoints = sorted(glob.glob(os.path.join(output_dir, 'ckpt.*')))
    if known_checkpoints:
        return known_checkpoints[-1]
    else:
        return None


def load_agent_from_folder(agent_folder: str) -> NeuralModel:
    last_checkpoint = get_latest_checkpoint(agent_folder)
    assert last_checkpoint is not None, agent_folder
    logging.info('Loading a model from: %s', last_checkpoint)
    last_checkpoint = torch.load(last_checkpoint)
    model = build_model(**last_checkpoint['model_kwargs'])
    try:
        model.load_state_dict(last_checkpoint['model'])
    except RuntimeError:
        model = nn.DataParallel(model)
        model.load_state_dict(last_checkpoint['model'])
        model = model.module
    model.to(nets.DEVICE)
    return model


def finetune(model: NeuralModel, data: Sequence[Tuple[int,
                                                      phyre.SimulationStatus,
                                                      Sequence[float]]],
             simulator: phyre.ActionSimulator, learning_rate: float,
             num_updates: int) -> None:
    """Finetunes a model on a small new batch of data.

    Args:
        model: DQN network, e.g., built with build_model().
        data: a list of tuples (task_index, status, action).
        learning_rate: learning rate for Adam.
        num_updates: number updates to do. All data is used for every update.
    """

    data = [x for x in data if not x[1].is_invalid()]
    if not data:
        return
    task_indices, statuses, actions = zip(*data)
    if len(set(task_indices)) == 1:
        observations = np.expand_dims(simulator.initial_scenes[task_indices[0]],
                                      0)
    else:
        observations = simulator.initial_scenes[task_indices]

    device = model.module.device if isinstance(
        model, nn.DataParallel) else model.device
    is_solved = torch.tensor(statuses, device=device) > 0
    observations = torch.tensor(observations, device=device)
    actions = torch.tensor(actions, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(num_updates):
        optimizer.zero_grad()
        model.ce_loss(model(observations, actions), is_solved).backward()
        optimizer.step()


def refine_actions(model, actions, single_observarion, learning_rate,
                   num_updates, batch_size, refine_loss):
    device = model.module.device if isinstance(
        model, nn.DataParallel) else model.device
    observations = torch.tensor(single_observarion, device=device).unsqueeze(0)
    actions = torch.tensor(actions)

    refined_actions = []
    model.eval()
    preprocessed = model.preprocess(observations)
    preprocessed = {k: v.detach() for k, v in preprocessed.items()}
    for start in range(0, len(actions), batch_size):
        action_batch = actions[start:][:batch_size].to(device)
        action_batch = torch.nn.Parameter(action_batch)
        optimizer = torch.optim.Adam([action_batch], lr=learning_rate)
        losses = []
        for _ in range(num_updates):
            optimizer.zero_grad()
            logits = model(None, action_batch, preprocessed=preprocessed)
            if refine_loss == 'ce':
                loss = model.ce_loss(logits, actions.new_ones(len(logits)))
            elif refine_loss == 'linear':
                loss = -logits.sum()
            else:
                raise ValueError(f'Unknown loss: {refine_loss}')
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        action_batch = torch.clamp_(action_batch.data, 0, 1)
        refined_actions.append(action_batch.cpu().numpy())
    refined_actions = np.concatenate(refined_actions, 0).tolist()
    return refined_actions


def sample_distance_loss(model, embeddings, task_indices, rollouts, masks,
                         unique_task_inds, n_samples_per_task, loss_hyperparams):
    loss = 0
    n_pairs = []
    batch_size = embeddings.shape[0]
    for i in range(0, batch_size, n_samples_per_task):
        embeddings_in_task = embeddings[i: i + n_samples_per_task]
        rollouts_in_task = rollouts[i: i + n_samples_per_task]
        masks_in_task = masks[i: i + n_samples_per_task]


        loss_mat = comparative_loss_single_batch(model, embeddings_in_task,
                                                 rollouts_in_task,
                                                 masks_in_task,
                                                 loss_hyperparams)
        loss += loss_mat.sum()
        n_pairs.append(loss_mat.numel())
    loss = loss / sum(n_pairs)
    return loss


def comparative_loss_single_batch(model,
                                  embeddings,
                                  rollouts,
                                  masks,
                                  loss_hyperparams,
                                  distances=None):
    """Hyperparams are "regression_type", "repr_merging_method", "distance_cutoff"""
    hp = loss_hyperparams
    if isinstance(model, nn.DataParallel):
        model = model.module
    n_classes = model.n_regressor_outputs
    if distances is None:
        distances = rollout_dist.measure_all_distances(rollouts,
                                                       masks,
                                                       hp["distance_cutoff"],
                                                       cross_task=False,
                                                       time_slice=hp.get("time_slice", 0))

    distances = distances.to(nets.DEVICE).float()
    n = embeddings.shape[0]

    left = embeddings.view(n, 1, -1).expand(n, n, -1).reshape(n * n, -1)
    right = embeddings.view(1, n, -1).expand(n, n, -1).reshape(n * n, -1)

    predictions = model.compute_regressed_distances(left, right)

    distances = distances.view(-1).to(nets.DEVICE)

    if hp["regression_type"] == "binned":
        distances = torch.round(distances * (n_classes - 1)).long()

        loss_mat = nn.functional.cross_entropy(predictions,
                                               distances,
                                               reduction='none')
    elif hp["regression_type"] == "mse":
        loss_mat = nn.functional.mse_loss(predictions.view(-1),
                                          distances.float(),
                                          reduction='none')
    return loss_mat


def train(output_dir,
          action_tier_name,
          task_ids,
          cache,
          train_batch_size,
          learning_rate,
          max_train_actions,
          updates,
          negative_sampling_prob,
          save_checkpoints_every,
          fusion_place,
          network_type,
          balance_classes,
          num_auccess_actions,
          eval_every,
          action_layers,
          action_hidden_size,
          cosine_scheduler,
          n_samples_per_task,
          use_sample_distance_aux_loss,
          aux_loss_hyperparams,
          checkpoint_dir,
          tensorboard_dir="",
          dev_tasks_ids=None,
          debug=False,
          **excess_kwargs):

    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        logging.info("Tensorboard dir :" + tensorboard_dir)
        writer = SummaryWriter(tensorboard_dir)

    logging.info('Preprocessing train data')
    if debug:
        logging.warning("Debugging ON")
        logging.warning("Subsampling dataset")
        task_ids = task_ids[:10]
        if dev_tasks_ids:
            dev_tasks_ids = dev_tasks_ids[:10]
    training_data = cache.get_sample(task_ids, max_train_actions)
    task_indices, is_solved, actions, simulator, observations, positive_in_task, negative_in_task = (
        compact_simulation_data_to_trainset(action_tier_name, **training_data))
    logging.info('Creating eval subset from train')
    eval_train = create_balanced_eval_set(cache, simulator.task_ids,
                                          XE_EVAL_SIZE, action_tier_name)
    if dev_tasks_ids is not None:
        logging.info('Creating eval subset from dev')
        eval_dev = create_balanced_eval_set(cache, dev_tasks_ids, XE_EVAL_SIZE,
                                            action_tier_name)
    else:
        eval_dev = None

    aux_loss_eval = None
    aux_loss_eval_dev = None
    if use_sample_distance_aux_loss:
        logging.info("Creating eval set for auxiliary loss from train")
        aux_loss_eval = create_metric_eval_set(task_ids, action_tier_name,
                                               cache, AUX_LOSS_EVAL_TASKS,
                                               AUX_EVAL_ACTIONS_PER_TASK,
                                               simulator)
        if dev_tasks_ids is not None:
            logging.info("Creating eval set for auxiliary loss from dev")
            aux_loss_eval_dev = create_metric_eval_set(
                dev_tasks_ids,
                action_tier_name,
                cache,
                AUX_LOSS_EVAL_TASKS,
                AUX_EVAL_ACTIONS_PER_TASK,
                simulator=None)
    logging.info('Tran set: size=%d, positive_ratio=%.2f%%', len(is_solved),
                 is_solved.float().mean().item() * 100)

    assert not balance_classes or (negative_sampling_prob == 1), (
        balance_classes, negative_sampling_prob)

    device = nets.DEVICE
    model_kwargs = dict(network_type=network_type,
                        action_space_dim=simulator.action_space_dim,
                        fusion_place=fusion_place,
                        action_hidden_size=action_hidden_size,
                        action_layers=action_layers)
    if use_sample_distance_aux_loss:
        n_regressor_outputs = 1 if aux_loss_hyperparams[
            "regression_type"] == "mse" else aux_loss_hyperparams[
                "n_regression_bins"]
        model_kwargs.update(
            dict(
                repr_merging_method=aux_loss_hyperparams["repr_merging_method"],
                embedding_dim=aux_loss_hyperparams["embedding_dim"],
                embeddor_type=aux_loss_hyperparams["embeddor_type"],
                n_regressor_outputs=n_regressor_outputs))
    model = build_model(**model_kwargs)
    model = nn.DataParallel(model)
    model.train()
    model.to(device)
    logging.info(model)
    logging.info(f"Model will use {torch.cuda.device_count()} GPU's to train")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=updates)
    else:
        scheduler = None
    logging.info('Starting actual training for %d updates', updates)

    rng = np.random.RandomState(42)

    def task_balanced_sampler():
        n_tasks_per_batch = train_batch_size // n_samples_per_task
        n_tasks_total = len(simulator.initial_scenes)
        n_samples_per_task_label = n_samples_per_task // 2
        while True:
            batch_task_indices = rng.choice(n_tasks_total,
                                            size=n_tasks_per_batch,
                                            replace=debug)
            indices = []
            for task_idx in batch_task_indices:
                indices.append(
                    rng.choice(negative_in_task[task_idx],
                               n_samples_per_task_label))
                try:
                    indices.append(
                        rng.choice(positive_in_task[task_idx],
                                   n_samples_per_task_label))
                except ValueError:  # No solutions for given template in dataset hence add more negative samples
                    indices.append(
                        rng.choice(negative_in_task[task_idx],
                                   n_samples_per_task_label))
            indices = np.concatenate(indices)
            yield indices, batch_task_indices

    def train_indices_sampler():
        indices = np.arange(len(is_solved))
        if balance_classes:
            solved_mask = is_solved.numpy() > 0
            positive_indices = indices[solved_mask]
            negative_indices = indices[~solved_mask]
            positive_size = train_batch_size // 2
            while True:
                positives = rng.choice(positive_indices, size=positive_size)
                negatives = rng.choice(negative_indices,
                                       size=train_batch_size - positive_size)
                positive_size = train_batch_size - positive_size
                yield np.stack((positives, negatives), axis=1).reshape(-1), None
        elif negative_sampling_prob < 1:
            probs = (is_solved.numpy() * (1.0 - negative_sampling_prob) +
                     negative_sampling_prob)
            probs /= probs.sum()
            while True:
                yield rng.choice(indices, size=train_batch_size, p=probs), None
        else:
            while True:
                yield rng.choice(indices, size=train_batch_size), None

    last_checkpoint = get_latest_checkpoint(checkpoint_dir)
    batch_start = 0
    if last_checkpoint is not None:
        logging.info('Going to load from %s', last_checkpoint)
        last_checkpoint = torch.load(last_checkpoint)
        model.load_state_dict(last_checkpoint['model'])
        optimizer.load_state_dict(last_checkpoint['optim'])
        rng.set_state(last_checkpoint['rng'])
        batch_start = last_checkpoint['done_batches']
        if scheduler is not None:
            scheduler.load_state_dict(last_checkpoint['scheduler'])

    def print_eval_stats(batch_id):
        with torch.no_grad():
            logging.info('Start eval')
            eval_batch_size = train_batch_size * 2
            stats = {}
            stats['batch_id'] = batch_id + 1
            stats['train_loss'] = eval_loss(model, eval_train, eval_batch_size)
            if eval_dev:
                stats['dev_loss'] = eval_loss(model, eval_dev, eval_batch_size)
            if num_auccess_actions > 0:
                logging.info('Start AUCCESS eval')
                stats['train_auccess'] = _eval_and_score_actions(
                    cache, model, eval_train[3], num_auccess_actions,
                    eval_batch_size, eval_train[4])
                if eval_dev:
                    stats['dev_auccess'] = _eval_and_score_actions(
                        cache, model, eval_dev[3], num_auccess_actions,
                        eval_batch_size, eval_dev[4])
            if aux_loss_eval:
                logging.info("starting aux loss eval")
                stats["aux_loss"] = compute_loss_on_eval_batch(
                    model, n_samples_per_task, aux_loss_hyperparams, *aux_loss_eval)
            if aux_loss_eval_dev:
                logging.info("starting aux loss dev eval")
                stats["aux_loss_dev"] = compute_loss_on_eval_batch(
                    model, n_samples_per_task, aux_loss_hyperparams, *aux_loss_eval_dev)
            if tensorboard_dir:
                for stat in stats:
                    if stat == "batch_id":
                        continue
                    writer.add_scalar(stat, stats[stat], batch_id)

            logging.info('__log__: %s', stats)

    report_every = 5 if debug else 100
    logging.info('Report every %d; eval every %d', report_every, eval_every)
    if save_checkpoints_every > eval_every:
        save_checkpoints_every -= save_checkpoints_every % eval_every
    
    if batch_start == 0:
        print_eval_stats(0)

    losses = []
    aux_losses = []
    ce_losses = []
    last_time = time.time()
    observations = observations.to(device)
#    actions = actions.pin_memory()
 #   is_solved = is_solved.pin_memory()

    if use_sample_distance_aux_loss:
        num_workers = 5 if debug else 20
        parallel_simulator = ParallelPhyreSimulator(task_ids,
                                                    action_tier_name,
                                                    num_workers,
                                                    MAX_LEN,
                                                    train_batch_size,
                                                    requires_imgs=False)
        logging.info(f"Starting parallel simulator with {num_workers} workers")
        f = weakref.finalize(parallel_simulator, parallel_simulator.close)
    if n_samples_per_task == 0:
        assert not use_sample_distance_aux_loss
        sampler = train_indices_sampler()
    else:
        assert (n_samples_per_task % 2 == 0 and balance_classes)
        sampler = task_balanced_sampler()
    for batch_id, (batch_indices,
                   unique_task_indices) in enumerate(sampler,
                                                     start=batch_start):
        if batch_id >= updates:
            break
        if scheduler is not None:
            scheduler.step()
        model.train()
        batch_task_indices = task_indices[batch_indices]
        batch_observations = observations[batch_task_indices]
        batch_actions = actions[batch_indices]
        batch_is_solved = is_solved[batch_indices].to(device, non_blocking=True)


        if use_sample_distance_aux_loss:
            _, _, batch_rollouts, batch_masks = parallel_simulator.simulate_parallel(
                batch_task_indices,
                batch_actions,
                need_images=False,
                need_featurized_objects=True)
            batch_rollouts = torch.from_numpy(batch_rollouts)
            batch_masks = torch.from_numpy(batch_masks)

        batch_actions = batch_actions.to(device, non_blocking=True)

        if use_sample_distance_aux_loss:
            logits, embeddings = model(
                batch_observations,
                batch_actions,
                get_embeddings=use_sample_distance_aux_loss)
            aux_loss = sample_distance_loss(model, embeddings,
                                            batch_task_indices, batch_rollouts,
                                            batch_masks, unique_task_indices, n_samples_per_task,
                                            aux_loss_hyperparams)
        else:
            logits = model(batch_observations, batch_actions)
            aux_loss = torch.tensor([0.0]).to(nets.DEVICE)

        #TODO add tensorboard, detailed evaluator and parallel simulator
        optimizer.zero_grad()
        classification_loss = model.module.ce_loss(logits,
                                                   batch_is_solved).mean()
        loss = classification_loss + aux_loss * aux_loss_hyperparams["weight"]

        loss.backward()
        optimizer.step()
        losses.append(loss.mean().item())
        aux_losses.append(aux_loss.mean().item())
        ce_losses.append(classification_loss.mean().item())
        if save_checkpoints_every > 0:
            if (batch_id + 1) % save_checkpoints_every == 0 or (batch_id +
                                                                1) == updates:
                fpath = os.path.join(checkpoint_dir, 'ckpt.%08d' % (batch_id + 1))
                logging.info('Saving: %s', fpath)
                torch.save(
                    dict(
                        model_kwargs=model_kwargs,
                        model=model.state_dict(),
                        optim=optimizer.state_dict(),
                        done_batches=batch_id + 1,
                        rng=rng.get_state(),
                        scheduler=scheduler and scheduler.state_dict(),
                    ), fpath)
        if (batch_id + 1) % eval_every == 0:
            print_eval_stats(batch_id)
        if (batch_id + 1) % report_every == 0:
            speed = report_every / (time.time() - last_time)
            last_time = time.time()
            logging.debug(
                'Iter: %s, examples: %d, mean loss: %f, mean ce: %f, mean aux: %f,'
                'speed: %.1f batch/sec, lr: %f', batch_id + 1,
                (batch_id + 1) * train_batch_size,
                np.mean(losses[-report_every:]),
                np.mean(ce_losses[-report_every:]),
                np.mean(aux_losses[-report_every:]), speed, get_lr(optimizer))
    return model.module.cpu()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def eval_loss(model, data, batch_size):
    task_indices, is_solved, actions, _, observations = data
    losses = []
    device = model.module.device if isinstance(
        model, nn.DataParallel) else model.device
    ce_loss = model.module.ce_loss if isinstance(
        model, nn.DataParallel) else model.ce_loss

    observations = observations.to(device)
    with torch.no_grad():
        model.eval()
        for i in range(0, len(task_indices), batch_size):
            batch_indices = task_indices[i:i + batch_size]
            batch_task_indices = task_indices[batch_indices]
            batch_observations = observations[batch_task_indices]
            batch_actions = actions[batch_indices]
            batch_is_solved = is_solved[batch_indices]
            loss = ce_loss(model(batch_observations, batch_actions),
                           batch_is_solved)
            losses.append(loss.item() * len(batch_indices))
    return sum(losses) / len(task_indices)


def eval_actions(model, actions, batch_size, observations):
    scores = []
    with torch.no_grad():
        model.eval()
        preprocess = model.module.preprocess if isinstance(
            model, nn.DataParallel) else model.preprocess
        device = model.module.device if isinstance(
            model, nn.DataParallel) else model.device
        # TODO Find elegant solution for accessing attributes on DataParallel vs nn.Module
        preprocessed = preprocess(
            torch.LongTensor(observations).to(device).unsqueeze(0))
        for batch_start in range(0, len(actions), batch_size):
            batch_end = min(len(actions), batch_start + batch_size)
            batch_actions = torch.FloatTensor(
                actions[batch_start:batch_end]).to(device)
            batch_preprocessed = {
                "features":
                    preprocessed["features"].expand(batch_actions.shape[0], -1,
                                                    -1, -1)
            }
            batch_scores = model(None,
                                 batch_actions,
                                 preprocessed=batch_preprocessed)
            assert len(batch_scores) == len(batch_actions), (
                batch_actions.shape, batch_scores.shape)
            scores.append(batch_scores.cpu().numpy())
    return np.concatenate(scores)


def _eval_and_score_actions(cache, model, simulator, num_actions, batch_size,
                            observations):
    actions = cache.action_array[:num_actions]
    indices = np.random.RandomState(1).permutation(
        len(observations))[:AUCCESS_EVAL_TASKS]
    evaluator = phyre.Evaluator(
        [simulator.task_ids[index] for index in indices])
    for i, task_index in enumerate(indices):
        scores = eval_actions(model, actions, batch_size,
                              observations[task_index]).tolist()

        _, sorted_actions = zip(
            *sorted(zip(scores, actions), key=lambda x: (-x[0], tuple(x[1]))))
        for action in sorted_actions:
            if (evaluator.get_attempts_for_task(i) >= phyre.MAX_TEST_ATTEMPTS):
                break
            status = simulator.simulate_action(task_index,
                                               action,
                                               need_images=False).status
            evaluator.maybe_log_attempt(i, status)
    return evaluator.get_aucess()
