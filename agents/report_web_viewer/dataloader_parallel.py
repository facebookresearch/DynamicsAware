import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import phyre
import numpy as np
import matplotlib.pyplot as plt
import os



INF = 10 ** 20


def dump_img(observation, name):
    img = phyre.observations_to_uint8_rgb(observation)
    name = "rollout/" + str(name) + ".jpg"
    plt.imsave(name, img)
    print(name, os.getcwd())

class PhyreParallel(Dataset):
    def __init__(self, simulator):
        self._sim = simulator
        self._len = INF

    def __getitem__(self, args):
        # IF args is an instance of JunkKeys this means you are sampling without feeding task/action pairs to 
        # sampler via Sampler.feed_task_action
        task_idx, action = args
        status, imgs = self._sim.simulate_single(task_idx, action)
        return torch.LongTensor(imgs)

    def __len__(self):
        return self._len

class JunkKeys:
    pass

class SimulationSampler(Sampler):
    def __init__(self):
        """Sampler may need to be primed by supplying a dummy  batch of task/action pairs before passing to torch.data.Dataloader"""
        self.keys = JunkKeys()

    def __len__(self):
        return INF
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.keys is None:
            raise ValueError("Cannot sample from simulator as no task/action pairs have been provided")
        keys = self.keys
        self.keys = None
        return keys
    
    def feed_task_action(self, task_idxs, actions):
        """Feed a list of task indexes and a list/tensor of actions for sampler to supply to data loader"""
        assert len(task_idxs) == len(actions)
        self.keys = zip(task_idxs, actions)


if __name__ == "__main__":
  

    train_id, dev_id, test_id  = phyre.get_fold("ball_cross_template", 0)

    train_id = train_id[:5]

    simulator = phyre.initialize_simulator(train_id, "ball")


    dset = PhyreParallel(simulator)

    sampler = SimulationSampler()
    sampler.feed_task_action([0], [np.array([0.8, 0.8, 0.05])])

    dloader = iter(DataLoader(dset, batch_sampler=sampler))

    sampler.feed_task_action([1, 2, 1], np.array([[0.8, 0.8, 0.05], [0.8, 0.8, 0.05], [0.8, 0.8, 0.1]]))
    imgs = next(dloader)
    print(imgs.shape)
    dump_img(imgs[0][0], "b0")
    dump_img(imgs[1][0], "b1")
    dump_img(imgs[2][0], "b2")

    print(imgs.shape)

