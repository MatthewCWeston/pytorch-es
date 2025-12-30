from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.core.columns import Columns

def obs_to_tensor(obs):
    if (type(obs) is dict):
        r = {}
        for k, v in obs.items():
            r[k] = obs_to_tensor(v)
        return r
    else:
        return torch.tensor(obs).unsqueeze(0).float()

class ES_fromRLlib(torch.nn.Module):

    def __init__(self, module):
        super(ES_fromRLlib, self).__init__()
        self.module = module
        self.train()

    def forward(self, inputs):
        inputs = obs_to_tensor(inputs)
        inputs = {Columns.OBS: inputs}
        x = self.module.forward_train(inputs)
        x = x[Columns.ACTION_DIST_INPUTS]
        return x

    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.data.numpy().flatten().shape[0]
        return count

    def es_params(self):
        """
        The params that should be trained by ES (all of them)
        """
        return [(k, v) for k, v in zip(self.state_dict().keys(),
                                       self.state_dict().values()) if ('.vf.' not in k and 'const' not in k)]
