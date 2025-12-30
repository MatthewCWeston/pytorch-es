from __future__ import absolute_import, division, print_function
import gymnasium as gym

import os
import argparse

import torch

from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule

from model import ES_fromRLlib
from train import train_loop

parser = argparse.ArgumentParser(description='ES')

parser.add_argument('--env-name', default='PongDeterministic-v4',
                    metavar='ENV', help='environment')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr-decay', type=float, default=1, metavar='LRD',
                    help='learning rate decay')
parser.add_argument('--sigma', type=float, default=0.05, metavar='SD',
                    help='noise standard deviation')
parser.add_argument('--useAdam', action='store_true',
                    help='bool to determine if to use adam optimizer')
parser.add_argument('--n', type=int, default=40, metavar='N',
                    help='batch size, must be even')

parser.add_argument('--max-episode-length', type=int, default=100000,
                    metavar='MEL', help='maximum length of an episode')
parser.add_argument('--max-gradient-updates', type=int, default=100000,
                    metavar='MGU', help='maximum number of epochs, when multiplied by batch size')

parser.add_argument('--restore', default='', metavar='RES',
                    help='checkpoint from which to restore')
parser.add_argument('--chkpt_dir', default=None, metavar='CPT',
                    help='checkpoint directory, if any')

parser.add_argument('--variable-ep-len', action='store_true',
                    help="Change max episode length during training")
parser.add_argument('--silent', action='store_true',
                    help='Silence print statements during training')

if __name__ == '__main__':
    args = parser.parse_args()
    assert args.n % 2 == 0
    env = gym.make(args.env_name)

    if (args.chkpt_dir is not None):
        if not os.path.exists(args.chkpt_dir):
            os.makedirs(args.chkpt_dir)

    # Instantiate RLModuleSpec and build the RLModule
    module = RLModuleSpec(
        module_class=DefaultPPOTorchRLModule,
        observation_space=env.observation_space,
        action_space=env.action_space,
        model_config={
            "fcnet_hiddens": [64, 64], # Sets the encoder arch. The head's a single FC layer
            "vf_share_layers": True,
        },
    ).build()
            
    synced_model = ES_fromRLlib(module)
    
    for param in synced_model.parameters():
        param.requires_grad = False
      
    if args.restore:
        state_dict = torch.load(args.restore)
        synced_model.load_state_dict(state_dict)

    train_loop(args, synced_model, env, args.chkpt_dir)
