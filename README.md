# Evolution Strategies

This is an adaptation of [atgambardella's PyTorch implementation](https://github.com/atgambardella/pytorch-es) of [Evolution Strategies](https://arxiv.org/abs/1703.03864). I have updated it to be compatible with the current version of PyTorch, along with some other changes. The major ones are as follows:

* Models are now imported from [RLlib](https://github.com/ray-project/ray/tree/master/rllib/examples)'s `RLModuleSpec`. This is because providing a genetic baseline for RL problems is the primary purpose of this repository.

* Action selection and input preprocessing have been moved from the train loop to the agent class. Dictionary and Repeated observation spaces are now supported.

* PyTorch syntax has been updated to remove deprecated features and restore core functionality to the repository.

* As Universe has been deprecated, we no sadly longer support it.

# Tips *(from original repo)*

* If you increase the batch size, `n`, you should increase the learning rate as well.

* Feel free to stop training when you see that the unperturbed model is consistently solving the environment, even if the perturbed models are not.

* During training you probably want to look at the rank of the unperturbed model within the population of perturbed models. Ideally some perturbation is performing better than your unperturbed model (if this doesn't happen, you probably won't learn anything useful). This requires 1 extra rollout per gradient step, but as this rollout can be computed in parallel with the training rollouts, this does not add to training time. It does, however, give us access to one less CPU core.

* Sigma is a tricky hyperparameter to get right -- higher values of sigma will correspond to less variance in the gradient estimate, but will be more biased. At the same time, sigma is controlling the variance of our perturbations, so if we need a more varied population, it should be increased. It might be possible to adaptively change sigma based on the rank of the unperturbed model mentioned in the tip above. I tried a few simple heuristics based on this and found no significant performance increase, but it [might be possible to do this more intelligently](http://www.inference.vc/evolution-strategies-variational-optimisation-and-natural-es-2/).

* I found, as OpenAI did in their paper, that performance on Atari increased as I increased the size of the neural net.

# Your code is making my computer slow help

Short answer: decrease the batch size to the number of cores in your computer, and decrease the learning rate as well. This will most likely hurt the performance of the algorithm.

Long answer: If you want large batch sizes while also keeping the number of spawned threads down, I have provided an old version in the `slow_version` branch which allows you to do multiple rollouts per thread, per gradient step. This code is not supported, however, and it is not recommended that you use it.

# Contributions

Please feel free to make Github issues or send pull requests.

# License

MIT
