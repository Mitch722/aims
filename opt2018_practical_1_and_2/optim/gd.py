import torch
import random
import math

from optim.base import Optimizer, Variable, HParams
from utils import assert_true


class HParamsGD(HParams):
    required = ('n_samples', 'n_features', 'n_classes', 'fix_lr', 'init_lr')
    defaults = {'verbose': 1, 'mu': 0, 'temp': 1}

    def __init__(self, **kwargs):
        super(HParamsGD, self).__init__(kwargs)


class HParamsSGD(HParams):
    required = ('n_samples', 'batch_size', 'n_features', 'n_classes', 'fix_lr', 'init_lr')
    defaults = {'verbose': 1, 'mu': 0, 'temp': 1}

    def __init__(self, **kwargs):
        super(HParamsSGD, self).__init__(kwargs)


class VariablesGD(Variable):
    def __init__(self, hparams):
        super(VariablesGD, self).__init__(hparams)

    def init(self):
        # TODO: Compute size
        size = (self.hparams.n_features, 1)

        assert isinstance(size, tuple)
        # Will contain the weights
        self.w = torch.rand(size, requires_grad=True)
        # Will contain the current step size
        self.lr = torch.tensor(1., requires_grad=False)
        # Will contain the current iteration
        self.it = torch.tensor(1., requires_grad=False)


class GD(Optimizer):
    def __init__(self, hparams):
        super(GD, self).__init__(hparams)

    def create_vars(self):
        return VariablesGD(self.hparams)

    def get_sampler(self, dataset):
        # this sampler yields the entire dataset
        all_indices = list(range(len(dataset)))
        yield (-1,) + dataset[all_indices]

    def get_sampler_len(self, dataset):
        return 1

    def _step(self, oracle_info):
        assert_true("dw" in oracle_info,
                    "The oracle_info should contain the gradient as dw")
        assert_true(oracle_info["dw"].size() == self.variables.w.size(),
                    "The gradient should be the same size as w")
        
        # TODO: Update self.variables.w, self.variables.lr and self.variables.it
        w = self.variables.w
        lr = self.variables.lr
        it = self.variables.it
        dw = oracle_info['dw']

        # compute learning rate
        if self.hparams.fix_lr:
            lr.fill_(self.hparams.init_lr)
        else:
            lr.fill_(self.hparams.init_lr / it.sqrt())
        it += 1

        # update
        w -= lr * dw


class SGD(GD):
    def __init__(self, hparams):
        super(SGD, self).__init__(hparams)

    def get_sampler(self, dataset):
        # this sampler yields random mini-batches of size batch_size
        # (except for the last mini-batch which may be of smaller size)
        dataset_size = len(dataset)
        batch_size = self.hparams.batch_size
        all_indices = list(range(dataset_size))
        random.shuffle(all_indices)
        batch_indices = []
        for i in range(int(math.ceil(dataset_size / float(batch_size)))):
            batch_indices.append(all_indices[i * batch_size: i * batch_size + batch_size])
        for batch_index in batch_indices:
            yield (-1,) + dataset[torch.LongTensor(batch_index)]

    def get_sampler_len(self, dataset):
        batch_size = self.hparams.batch_size
        return int(math.ceil(len(dataset) / float(batch_size)))

    def _step(self, oracle_info):
        assert_true("dw" in oracle_info,
                    "The oracle_info should contain the gradient as dw")
        assert_true(oracle_info["dw"].size() == self.variables.w.size(),
                    "The gradient should be the same size as w")
        
        # TODO: Update self.variables.w, self.variables.lr and self.variables.it
