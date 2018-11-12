import torch

from objective.base import Objective
from utils import assert_true


class Lasso(Objective):
    def _validate_inputs(self, w, x, y):
        assert_true(w.dim() == 2,
                    "Input w should be 2D")
        assert_true(w.size(1) == 1,
                    "Lasso regression can only perform regression (size 1 output)")
        assert_true(x.dim() == 2,
                    "Input datapoint should be 2D")
        assert_true(y.dim() == 1,
                    "Input label should be 1D")
        assert_true(x.size(0) == y.size(0),
                    "Input datapoint and label should contain the same number of samples")


class Lasso_subGradient(Lasso):
    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute mean squared error
        err = y - torch.mm(x, w).squeeze(-1)

        error = torch.mean(err ** 2)
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute objective value
        obj = self.task_error(w, x, y) + 0.5 * self.hparams.mu * torch.sum(torch.abs(w))
        # TODO: compute subgradient
        du = torch.zeros(w.size())

        du[w > 0] = 1
        du[w < 0] = -1

        inter_mat = torch.mm(x, w).squeeze(-1) - y

        dw = 2/self.hparams.n_samples * torch.mv(x.t(), inter_mat)
        dw = dw.unsqueeze(-1)
        dw += 0.5 * self.hparams.mu * du

        return {'obj': obj, 'dw': dw}


class SmoothedLasso_Gradient(Lasso):
    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute mean squared error
        err = y - torch.mm(x, w).squeeze(-1)

        error = torch.mean(err ** 2)
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute objective value

        obj = None
        # TODO: compute gradient
        dw = None
        return {'obj': obj, 'dw': dw}
