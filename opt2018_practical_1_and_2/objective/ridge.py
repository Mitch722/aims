import torch

from objective.base import Objective
from utils import assert_true


class Ridge(Objective):
    def _validate_inputs(self, w, x, y):
        assert_true(w.dim() == 2,
                    "Input w should be 2D")
        assert_true(w.size(1) == 1,
                    "Ridge regression can only perform regression (size 1 output)")
        assert_true(x.dim() == 2,
                    "Input datapoint should be 2D")
        assert_true(y.dim() == 1,
                    "Input label should be 1D")
        assert_true(x.size(0) == y.size(0),
                    "Input datapoint and label should contain the same number of samples")


class Ridge_ClosedForm(Ridge):
    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute mean squared error
        err = y - torch.mm(x, w).squeeze(-1)

        error = torch.mean(err ** 2)
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute objective value
        n, d = x.size()

        obj = self.task_error(w, x, y) + 0.5 * self.hparams.mu * torch.mm(w.t(), w)
        # TODO: compute close form solution
        mu_eye = (0.5 * self.hparams.mu) * torch.eye(d)

        x_xt = torch.mm(x.t(), x)
        mat_inv = x_xt/n + mu_eye
        inv_mat = torch.inverse(mat_inv)
        xy_mat = torch.mv(x.t(), y) / n

        sol = torch.mv(inv_mat, xy_mat).unsqueeze(-1)

        return {'obj': obj, 'sol': sol}


class Ridge_Gradient(Ridge):
    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute mean squared error
        error = None
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute objective value
        obj = None
        # TODO: compute gradient
        dw = None
        return {'obj': obj, 'dw': dw}
