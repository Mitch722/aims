import torch

from objective.base import Objective
from utils import assert_true


class Logistic_Gradient(Objective):
    def _validate_inputs(self, w, x, y):
        assert_true(w.dim() == 2,
                    "Input w should be 2D")
        assert_true(x.dim() == 2,
                    "Input datapoint should be 2D")
        assert_true(y.dim() == 1,
                    "Input label should be 1D")
        assert_true(x.size(0) == y.size(0),
                    "Input datapoint and label should contain the same number of samples")

    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute cross entropy prediction error

        xw = torch.mm(x, w)
        expxw = torch.exp(xw)
        sum_exp = torch.sum(expxw, 1)

        expand_y = y.squeeze(0).expand(self.hparams.n_features, self.hparams.n_samples)
        gather_w = torch.gather(w, 1, expand_y)

        x_wy = torch.mm(x, gather_w)
        # import pdb; pdb.set_trace()
        tot_loss = sum_exp + torch.diag(x_wy)
        loss = torch.sum(tot_loss)/sum_exp.size()[0]

        error = loss
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute objective value
        obj = self.task_error(w, x, y) + self.hparams.mu * torch.trace(w)

        # TODO: compute gradient
        xw = torch.mm(x, w)
        expxw = torch.exp(xw)
        sum_exp = torch.sum(expxw, 1)
        one_o_sum = torch.pow(sum_exp, -1)


        dw = None
        return {'obj': obj, 'dw': dw}
