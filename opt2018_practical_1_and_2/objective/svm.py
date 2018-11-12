import torch

from objective.base import Objective
from utils import accuracy, assert_true


class SVM(Objective):
    def __init__(self, hparams):
        super(SVM, self).__init__(hparams)
        self._range = torch.arange(hparams.n_classes)[None, :]

    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute mean misclassification
        error = None
        return error

    def _validate_inputs(self, w, x, y):
        assert_true(w.dim() == 2, "Input w should be 2D")
        assert_true(x.dim() == 2, "Input datapoint should be 2D")
        assert_true(y.dim() == 1, "Input label should be 1D")
        assert_true(x.size(0) == y.size(0),
                    "Input datapoint and label should contain the same number of samples")


class SVM_SubGradient(SVM):
    def __init__(self, hparams):
        super(SVM_SubGradient, self).__init__(hparams)

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)

        # TODO: Compute objective value
        obj = None
        # TODO: compute subgradient
        dw = None

        return {'obj': primal, 'dw': dw}


class SVM_ConditionalGradient(SVM):
    def __init__(self, hparams):
        super(SVM_ConditionalGradient, self).__init__(hparams)

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)

        # TODO: Compute primal objective value
        primal = None
        # TODO: Compute w_s
        w_s = None
        # TODO: Compute l_s
        l_s = None

        return {'obj': primal, 'w_s': w_s, 'l_s': l_s}
