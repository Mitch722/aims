import os
import sys
import socket
import torch
import logger
import random
import numpy as np


def set_seed(args, print_out=True):
    if args.seed is None:
        np.random.seed(None)
        args.seed = np.random.randint(1e5)
    if print_out:
        print('Seed:\t {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_xp(args, optimizer):
    # various useful information to store
    args.command_line = 'python ' + ' '.join(sys.argv)
    args.pid = os.getpid()
    args.cwd = os.getcwd()
    args.hostname = socket.gethostname()

    xp = logger.Experiment(args.xp_name,
                           use_visdom=args.visdom,
                           visdom_opts={'server': 'http://localhost',
                                        'port': args.port},
                           time_indexing=False, xlabel='Epoch')

    xp.SumMetric(name='epoch', to_plot=False)

    xp.AvgMetric(name='error', tag='train')
    xp.AvgMetric(name='error', tag='val')
    xp.AvgMetric(name='error', tag='test')

    xp.TimeMetric(name='timer', tag='train')
    xp.TimeMetric(name='timer', tag='val')
    xp.TimeMetric(name='timer', tag='test')

    xp.AvgMetric(name='obj', tag='train')

    xp.log_config(vars(args))

    return xp


def print_total_time(xp):
    times = list(xp.logged['timer_train'].values())
    avg_time = np.mean(times)
    total_time = np.sum(times)
    print("\nTotal training time: \t {0:g}s (avg of {1:g}s per epoch)"
          .format(total_time, avg_time))


@torch.autograd.no_grad()
def accuracy(out, targets):
    _, pred = torch.max(out, 1)
    targets = targets.type_as(pred)
    acc = torch.mean(torch.eq(pred, targets).float())
    return acc


def update_metrics(xp, state):
    xp.Error_Train.update(state['error'], n=state['size'])
    xp.Obj_Train.update(state['obj'], n=state['size'])


def log_metrics(xp, epoch):
    # Average of accuracy and loss on training set
    xp.Error_Train.log_and_reset(epoch)
    xp.Obj_Train.log_and_reset(epoch-1)

    # timer of epoch
    xp.Timer_Train.log_and_reset()


def assert_true(value, error_msg):
    if not value:
        raise RuntimeError(error_msg)


def assert_arg(value, expected_value, error_msg):
    if not value == expected_value:
        raise RuntimeError(error_msg)
