import torch

from tqdm import tqdm
from utils import log_metrics, update_metrics


def train(obj, optimizer, dataset, xp, args, epoch):

    xp.Timer_Train.reset()
    stats = {}

    for i, x, y in tqdm(optimizer.get_sampler(dataset), desc='Train Epoch',
                        leave=False, total=optimizer.get_sampler_len(dataset)):

        oracle_info = obj.oracle(optimizer.variables.w, x, y)
        oracle_info['i'] = i
        optimizer.step(oracle_info)

        # track statistics for monitoring
        stats['obj'] = float(oracle_info['obj'])
        stats['error'] = float(obj.task_error(optimizer.variables.w, x, y))
        stats['size'] = float(x.size(0))
        update_metrics(xp, stats)

    xp.Timer_Train.update()

    print('\nEpoch: [{0}] (Train) \t'
          '({timer:.2f}s) \t'
          'Obj {obj:.3f}\t'
          'Error {error:.2f}\t'
          .format(int(xp.Epoch.value),
                  timer=xp.Timer_Train.value,
                  error=xp.Error_Train.value,
                  obj=xp.Obj_Train.value,
                  ))
    log_metrics(xp, epoch)


@torch.autograd.no_grad()
def test(obj, optimizer, dataset, xp, args, epoch):
    Error = xp.get_metric(name='error', tag=dataset.tag)
    Timer = xp.get_metric(name='timer', tag=dataset.tag)
    Error.reset()
    Timer.reset()

    for idx, x, y in tqdm(optimizer.get_sampler(dataset), leave=False,
                          desc='{} Epoch'.format(dataset.tag.title()),
                          total=optimizer.get_sampler_len(dataset)):
        Error.update(obj.task_error(optimizer.variables.w, x, y), n=x.size(0))

    Timer.update().log(epoch)
    Error.log(epoch)
    print('Epoch: [{0}] ({tag})\t'
          '({timer:.2f}s) \t'
          'Obj ----\t'
          'Error {error:.2f} \t'
          .format(int(xp.Epoch.value),
                  tag=dataset.tag.title(),
                  timer=Timer.value,
                  error=Error.value
                  ))
