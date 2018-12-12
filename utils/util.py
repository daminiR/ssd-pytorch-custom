from __future__ import print_function
import torch
from PIL import Image
import inspect
import re
import numpy as np
import os
import time
import collections
import torch.nn as nn
import sys
import math
from torch.nn.modules.module import _addindent

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()     # only draw the first image in a mini-batch
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]))


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _process(ids):
    str_ids = ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    return gpu_ids


def weights_init(m):
    """
        init random weights
    """
    # if isinstance(m, nn.Conv2d):
    #     nn.init.xavier_normal(m.weight.data)
    #     m.bias.data.zero_()
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        # print('linear layer!')
        m.weight.data.normal_(std=0.05)
        m.bias.data.zero_()


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def remove(file_name):
    try:
        os.remove(file_name)
    except:
        pass


def print_log(msg, file=None, init=False):

    print(msg)
    if file is None:
        pass
    else:
        if init:
            remove(file)
        with open(file, 'a') as log_file:
            log_file.write('%s\n' % msg)


def show_jot_opt(opt):

    """
        by Hongyang.
        A starter for logging the training/test process
    """
    if opt.phase == 'test':
        file_name = os.path.join(opt.save_folder, 'opt_{:s}.txt'.format(opt.phase))
    elif opt.phase == 'train':
        file_name = os.path.join(opt.save_folder,
                                 'opt_{:s}_START_epoch_{:d}_iter_{:d}_END_{:d}.txt'.format(
                                     opt.phase, opt.start_epoch, opt.start_iter, opt.max_epoch))
    elif opt.phase == 'train_val':
        file_name = os.path.join(opt.save_folder,
                                 'opt_{:s}_START_epoch_0_END_{:d}.txt'.format(
                                     opt.phase, opt.max_epoch))

    opt.file_name = file_name
    args = vars(opt)

    print_log('Experiment: {:s}'.format(opt.experiment_name), file_name, init=True)
    if opt.phase == 'train':
        print_log('------------ Training Options -------------', file_name)
    elif opt.phase == 'test':
        print_log('------------ Test Options -----------------', file_name)
    elif opt.phase == 'train_val':
        print_log('------------ Train and Test Options -----------------', file_name)

    for k, v in sorted(args.items()):
        print_log('%s: %s' % (str(k), str(v)), file_name)
    print_log('------------------ End --------------------', file_name)
    return opt


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    params_num = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()

        if isinstance(modstr, str):
            modstr = _addindent(modstr, 2)
        elif isinstance(modstr, tuple):
            modstr = _addindent(modstr[0], 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        # rest = b if b > 0 else params
        # params_num = params_num + rest
        params_num += params

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr, params_num * 4. / (1024**2)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count