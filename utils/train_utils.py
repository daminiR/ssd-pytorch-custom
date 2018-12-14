import torch.optim as optim
from .util import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, \
    ExponentialLR, MultiStepLR, StepLR, LambdaLR


def adjust_learning_rate(optimizer, step, args):
    """
        Sets the learning rate to the initial LR decayed by gamma
        at every specified step/epoch

        Adapted from PyTorch Imagenet example:
        https://github.com/pytorch/examples/blob/master/imagenet/main.py

        step could also be epoch
    """
    schedule_list = np.array(args.schedule)
    decay = args.gamma ** (sum(step >= schedule_list))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def set_lr_schedule(optimizer, plan, others=None):
    scheduler = []
    if plan == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, 'min',
                                      patience=25,
                                      factor=0.7,
                                      min_lr=0.00001)
    elif plan == 'multi_step':
        scheduler = MultiStepLR(optimizer,
                                milestones=others['milestones'],
                                gamma=others['gamma'])
    return scheduler


def save_model(progress, args, others):

    epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]
    model, dataset = others[0], others[1]
    prefix = 'debug_' if args.debug_mode else ''
    print_log('Saving state at epoch/iter [{:d}/{:d}][{:d}/{:d}] ...'.format(
        epoch, args.max_epoch, iter_ind, epoch_size), args.file_name)
    torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'iter': iter_ind,
    }, '{:s}/{:s}ssd{:d}_{:s}_epoch_{:d}_iter_{:d}.pth'.format(
        args.save_folder, prefix, args.ssd_dim, dataset.name, epoch, iter_ind))


def _remove_batch(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            remove(os.path.join(dir, f))


def save_checkpoint(state, is_best, args, epoch):
    # for the capsule project
    filepath = os.path.join(args.save_folder, 'epoch_{:d}.pth'.format(epoch+1))
    if (epoch+1) % args.save_epoch == 0 \
            or epoch == 0 or (epoch+1) == args.max_epoch:
        torch.save(state, filepath)
        print_log('model saved at {:s}'.format(filepath), args.file_name)
    if is_best:
        # save the best model
        _remove_batch(args.save_folder, 'model_best')
        best_path = os.path.join(args.save_folder, 'model_best_at_epoch_{:d}.pth'.format(epoch+1))
        torch.save(state, best_path)
        print_log('best model saved at {:s}'.format(best_path), args.file_name)


def set_optimizer(net, opt):

    optimizer = []
    if opt.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=opt.lr,
                              momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr,
                               weight_decay=opt.weight_decay, betas=(opt.beta1, 0.999))
    elif opt.optim == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=opt.lr,
                                  weight_decay=opt.weight_decay, momentum=opt.momentum,
                                  alpha=0.9, centered=True)
    return optimizer


def set_model_weight_train(model, opts):

    start_epoch = 0
    start_iter = 0
    if opts.resume:
        resume_file = os.path.join(opts.base_save_folder, (opts.resume + '.pth'))
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{:s}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            weights = checkpoint['state_dict']
            try:
                model.load_state_dict(weights)
            except KeyError:
                weights_new = collections.OrderedDict([(k[7:], v) for k, v in weights.items()])
                model.load_state_dict(weights_new)
            start_epoch = checkpoint['epoch']
            start_iter = checkpoint['iter']
        else:
            print("=> no checkpoint found at '{}'".format(opts.resume))
            exit()
    else:
        if opts.no_pretrain:
            print('Train from scratch...')
            model.apply(weights_init)
        else:
            # use pretrain model to init the network/model
            vgg_weights = torch.load(opts.pretrain_model)
            print('Loading pretrain network...')
            model.vgg.load_state_dict(vgg_weights)
            print('Initializing weights of the newly added layers...')
            # initialize newly added layers' weights with xavier method
            model.extras.apply(weights_init)
            model.loc.apply(weights_init)
            model.conf.apply(weights_init)

    return model, start_epoch, start_iter


def set_model_weight_test(model, opts):

    checkpoint = torch.load(opts.trained_model)
    try:
        # for small-scale datasets, capsule project
        print('best test accu is: {:.4f}'.format(checkpoint['best_test_acc']))
    except KeyError:
        pass
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except KeyError:
        weights = collections.OrderedDict([(k[7:], v) for k, v in checkpoint['state_dict'].items()])
        model.load_state_dict(weights)
    print('Finished loading model in test phase!')
    return model
