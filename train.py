"""SSD model training script

For help and usage:

  python train.py -h

Example:

  python train.py --experiment_name ssd_custom --dataset Custom --base_save_folder run --num_workers 0 --ssd_dim 300 --batch_size 4 --lr 1e-4 --max_epoch 10 --pretrain_model weights/vgg16_reducedfc.pth

"""

from data import custom, VOC_ROOT, CUSTOM_ROOT, detection_collate
from data.custom import CustomDetection, MEANS
from option.config import custom
from utils.augmentations import SSDAugmentation, SSDAugmentationLimited
from layers.modules import MultiBoxLoss
from layers.ssd import build_ssd
from option.train_opt import TrainOptions
from utils.train_utils import set_optimizer, save_model
from utils.util import print_log, show_jot_opt
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import copy


# config
option = TrainOptions()
option.setup_config()
args = option.opt

# If GPU is available use it, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('On {} device'.format(device))

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def train_model(model, criterion, optimizer, scheduler, dataset, num_epochs=25, phase='train'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False, 
                                  collate_fn=detection_collate,
                                  pin_memory=True)
    print('Data loader length...', len(data_loader))
    
    epoch_size = len(dataset) // args.batch_size

    for epoch in range(num_epochs):
        # Reset iterations
        batch_iterator = iter(data_loader)

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model = model.to(device)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                lr = scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            batch_size = args.batch_size
            epoch_steps = len(data_loader) // batch_size
            for i in range(epoch_steps):
                # load train data
                inputs, labels = next(batch_iterator)           
                inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss_l, loss_c = criterion(outputs, labels, debug=False)
                    loss = loss_l + loss_c

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / (batch_size * epoch_steps)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # one epoch ends, save results
            # by default the debug mode won't go here
            if epoch % args.save_freq == 0 or epoch == args.max_epoch-1:
                progress = (epoch, i, epoch_size)
                save_model(progress, args, (model, dataset))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # train()

    if args.dataset == 'VOC':
        cfg = voc
        dataset = VOCDetection(root=VOC_ROOT,
                               transform=SSDAugmentation(args.ssd_dim,
                                                         MEANS))
    else:
        cfg = custom
        dataset = CustomDetection(root=CUSTOM_ROOT,
                               transform=SSDAugmentationLimited(args.ssd_dim,
                                                         MEANS),
                                                         phase='train')

    since = time.time()

    ssd_net, (args.start_epoch, args.start_iter) = build_ssd(args, num_classes=dataset.num_classes)

    # init log file
    show_jot_opt(args)

    criterion = MultiBoxLoss(dataset.num_classes, overlap_thresh=0.4, 
        prior_for_matching=True, bkg_label=0, neg_mining=True, neg_pos=3, 
        neg_overlap=0.5, encode_target=False)

    # Observe that all parameters are being optimized
    optimizer = set_optimizer(ssd_net, args)

    # Decay LR by a factor of 0.05 every 7 epochs - if getting NaN might nee to lower less
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.05)

    model_ft = train_model(ssd_net, criterion, optimizer, exp_lr_scheduler, 
        dataset=dataset, num_epochs=args.max_epoch, phase='train')
