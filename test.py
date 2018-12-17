"""SSD model testing script

For help and usage:

  python test.py -h

Example:

python test.py --experiment_name ssd_custom --dataset Custom --base_save_folder run --num_workers 0 --ssd_dim 300 --trained_model run/ssd_custom/train/debug_ssd300_CUSTOM_epoch_20_iter_3.pth --prior_config custom
"""

from __future__ import print_function
import os
import numpy as np
import pickle
import torch
from layers.ssd import build_ssd
from option.test_opt import TestOptions
from utils.eval_utils import write_voc_results_file, show_jot_opt, Timer, do_python_eval
from utils.visualizer import Visualizer, print_log
from data.custom import CUSTOM_ROOT, CustomAnnotationTransform, CustomDetection, MEANS
from data import BaseTransform # in __init__.py
from utils.augmentations import SSDAugmentationLimited

# If GPU is available use it, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('On {} device'.format(device))


# # TODO add VOC back
# if args.dataset == 'VOC':
#     from data import VOC_CLASSES as labelmap
# else:
from data import CUSTOM_CLASSES as labelmap


option = TestOptions()
option.setup_config()
args = option.opt

# Maybe add later
# dataset = create_dataset(args)

# Load data (TODO:  add COCO)
# if args.dataset == 'VOC':
#     dataset = VOCDetection(args.dataset_root, [(set_type)],
#                         BaseTransform(args.ssd_dim, MEANS),
#                         VOCAnnotationTransform())
# else:
dataset = CustomDetection(root=CUSTOM_ROOT, 
                            image_set='test', 
                            transform=SSDAugmentationLimited(args.ssd_dim,
                                                         MEANS),
                            target_transform=CustomAnnotationTransform(train=False),
                            phase='test')

# init log file
show_jot_opt(args)

# init visualizer
visual = Visualizer(args, dataset)

# all detections are collected into:
#    all_boxes[cls][image] = N x 5 array of detections in
#    (x1, y1, x2, y2, score)
if os.path.isfile(args.det_file):
    print_log('\nRaw boxes exist! skip prediction and directly evaluate!',
              args.file_name)
    all_boxes = pickle.load(open(args.det_file, 'rb'))
else:
    ssd_net = build_ssd(args, dataset.num_classes)
    t = Timer()
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(dataset.num_classes)]
    for i in range(num_images):
        im, _, h, w, orgin_im, im_name = dataset.pull_item(i)
        x = im.unsqueeze(0)
        t.tic()
        with torch.no_grad():
            x = x.to(device)
            detections = ssd_net(x).data
        detect_time = t.toc(average=False)
        # skip j = 0 (background class)
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if len(dets) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            all_boxes[j][i] = \
                np.hstack((boxes.cpu().numpy(), \
                    scores[:, np.newaxis])).astype(np.float32, copy=False)
        try:
            dets_current = np.array(all_boxes).transpose()[i]
            progress = (i, num_images, detect_time)
            show_off = (dets_current, orgin_im[...,[2,1,0]], im_name)
            visual.show_image(progress, show_off)
        except IndexError as err:
            print('IndexError')
            continue

    # save the raw boxes
    with open(args.det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

print_log('\nEvaluating detection results ...', args.file_name)
# if dataset.name == 'COCO':
#     write_coco_results_file(dataset, all_boxes, args)
#     coco_do_detection_eval(dataset, args)
# else:
write_voc_results_file(args, all_boxes, dataset)
do_python_eval(args)

print_log('\nHurray! Testing done!', args.file_name)