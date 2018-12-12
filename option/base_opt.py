import argparse
from utils.util import *
import random


class BaseOptions(object):

    def initialize(self):
        self.parser = argparse.ArgumentParser(description='Object Detection')
        self.parser.add_argument('--experiment_name', default='ssd_rerun')
        self.parser.add_argument('--dataset', default='coco', help='[ voc|coco ]')
        self.parser.add_argument('--base_save_folder', default='result')

        self.parser.add_argument('--manual_seed', default=-1, type=int)
        self.parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--no_visdom', action='store_true')
        self.parser.add_argument('--port_id', default=8090, type=int)

        # model params
        self.parser.add_argument('--ssd_dim', default=512, type=int)
        self.parser.add_argument('--prior_config', default='v2_512', type=str)

    def setup_config(self):
        """pre-process stuff for both train and test"""
        if hasattr(self.opt, 'subname'):
            # for test folder name
            _temp = '' if self.opt.subname == '' else '_'
            suffix = self.opt.trained_model + _temp + self.opt.subname
        else:
            suffix = ''
        self.opt.save_folder = os.path.join(self.opt.base_save_folder,
                                            self.opt.experiment_name,
                                            self.opt.phase, suffix)
        if not os.path.exists(self.opt.save_folder):
            mkdirs(self.opt.save_folder)

        seed = random.randint(1, 10000) if self.opt.manual_seed == -1 else self.opt.manual_seed
        self.opt.random_seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            self.opt.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.opt.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')

        if self.opt.phase == 'train':
            if self.opt.debug_mode:
                self.opt.loss_freq = 10
                self.opt.save_freq = self.opt.loss_freq
            else:
                self.opt.loss_freq = 200    # in iter unit
                self.opt.save_freq = 5      # in epoch unit
        else:
            self.opt.trained_model = os.path.join(self.opt.base_save_folder,
                                                  self.opt.experiment_name,
                                                  'train', self.opt.trained_model+'.pth')
            if not os.path.isfile(self.opt.trained_model):
                print('trained model not exist! {:s}'.format(self.opt.trained_model))
                quit()

            self.opt.show_freq = 50
            self.opt.det_file = os.path.join(self.opt.save_folder, 'detections_all_boxes.pkl')



