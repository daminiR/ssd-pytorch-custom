from .base_opt import BaseOptions
from utils.util import *


class TrainOptions(BaseOptions):
    def __init__(self):

        BaseOptions.initialize(self)

        self.parser.add_argument('--debug_mode', default=True, type=str2bool)

        self.parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
        self.parser.add_argument('--scheduler', default=None, help='plateau, multi_step')
        self.parser.add_argument('--optim', default='rmsprop', type=str)
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--weight_decay', default=5e-4, type=float)
        self.parser.add_argument('--gamma', default=0.1, type=float)
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')

        self.parser.add_argument('--batch_size', default=4, type=int)
        self.parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
        # self.parser.add_argument('--resume', default='ssd_rerun_resume/train/debug_ssd512_COCO_epoch_0_iter_90', type=str)
        self.parser.add_argument('--no_pretrain', action='store_true', help='default is using pretrain')
        self.parser.add_argument('--pretrain_model', default='vgg16_reducedfc.pth')

        self.parser.add_argument('--max_epoch', default=20, type=int, help='Number of training epoches')
        self.parser.add_argument('--schedule', default=[6, 12, 16], nargs='+', type=int)

        self.opt = self.parser.parse_args()
        self.opt.phase = 'train'

