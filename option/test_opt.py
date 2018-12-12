from .base_opt import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):

        BaseOptions.initialize(self)

        self.parser.add_argument('--trained_model', default='ssd512_COCO_epoch_20_iter_29570', type=str)
        self.parser.add_argument('--subname', default='', type=str)

        self.parser.add_argument('--soft_nms', type=int, default=-1)  # set -1 if not used
        self.parser.add_argument('--conf_thresh', default=0.05, type=float, help='detection confidence threshold')
        self.parser.add_argument('--top_k', default=300, type=int,
                                 help='The Maximum number of box preds to consider in NMS.')
        self.parser.add_argument('--nms_thresh', default=0.5, type=float)
        self.parser.add_argument('--visualize_thres', default=0.2, type=float)

        self.opt = self.parser.parse_args()
        self.opt.phase = 'test'


