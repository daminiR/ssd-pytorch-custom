import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch

from utils.train_utils import set_model_weight_train, set_model_weight_test
from option.config import mbox, base, extras
from layers.models.models import *
from layers.modules.prior_box import PriorBox
from layers.modules.l2norm import L2Norm
from layers.detection import Detect

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_ssd(opts, num_classes):
    """shared by both train and test
    """
    phase = opts.phase
    size = opts.ssd_dim

    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return

    vgg_layers = vgg(base[str(size)], 3)
    print('Layer count ', len(vgg_layers))
    extra_layers = add_extras(extras[str(size)], 1024)
    mbox_setting = mbox['more_ar'] \
        if opts.prior_config == 'v2_512_stan_more_ar' or opts.prior_config == 'v2_634' \
            else mbox['original']
    model = SSD(opts, phase, num_classes,
                *multibox(vgg_layers, extra_layers, mbox_setting, num_classes))

    if phase == 'train':
        model.train()
        # init the network
        model, start_epoch, start_iter = \
            set_model_weight_train(model, opts)
        # model.load_weight_new()
        if opts.debug_mode:
            print(model)
        else:
            print('Network structure not shown in deploy mode')
        if opts.debug_mode:
            model = model.to(device)
        else:
            model = torch.nn.DataParallel(model).to(device)
            print('launch the parallel mode ...')
        cudnn.benchmark = True
        return model, (start_epoch, start_iter)

    elif phase == 'test':
        model = set_model_weight_test(model, opts)
        model.eval()
        model = model.to(device)
        cudnn.benchmark = True
        return model


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, opts, phase, num_classes, base, extras, head):
        super(SSD, self).__init__()
        self.opts = opts
        self.phase = phase
        self.num_classes = num_classes
        self.priorbox = PriorBox(opts.prior_config, opts.ssd_dim)

        # for ssd300, priors: [8732 x 4] boxes/anchors
        with torch.no_grad():
            self.priors = self.priorbox.forward()

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax()
            # num_classes, bkg_label, top_k, conf_thresh, nms_thresh
            self.detect = Detect(num_classes, 0, opts.top_k,
                                 opts.conf_thresh, opts.nms_thresh,
                                 opts.soft_nms)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                tensor of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # If test phase, don't track gradients
        if self.phase == 'test':
            with torch.no_grad():
                # apply vgg up to conv4_3 relu
                for k in range(23):
                    x = self.vgg[k](x)
        else:
            # apply vgg up to conv4_3 relu
            for k in range(23):
                x = self.vgg[k](x)



        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                      # loc preds
                self.softmax(conf.view(-1, self.num_classes)),      # conf preds
                self.priors#.type(type(x.data))               # default boxes
            )
        elif self.phase == "train":
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output