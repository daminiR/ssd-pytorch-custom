from .util import *
import numpy as np
from matplotlib import pyplot as plt
#import plotly.tools as tls
from scipy.misc import imread
plt.switch_backend('agg')


# a reference Visualizer for capsule project:
# https://github.com/hli2020/object_detection/blob/214d2fa1e0352cd1643bc41949c9c66edc71040e/utils/visualizer.py
class Visualizer(object):
    def __init__(self, opt, dataset=None):
        self.opt = opt
        if self.opt.no_visdom is False:
            import visdom
            name = opt.experiment_name
            self.vis = visdom.Visdom(port=opt.port_id, env=name)
            # loss/line 100, text 200, images/hist/etc 300
            self.dis_win_id_line = 100
            self.dis_win_id_txt = 200
            self.dis_win_id_im, self.dis_im_cnt, self.dis_im_cycle = 300, 0, 4
            self.loss_data = {'X': [], 'Y': [], 'legend': ['total_loss', 'loss_c', 'loss_l']}
            # for visualization
            # TODO: visualize in the training process
            if hasattr(dataset, 'num_classes'):
                self.num_classes = dataset.num_classes
                self.class_name = dataset.COCO_CLASSES_names
                self.color = plt.cm.hsv(np.linspace(0, 1, (self.num_classes-1))).tolist()
                # for both train and test
                self.save_det_res_path = os.path.join(self.opt.save_folder, 'det_result')
                mkdirs(self.save_det_res_path)

    def plot_loss(self, errors, progress, others=None):
        """draw loss on visdom console"""
        # TODO: set figure height and width in visdom
        try:
            loss, loss_l, loss_c = errors[0].data[0], errors[1].data[0], errors[2].data[0]
        except AttributeError:
            loss, loss_l, loss_c = errors[0], 100. - errors[1], 100. - errors[2]
            self.loss_data['legend'] = ['total_loss', 'top1_error', 'top5_error']

        epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]
        x_progress = epoch + float(iter_ind/epoch_size)

        self.loss_data['X'].append([x_progress, x_progress, x_progress])
        self.loss_data['Y'].append([loss, loss_l, loss_c])

        self.vis.line(
            X=np.array(self.loss_data['X']),
            Y=np.array(self.loss_data['Y']),
            opts={
                'title': 'Train loss over time',
                'legend': self.loss_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss/acc error'},
            win=self.dis_win_id_line
        )

    def print_loss(self, errors, progress, others=None):
        """show loss info in console"""
        try:
            loss, loss_l, loss_c = errors[0].data[0], errors[1].data[0], errors[2].data[0]
            capsule_project = False
            t0, t1 = others[0], others[1]
        except:
            capsule_project = True
            if isinstance(errors, dict):
                summary = True
                train_err1 = errors['train_acc_error']
                train_err5 = errors['train_acc5_error']
                test_err1 = errors['test_acc_error']
                test_err5 = errors['test_acc5_error']
            else:
                summary = False
                loss, top1_err, top5_err = errors[0], 100. - errors[1], 100. - errors[2]
                data_time, batch_time = others[0], others[1]

        epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]

        if capsule_project:
            if summary:
                msg = 'Summary\tepoch/iter [{:d}/{:d}] ||\t' \
                      'TRAIN, Top1_err: {:.4f}, Top5_err: {:.4f} ||\t' \
                      'TEST, Top1_err: {:.4f}, Top5_err: {:.4f} ||\n'.format(
                        epoch, self.opt.max_epoch,
                        train_err1, train_err5, test_err1, test_err5)
            else:
                # means to print out training stats
                msg = '[{:s}]\tepoch/iter [{:d}/{:d}][{:d}/{:d}] ||\t' \
                      'Loss: {:.4f}, Top1_err: {:.4f}, Top5_err: {:.4f} ||\t' \
                      'Data/batch time: {:.4f}/{:.4f}'.format(
                        self.opt.experiment_name, epoch, self.opt.max_epoch, iter_ind, epoch_size,
                        loss, top1_err, top5_err, data_time, batch_time)
        else:
            # object detetion
            msg = '[{:s}]\tepoch/iter [{:d}/{:d}][{:d}/{:d}] ||\t' \
                  'Loss: {:.4f}, loc: {:.4f}, cls: {:.4f} ||\t' \
                  'Time: {:.4f} sec/image'.format(
                    self.opt.experiment_name, epoch, self.opt.max_epoch, iter_ind, epoch_size,
                    loss, loss_l, loss_c, (t1 - t0)/self.opt.batch_size)
        print_log(msg, self.opt.file_name)

    def print_info(self, progress, others):
        """print useful info on visdom console"""
        # TODO: test case and set size

        epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]
        try:
            still_run, lr, time_per_iter, test_acc, best_acc, best_epoch, param_num, total_time = \
                others[0], others[1], others[2], others[3], others[4], others[5], others[6], others[7]
            show_best = 'curr test error: {:.4f}<br/>best test error: <b>{:.4f}</b> at epoch {:3d}<br/>' \
                        'param_num: {:.4f} Mb<br/>'.format(
                            test_acc, best_acc, best_epoch, param_num)
            self.opt.start_epoch, self.opt.start_iter = 0, 0    # for compatibility
            self.opt.batch_size = self.opt.batch_size_train
        except:
            still_run, lr, time_per_iter = others[0], others[1], others[2]
            show_best = '<br/>'

        left_time = time_per_iter * (epoch_size-1-iter_ind + (self.opt.max_epoch-1-epoch)*epoch_size) / 3600 if \
            still_run else 0
        status = 'RUNNING' if still_run else 'DONE'
        dynamic = 'start epoch: {:d}, iter: {:d}<br/>' \
                  'curr lr {:.8f}<br/>' \
                  'progress epoch/iter [{:d}/{:d}][{:d}/{:d}]<br/><br/>' \
                  'est. left time: {:.4f} hours<br/>' \
                  'time/image: {:.4f} sec<br/><br/>{:s}'.format(
                    self.opt.start_epoch, self.opt.start_iter,
                    lr,
                    epoch, self.opt.max_epoch, iter_ind, epoch_size,
                    left_time, time_per_iter/self.opt.batch_size,
                    show_best)
        total_t_str = '' if still_run else 'total_time: {:.4f} hrs<br/>'.format(total_time)
        common_suffix = '<br/><br/>-----------<br/>' \
                        'batch_size: {:d}<br/>' \
                        'optim: {:s}<br/>' \
                        'loss type: {:s}<br/>' \
                        'device_id: {:s}<br/>'.format(
                            self.opt.batch_size,
                            self.opt.optim,
                            self.opt.loss_form,
                            self.opt.device_id) + total_t_str

        msg = 'phase: {:s}<br/>status: <b>{:s}</b><br/>'.format(self.opt.phase, status)\
              + dynamic + common_suffix
        self.vis.text(msg, win=self.dis_win_id_txt)

    def show_image(self, progress, others=None):
        """for test, print log info in console and show detection results on visdom"""
        if self.opt.phase == 'test':
            name = os.path.basename(os.path.dirname(self.opt.det_file))
            i, total_im, test_time = progress[0], progress[1], progress[2]
            all_boxes, im, im_name = others[0], others[1], others[2]

            print_log('[{:s}][{:s}]\tim_detect:\t{:d}/{:d} {:.3f}s'.format(
                self.opt.experiment_name, name, i, total_im, test_time), self.opt.file_name)

            dets = np.asarray(all_boxes)
            result_im = self._show_detection_result(im, dets[:, i], im_name)
            result_im = np.moveaxis(result_im, 2, 0)
            win_id = self.dis_win_id_im + (self.dis_im_cnt % self.dis_im_cycle)
            self.vis.image(result_im, win=win_id,
                           opts={
                               'title': 'subfolder: {:s}, name: {:s}'.format(
                                   os.path.basename(self.opt.save_folder), im_name),
                               'height': 320,
                               'width': 400,
                           })
            self.dis_im_cnt += 1

    def _show_detection_result(self, im, results, im_name):

        plt.figure()
        plt.axis('off')     # TODO, still the axis remains
        plt.imshow(im)
        currentAxis = plt.gca()

        for cls_ind in range(1, len(results)):
            if results[cls_ind] == []:
                continue
            else:

                cls_name = self.class_name[cls_ind-1]
                cls_color = self.color[cls_ind-1]
                inst_num = results[cls_ind].shape[0]
                for inst_ind in range(inst_num):
                    if results[cls_ind][inst_ind, -1] >= self.opt.visualize_thres:

                        score = results[cls_ind][inst_ind, -1]
                        pt = results[cls_ind][inst_ind, 0:-1]
                        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                        display_txt = '{:s}: {:.2f}'.format(cls_name, score)

                        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=cls_color, linewidth=2))
                        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': cls_color, 'alpha': .5})
                    else:
                        break
        result_file = '{:s}/{:s}.png'.format(self.save_det_res_path, im_name[:-4])

        plt.savefig(result_file, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
        # ref: https://github.com/facebookresearch/visdom/issues/119
        # plotly_fig = tls.mpl_to_plotly(fig)
        # self.vis._send({
        #     data=plotly_fig.data,
        #     layout=plotly_fig.layout,
        # })
        result_im = imread(result_file)
        return result_im

# idx = 2 if self.opt.model == 'default' or self.opt.add_gan_loss else 1
# for label, image_numpy in images.items():
#     self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
#                    win=self.display_win_id + idx)
#     idx += 1
# if args.use_visdom:
#     random_batch_index = np.random.randint(images.size(0))
#     args.vis.image(images.data[random_batch_index].cpu().numpy())

    def plot_hist(self, stats_data, info, all_sample=False):
        # TODO: complete the histogram visualization
        target_suffix = 'target' if info['target'] is True else 'non_target'
        title_suffix = 'batch_id={:d} Model: {:s}, {:s}'.format(
            info['curr_iter'],
            info['model'], target_suffix) \
            if all_sample is False else 'Model: {:s}, {:s}'.format(info['model'], target_suffix)
        data1 = stats_data[0]
        data2 = stats_data[1]
        data3 = stats_data[2]
        data4 = stats_data[3]

        # if all_sample is False:
        #     title_str = 'CosDist: i - i, ' + title_suffix
        #     self.vis.histogram(
        #         data1,
        #         win=self.dis_win_id_im,
        #         opts={
        #             'title': title_str,
        #             'xlabel': 'bin',
        #             'ylabel': 'percentage',
        #             'numbins': 60,
        #         },
        #     )
        #     self.dis_win_id_im += 1

        # title_str = '| u_hat_i |, ' + title_suffix
        # self.vis.histogram(
        #     data2,
        #     win=self.dis_win_id_im,
        #     opts={
        #         'title': title_str,
        #         'xlabel': 'bin',
        #         'ylabel': 'percentage',
        #         'numbins': 30
        #     },
        # )
        # self.dis_win_id_im += 1

        title_str = 'CosDist: i - j, ' + title_suffix
        self.vis.histogram(
            data3,
            win=self.dis_win_id_im,
            opts={
                'title': title_str,
                'xlabel': 'bin',
                'ylabel': 'percentage',
                'numbins': 30
            },
        )
        self.dis_win_id_im += 1

        title_str = 'AvgLen: i - j, ' + title_suffix
        self.vis.line(
            X=np.linspace(-1, 1, 21),
            Y=np.array(data4['Y']),
            win=self.dis_win_id_im,
            opts={
                'title': title_str,
                'xlabel': 'distance',
                'ylabel': 'length',
            },
        )
        self.dis_win_id_im += 1

