import pickle
from .util import *
from data.custom import CUSTOM_ROOT, get_targets
from .visualizer import Visualizer, print_log
import cv2
from matplotlib import pyplot as plt

try:
    from data.setup_dset import VOC_CLASSES as labelmap
except ModuleNotFoundError:
    pass

try:
    from data.custom import CUSTOM_CLASSES as labelmap
except ModuleNotFoundError:
    pass

def accuracy(output, target, topk=(1,), acc_per_cls=None):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    if acc_per_cls is not None:
        target_min, target_max = target.min(), target.max()
        for i in range(target_min, target_max + 1):
            acc_per_cls['top1'][i, 1] += sum(target == i)
            acc_per_cls['top5'][i, 1] += sum(target == i)

            _check = correct[:1].sum(dim=0)
            acc_per_cls['top1'][i, 0] += sum(_check[target == i])
            _check = correct[:5].sum(dim=0)
            acc_per_cls['top5'][i, 0] += sum(_check[target == i])

    return res, acc_per_cls


def _parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects

def _parse_rec_custom(filename):
    """Parse a json annotation file and return all bounding
    boxes for all images as a dict of dict of list.
    """
    # Process annot file
    targets = get_targets(filename)
    # This will be a list of images and bboxes therein
    objects_all = {}
    # scale = np.array([width, height, width, height])
    for target_id in targets:
        objects = []
        img = cv2.imread(os.path.join(CUSTOM_ROOT, 'test', target_id))
        height, width, _ = img.shape
        scale = np.array([width, height, width, height])
        # Loop through all bboxes in an image
        for _, elem in enumerate(targets[target_id]):
            for cls_name in labelmap:
                obj_struct = {}
                bbox = np.zeros(shape=4)
                # VGG Image Annotator produces x_1, y_1, width, height
                bbox[0] = elem['x']
                bbox[1] = elem['y']
                bbox[2] = bbox[0] + elem['width']
                bbox[3] = bbox[1] + elem['height']
                final_box = list(np.array(bbox)/scale)
                # final_box.append(custom_class)
                # Add the new bbox to dict of lists of lists
                obj_struct['bbox'] = final_box
                obj_struct['name'] = cls_name
                obj_struct['difficult'] = 0 # False for now, no difficult gt boxes
                objects.append(obj_struct)
        # Append all bboxes from an image to the list of images
        objects_all[target_id] = objects
    return objects_all

# def get_output_dir(name, phase):
#     # DEPRECATED
#     """Return the directory where experimental artifacts are placed.
#     If the directory does not exist, it is created.
#     A canonical path is built using the name from an imdb and a network
#     (if not None).
#     """
#     filedir = os.path.join(name, phase)
#     if not os.path.exists(filedir):
#         os.makedirs(filedir)
#     return filedir


def _get_voc_results_file_template(save_folder, image_set, cls):
    print('save folder in template ', save_folder)
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(save_folder, 'detection_per_cls')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    # path = result/EXPERIMENT_NAME/test/detection_per_cls/det_test_XXX_CLS.txt
    path = os.path.join(filedir, filename)
    return path


def _voc_eval(dataset, detfile, annopath, imagesetfile, classname,
              cachedir, ovthresh=0.5, use_07_metric=True):
    """
    Top level function that does the PASCAL VOC evaluation.
    detpath:            Path to detections, detpath.format(classname) should produce the detection results file.
    annopath:           Path to annotations, annopath.format(imagename) should be the xml annotations file.
    imagesetfile:       Text file containing the list of images, one image per line.
    classname:          Category name (duh)
    cachedir:           Directory for caching the annotations
    [ovthresh]:         Overlap threshold (default = 0.5)
    [use_07_metric]:    Whether to use VOC07's 11 point AP computation (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # if not os.path.isfile(cachefile):
        # Load annotations
    recs = {}
    if dataset ==  'VOC':
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = _parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imagenames)))
    # TODO: COCO annots
    else:
        recs = _parse_rec_custom(os.path.join(CUSTOM_ROOT, 'test', 'annot', 
            'via_region_data.json'))
        
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
        pickle.dump(recs, f)
    # else:
    #     # load
    #     with open(cachefile, 'rb') as f:
    #         recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        if imagename in recs:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}

    # read dets
    print(detfile)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        print(image_ids)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                print(overlaps)
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                # if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    print('true pos!')
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = _voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def _voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def write_voc_results_file(args, all_boxes, dataset):
    # color = plt.cm.hsv(np.linspace(0, 1, (dataset.num_classes-1))).tolist()
    for cls_ind, cls_name in enumerate(labelmap):
        print('Writing {:s} results file'.format(cls_name))
        set_type = 'test'
        filename = _get_voc_results_file_template(args.save_folder, set_type, cls_name)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

def do_python_eval(opts):
    prefix = opts.base_save_folder
    log_file_name = opts.log_file_name
    save_folder = opts.save_folder
    home = os.path.expanduser("~")
    if opts.dataset == "VOC":
        data_root = os.path.join(home, "data/VOCdevkit/")
        devkit_path = data_root + 'VOC' + '2007'
        cachedir = os.path.join(devkit_path, 'annotations_cache')
        annopath = os.path.join(data_root, 'VOC2007', 'Annotations', '%s.xml')
        imgsetpath = os.path.join(data_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
    else:
        data_root = CUSTOM_ROOT
        cachedir = os.path.join(data_root, 'annotations_cache')
        annopath = os.path.join(data_root, 'test', 'annot', '%s.json')
        imgsetpath = os.path.join(data_root, 'testimageset.txt')

    set_type = 'test'

    output_dir = os.path.join(prefix, opts.subname, 'pr_curve_per_cls')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    use_07_metric = True
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

    with open(log_file_name, 'a') as log_file:
        aps = []
        for i, cls in enumerate(labelmap):
            print('save_folder ', save_folder)
            filename = _get_voc_results_file_template(save_folder, set_type, cls)
            # filename = os.path.join(save_folder, set_type, 'det_' + set_type + '_%s.txt' % (cls))
            print('filename ', filename)
            rec, prec, ap = _voc_eval(
                opts.dataset, filename, annopath, imgsetpath.format(set_type), cls, cachedir,
                ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            log_file.write('AP for {} = {:.4f}\n'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        log_file.write('Mean AP = {:.4f}\n'.format(np.mean(aps)))
        # print('~~~~~~~~')
        # print('Results:')
        # for ap in aps:
        #     print('{:.3f}'.format(ap))
        # print('{:.3f}'.format(np.mean(aps)))
        # print('~~~~~~~~')
        # print('')
        # print('--------------------------------------------------------------')
        # print('Results computed with the **unofficial** Python eval code.')
        # print('Results should be very close to the official MATLAB eval code.')
        # print('--------------------------------------------------------------')


########### COCO ###########

def _coco_results_one_category(dataset, boxes, cat_id):
    results = []
    # len(dataset.ids) = 5000, the size of validation
    # boxes[im_ind] is of size (num_instance_in_this_image x 5)

    for im_ind, index in enumerate(dataset.ids):
        dets = np.array(boxes[im_ind], dtype=np.float)
        if len(dets) == 0:
            continue
        scores = dets[:, -1]
        xs = dets[:, 0]
        ys = dets[:, 1]
        ws = dets[:, 2] - xs + 1
        hs = dets[:, 3] - ys + 1
        results.extend([{
                            'image_id': index,
                            'category_id': cat_id,
                            'bbox': [xs[k], ys[k], ws[k], hs[k]],
                            'score': scores[k]
                        } for k in range(dets.shape[0])])  # k is the isntance number
    return results


def write_coco_results_file(dataset, all_boxes, args):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "bbox": [258.15,41.29,348.26,243.78],
    #   "score": 0.236}, ...]

    res_file = args.det_file[:-3] + 'json'
    if os.path.isfile(res_file):
        print_log('\nThe json file already exists ...\n', args.file_name)
    else:
        results = []
        for cls_ind, cls in enumerate(dataset.COCO_CLASSES_names):
            if cls == '__background__':  # we don't have this case
                continue
            # print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind, dataset.num_classes - 2))
            coco_cat_id = dataset.COCO_CLASSES[cls_ind]
            results.extend(
                _coco_results_one_category(dataset, all_boxes[cls_ind + 1], coco_cat_id))

        print_log('\nWriting results in json format to {} ...\n'.format(res_file), args.file_name)
        with open(res_file, 'w') as fid:
            json.dump(results, fid)


def coco_do_detection_eval(dataset, args):
    res_file = args.det_file[:-3] + 'json'
    ann_type = 'bbox'
    coco_dt = dataset.coco.loadRes(res_file)
    from pycocotools.cocoeval import COCOeval
    coco_eval = COCOeval(dataset.coco, coco_dt)
    coco_eval.params.useSegm = (ann_type == 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    _print_detection_eval_metrics(dataset, coco_eval, args)

def _print_detection_eval_metrics(dataset, coco_eval, args):
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    mAP = 100 * ap_default

    print_log('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~'.
              format(IoU_lo_thresh, IoU_hi_thresh), args.file_name)
    print_log('[{:s}][{:s}]\nMean AP: {:.2f}\n'.format(
        args.experiment_name,
        os.path.basename(os.path.dirname(args.det_file)), mAP,
    ), args.file_name)

    for cls_ind, cls in enumerate(dataset.COCO_CLASSES_names):
        if cls == '__background__':
            continue
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind, 0, 2]
        ap = np.mean(precision[precision > -1])
        print_log('{:s}:\t\t\t\t{:.2f}'.format(cls, 100 * ap), args.file_name)

    print_log('\n~~~~ Summary metrics ~~~~\n[{:s}][{:s}]'.format(
        args.experiment_name,
        os.path.basename(os.path.dirname(args.det_file))
    ), args.file_name)
    # coco_eval.summarize()
    coco_eval.stats = _summarize_only_to_log(coco_eval, args)


def _summarize_only_to_log(api, args):
    """
        Note by hyli: only to log in the file when printing. Exactly the same as official.
        'cocoeval.py'
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = api.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.2f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = api.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = api.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        print_log(
            iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s * 100),
            args.file_name)
        return mean_s

    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=.5, maxDets=api.params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=api.params.maxDets[2])
        stats[3] = _summarize(1, areaRng='small', maxDets=api.params.maxDets[2])
        stats[4] = _summarize(1, areaRng='medium', maxDets=api.params.maxDets[2])
        stats[5] = _summarize(1, areaRng='large', maxDets=api.params.maxDets[2])
        stats[6] = _summarize(0, maxDets=api.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=api.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=api.params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=api.params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=api.params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=api.params.maxDets[2])
        return stats

    def _summarizeKps():
        stats = np.zeros((10,))
        stats[0] = _summarize(1, maxDets=20)
        stats[1] = _summarize(1, maxDets=20, iouThr=.5)
        stats[2] = _summarize(1, maxDets=20, iouThr=.75)
        stats[3] = _summarize(1, maxDets=20, areaRng='medium')
        stats[4] = _summarize(1, maxDets=20, areaRng='large')
        stats[5] = _summarize(0, maxDets=20)
        stats[6] = _summarize(0, maxDets=20, iouThr=.5)
        stats[7] = _summarize(0, maxDets=20, iouThr=.75)
        stats[8] = _summarize(0, maxDets=20, areaRng='medium')
        stats[9] = _summarize(0, maxDets=20, areaRng='large')
        return stats

    if not api.eval:
        raise Exception('Please run accumulate() first')
    iouType = api.params.iouType
    if iouType == 'segm' or iouType == 'bbox':
        summarize = _summarizeDets
    elif iouType == 'keypoints':
        summarize = _summarizeKps

    return summarize()