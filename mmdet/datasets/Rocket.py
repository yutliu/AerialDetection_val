from .coco import CocoDataset
import numpy as np
from mmcv.utils import print_log
from mmdet.utils.rocket_mAP_count import voc_eval
import os

# class RocketDataset(CocoDataset):
#
#     CLASSES = ('1', '2', '3', '4', '5')


class RocketDataset(CocoDataset):

    CLASSES = ('1', '2', '3', '4', '5')

    def _parse_ann_info(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            # This config verified
            # if ann['area'] <= 0 or w < 10 or h < 10:
            #     continue
            # if ann['area'] <= 50 or max(w, h) < 10:
            #     continue
            # TODO: make the threshold a paramater in config
            if ann['area'] <= 80 or max(w, h) < 12:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def evaluate(self,results,
                 metric='rbbox',
                 logger=None,
                 classnames=['1', '2', '3', '4', '5'],
                 **eval_config):
        """Evaluation in Rocket protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]
        """

        # metrics = metric if isinstance(metric, list) else [metric]
        # allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        # for metric in metrics:
        #     if metric not in allowed_metrics:
        #         raise KeyError(f'metric {metric} is not supported')
        #
        # result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        #
        # eval_results = {}
        # cocoGt = self.coco
        # for metric in metrics:
        #     msg = f'Evaluating {metric}...'
        #     if logger is None:
        #         msg = '\n' + msg
        #     print_log(msg, logger=logger)
        #
        #     if metric == 'proposal_fast':
        #         ar = self.fast_eval_recall(
        #             results, proposal_nums, iou_thrs, logger='silent')
        #         log_msg = []
        #         for i, num in enumerate(proposal_nums):
        #             eval_results[f'AR@{num}'] = ar[i]
        #             log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
        #         log_msg = ''.join(log_msg)
        #         print_log(log_msg, logger=logger)
        #         continue
        #
        #     if metric not in result_files:
        #         raise KeyError(f'{metric} is not in results')
        #     try:
        #         cocoDt = cocoGt.loadRes(result_files[metric])
        #     except IndexError:
        #         print_log(
        #             'The testing results of the whole dataset is empty.',
        #             logger=logger,
        #             level=logging.ERROR)
        #         break
        #
        #     iou_type = 'bbox' if metric == 'proposal' else metric
        #     cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        #     cocoEval.params.catIds = self.cat_ids
        #     cocoEval.params.imgIds = self.img_ids
        #     if metric == 'proposal':
        #         cocoEval.params.useCats = 0
        #         cocoEval.params.maxDets = list(proposal_nums)
        #         cocoEval.evaluate()
        #         cocoEval.accumulate()
        #         cocoEval.summarize()
        #         metric_items = [
        #             'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
        #             'AR_l@1000'
        #         ]
        #         for i, item in enumerate(metric_items):
        #             val = float(f'{cocoEval.stats[i + 6]:.3f}')
        #             eval_results[item] = val
        #     else:
        #         cocoEval.evaluate()
        #         cocoEval.accumulate()
        #         cocoEval.summarize()
        #         if classwise:  # Compute per-category AP
        #             # Compute per-category AP
        #             # from https://github.com/facebookresearch/detectron2/
        #             precisions = cocoEval.eval['precision']
        #             # precision: (iou, recall, cls, area range, max dets)
        #             assert len(self.cat_ids) == precisions.shape[2]
        #
        #             results_per_category = []
        #             for idx, catId in enumerate(self.cat_ids):
        #                 # area range index 0: all area ranges
        #                 # max dets index -1: typically 100 per image
        #                 nm = self.coco.loadCats(catId)[0]
        #                 precision = precisions[:, :, idx, 0, -1]
        #                 precision = precision[precision > -1]
        #                 if precision.size:
        #                     ap = np.mean(precision)
        #                 else:
        #                     ap = float('nan')
        #                 results_per_category.append(
        #                     (f'{nm["name"]}', f'{float(ap):0.3f}'))
        #
        #             num_columns = min(6, len(results_per_category) * 2)
        #             results_flatten = list(
        #                 itertools.chain(*results_per_category))
        #             headers = ['category', 'AP'] * (num_columns // 2)
        #             results_2d = itertools.zip_longest(*[
        #                 results_flatten[i::num_columns]
        #                 for i in range(num_columns)
        #             ])
        #             table_data = [headers]
        #             table_data += [result for result in results_2d]
        #             table = AsciiTable(table_data)
        #             print_log('\n' + table.table, logger=logger)
        #
        #         metric_items = [
        #             'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
        #         ]
        #         for i in range(len(metric_items)):
        #             key = f'{metric}_{metric_items[i]}'
        #             val = float(f'{cocoEval.stats[i]:.3f}')
        #             eval_results[key] = val
        #         ap = cocoEval.stats[:6]
        #         eval_results[f'{metric}_mAP_copypaste'] = (
        #             f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
        #             f'{ap[4]:.3f} {ap[5]:.3f}')
        # if tmp_dir is not None:
        #     tmp_dir.cleanup()

        msg = f'Evaluating {metric}...'
        msg = '\n' + msg +'\n'


        classaps = []
        orgin_labelpath = os.path.join(os.path.split(eval_config['ann_file'])[0], 'labelTxt')
        map = 0
        for classname in classnames:
            rec, prec, ap = voc_eval(results,
                                     orgin_labelpath,
                                     eval_config['img_prefix'],
                                     classname,
                                     ovthresh=0.5,
                                     use_07_metric=True)
            map = map + ap
            msg = msg + 'class{} AP: {:.2f}, '.format(classname, ap)
            # print(classname, 'ap: ', ap)
            classaps.append(ap)

        map = map/len(classnames)
        msg = msg + 'mAP: {:.2f}'.format(map)

        eval_results = {}
        print_log(msg, logger=logger)
        return eval_results