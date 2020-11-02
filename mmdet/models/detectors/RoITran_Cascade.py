from __future__ import division

import torch
import torch.nn as nn

from .base_new import BaseDetectorNew
from .test_mixins import RPNTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import (build_assigner, bbox2roi, dbbox2roi, bbox2result, build_sampler,
                        dbbox2result, merge_aug_masks, roi2droi, mask2poly,
                        get_best_begin_point, polygonToRotRectangle_batch,
                        gt_mask_bp_obbs_list, choose_best_match_batch,
                        choose_best_Rroi_batch, dbbox_rotate_mapping, bbox_rotate_mapping)
from mmdet.core import (bbox_mapping, merge_aug_proposals, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms, merge_rotate_aug_proposals,
                        merge_rotate_aug_bboxes, multiclass_nms_rbbox)
import copy
from mmdet.core import RotBox2Polys, polygonToRotRectangle_batch
@DETECTORS.register_module
class RoITran_Cascade(BaseDetectorNew, RPNTestMixin):

    def __init__(self,
                 num_stages,
                 backbone,
                 neck=None,
                 shared_head=None,
                 shared_head_rbbox=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 rbbox_roi_extractor=None,
                 rbbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None

        assert rbbox_roi_extractor is not None
        assert rbbox_head is not None
        super(RoITran_Cascade, self).__init__()

        self.num_stages = num_stages
        self.backbone = builder.build_backbone(backbone)


        if neck is not None:
            self.neck = builder.build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if shared_head_rbbox is not None:
            self.shared_head_rbbox = builder.build_shared_head(shared_head_rbbox)

        if bbox_head is not None:
            self.bbox_roi_extractor = nn.ModuleList()
            self.bbox_head = nn.ModuleList()
            if not isinstance(bbox_roi_extractor, list):
                bbox_roi_extractor = [
                    bbox_roi_extractor for _ in range(num_stages)
                ]
            if not isinstance(bbox_head, list):
                bbox_head = [bbox_head for _ in range(num_stages)]
            assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
            for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
                self.bbox_roi_extractor.append(
                    builder.build_roi_extractor(roi_extractor))
                self.bbox_head.append(builder.build_head(head))
        # import pdb
        # pdb.set_trace()
        if rbbox_head is not None:
            self.rbbox_roi_extractor = builder.build_roi_extractor(
                rbbox_roi_extractor)
            self.rbbox_head = builder.build_head(rbbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.rbbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(RoITran_Cascade, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_shared_head_rbbox:
            self.shared_head_rbbox.init_weights(pretrained=pretrained)
        if self.with_bbox:
            for i in range(self.num_stages):
                if self.with_bbox:
                    self.bbox_roi_extractor[i].init_weights()
                    self.bbox_head[i].init_weights()
            if self.with_mask:
                if not self.share_roi_extractor:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()
        if self.with_rbbox:
            self.rbbox_roi_extractor.init_weights()
            self.rbbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses = dict()

        # trans gt_masks to gt_obbs
        gt_obbs = gt_mask_bp_obbs_list(gt_masks)

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)

            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
                bbox_sampler = build_sampler(
                    rcnn_train_cfg.sampler, context=self)
                num_imgs = img.size(0)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                            rois)

            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            cls_score, bbox_pred = bbox_head(bbox_feats)


            if i < self.num_stages - 1:
                bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes,
                                                    gt_labels, rcnn_train_cfg)
                loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
                for name, value in loss_bbox.items():
                    if 'loss' in name:
                        losses['s{}.{}'.format(i, name)] = (value * lw if 'cls' in name else value)
                    else:
                        losses['s{}.{}'.format(i, name)] = (value)

                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = bbox_targets[0]  # bbox_targets is a tuple
                with torch.no_grad():
                    proposal_list = bbox_head.refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)

            if i == self.num_stages - 1:
                ## rbbox
                rbbox_targets = bbox_head.get_target(
                    sampling_results, gt_masks, gt_labels, rcnn_train_cfg)

                loss_bbox = bbox_head.loss(cls_score, bbox_pred,
                                                *rbbox_targets)
                # losses.update(loss_bbox)
                for name, value in loss_bbox.items():
                    if 'loss' in name:
                        losses['s{}.{}'.format(i, name)] = (value * lw if 'cls' in name else value)
                    else:
                        losses['s{}.{}'.format(i, name)] = (value)

                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = rbbox_targets[0]
                with torch.no_grad():
                    # import pdb
                    # pdb.set_trace()
                    rotated_proposal_list = bbox_head.refine_rbboxes(
                        roi2droi(rois), roi_labels, bbox_pred, pos_is_gts, img_meta
                    )
                # import pdb
                # pdb.set_trace()
                # assign gts and sample proposals (rbb assign)
                if self.with_rbbox:
                    bbox_assigner = build_assigner(self.train_cfg.rcnn[-1].assigner)
                    bbox_sampler = build_sampler(
                        self.train_cfg.rcnn[-1].sampler, context=self)
                    num_imgs = img.size(0)
                    if gt_bboxes_ignore is None:
                        gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    sampling_results = []
                    for i in range(num_imgs):
                        gt_obbs_best_roi = choose_best_Rroi_batch(gt_obbs[i])
                        assign_result = bbox_assigner.assign(
                            rotated_proposal_list[i], gt_obbs_best_roi, gt_bboxes_ignore[i],
                            gt_labels[i])
                        sampling_result = bbox_sampler.sample(
                            assign_result,
                            rotated_proposal_list[i],
                            torch.from_numpy(gt_obbs_best_roi).float().to(rotated_proposal_list[i].device),
                            gt_labels[i],
                            feats=[lvl_feat[i][None] for lvl_feat in x])
                        sampling_results.append(sampling_result)

                if self.with_rbbox:
                    # (batch_ind, x_ctr, y_ctr, w, h, angle)
                    rrois = dbbox2roi([res.bboxes for res in sampling_results])
                    # feat enlarge
                    # rrois[:, 3] = rrois[:, 3] * 1.2
                    # rrois[:, 4] = rrois[:, 4] * 1.4
                    rrois[:, 3] = rrois[:, 3] * self.rbbox_roi_extractor.w_enlarge
                    rrois[:, 4] = rrois[:, 4] * self.rbbox_roi_extractor.h_enlarge
                    rbbox_feats = self.rbbox_roi_extractor(x[:self.rbbox_roi_extractor.num_inputs],
                                                           rrois)
                    if self.with_shared_head_rbbox:
                        rbbox_feats = self.shared_head_rbbox(rbbox_feats)
                    cls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
                    rbbox_targets = self.rbbox_head.get_target_rbbox(sampling_results, gt_obbs,
                                                                     gt_labels, self.train_cfg.rcnn[-1])
                    loss_rbbox = self.rbbox_head.loss(cls_score, rbbox_pred, *rbbox_targets)
                    for name, value in loss_rbbox.items():
                        losses['rbbbox_s{}.{}'.format(4, name)] = (value )

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']

        rcnn_test_cfg = self.test_cfg.rcnn
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        bbox_pred = []

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            bbox_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            cls_score, bbox_pred = bbox_head(bbox_feats)
            ms_scores.append(cls_score)

            # if self.test_cfg.keep_all_stages:
            #     det_bboxes, det_labels = bbox_head.get_det_bboxes(
            #         rois,
            #         cls_score,
            #         bbox_pred,
            #         img_shape,
            #         scale_factor,
            #         rescale=rescale,
            #         cfg=rcnn_test_cfg)
            #     bbox_result = bbox2result(det_bboxes, det_labels,
            #                               bbox_head.num_classes)
            #     ms_bbox_result['stage{}'.format(i)] = bbox_result

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])
        cls_score = sum(ms_scores) / self.num_stages

        bbox_head = self.bbox_head[-1]
        bbox_label = cls_score.argmax(dim=1)
        rrois = bbox_head.regress_by_class_rbbox(roi2droi(rois), bbox_label, bbox_pred,
                                                      img_meta[0])

        rrois_enlarge = copy.deepcopy(rrois)
        rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rbbox_roi_extractor.w_enlarge
        rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rbbox_roi_extractor.h_enlarge
        rbbox_feats = self.rbbox_roi_extractor(
            x[:len(self.rbbox_roi_extractor.featmap_strides)], rrois_enlarge)
        if self.with_shared_head_rbbox:
            rbbox_feats = self.shared_head_rbbox(rbbox_feats)

        rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
        det_rbboxes, det_labels = self.rbbox_head.get_det_rbboxes(
            rrois,
            rcls_score,
            rbbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        rbbox_results = dbbox2result(det_rbboxes, det_labels,
                                     self.rbbox_head.num_classes)

        return rbbox_results

    def aug_test(self, imgs, img_metas, proposals=None, rescale=None):
        # raise NotImplementedError
        # import pdb; pdb.set_trace()
        proposal_list = self.aug_test_rpn_rotate(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)

        rcnn_test_cfg = self.test_cfg.rcnn

        aug_rbboxes = []
        aug_rscores = []
        for x, img_meta in zip(self.extract_feats(imgs), img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)

            angle = img_meta[0]['angle']
            # print('img shape: ', img_shape)
            if angle != 0:
                try:

                    proposals = bbox_rotate_mapping(proposal_list[0][:, :4], img_shape,
                                                    angle)
                except:
                    import pdb; pdb.set_trace()
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            cls_score, bbox_pred = self.bbox_head(roi_feats)


            bbox_label = cls_score.argmax(dim=1)
            rrois = self.bbox_head.regress_by_class_rbbox(roi2droi(rois), bbox_label,
                                                          bbox_pred,
                                                          img_meta[0])

            rrois_enlarge = copy.deepcopy(rrois)
            rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rbbox_roi_extractor.w_enlarge
            rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rbbox_roi_extractor.h_enlarge
            rbbox_feats = self.rbbox_roi_extractor(
                x[:len(self.rbbox_roi_extractor.featmap_strides)], rrois_enlarge)
            if self.with_shared_head_rbbox:
                rbbox_feats = self.shared_head_rbbox(rbbox_feats)

            rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
            rbboxes, rscores = self.rbbox_head.get_det_rbboxes(
                rrois,
                rcls_score,
                rbbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=None)
            aug_rbboxes.append(rbboxes)
            aug_rscores.append(rscores)

        merged_rbboxes, merged_rscores = merge_rotate_aug_bboxes(
            aug_rbboxes, aug_rscores, img_metas, rcnn_test_cfg
        )
        det_rbboxes, det_rlabels = multiclass_nms_rbbox(
            merged_rbboxes, merged_rscores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)

        if rescale:
            _det_rbboxes = det_rbboxes
        else:
            _det_rbboxes = det_rbboxes.clone()
            _det_rbboxes[:, :4] *= img_metas[0][0]['scale_factor']

        rbbox_results = dbbox2result(_det_rbboxes, det_rlabels,
                                     self.rbbox_head.num_classes)
        return rbbox_results









