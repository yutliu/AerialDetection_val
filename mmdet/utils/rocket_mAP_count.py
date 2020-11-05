
import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt
from DOTA_devkit import polyiou
from functools import partial
import cv2

def parse_gt(filename):
    objects = []
    target = ET.parse(filename).getroot()
    for obj in target.iter('HRSC_Object'):
        object_struct = {}
        difficult = int(obj.find('difficult').text)
        box_xmin = int(obj.find('box_xmin').text)  # bbox
        box_ymin = int(obj.find('box_ymin').text)
        box_xmax = int(obj.find('box_xmax').text)
        box_ymax = int(obj.find('box_ymax').text)
        mbox_cx = float(obj.find('mbox_cx').text)  # rbox
        mbox_cy = float(obj.find('mbox_cy').text)
        mbox_w = float(obj.find('mbox_w').text)
        mbox_h = float(obj.find('mbox_h').text)
        mbox_ang = float(obj.find('mbox_ang').text)*180/np.pi
        rect = ((mbox_cx, mbox_cy), (mbox_w, mbox_h), mbox_ang)
        pts_4 = cv2.boxPoints(rect)  # 4 x 2
        bl = pts_4[0,:]
        tl = pts_4[1,:]
        tr = pts_4[2,:]
        br = pts_4[3,:]
        object_struct['name'] = 'ship'
        object_struct['difficult'] = difficult
        object_struct['bbox'] = [float(tl[0]),
                                 float(tl[1]),
                                 float(tr[0]),
                                 float(tr[1]),
                                 float(br[0]),
                                 float(br[1]),
                                 float(bl[0]),
                                 float(bl[1])]
        objects.append(object_struct)
    return objects

def txt2gt(anno):
    anno = np.array(anno).reshape(-1, 9)
    obj_numbers = anno.shape[0]
    objects = []

    for obj_index in range(obj_numbers):
        each_anno = anno[obj_index, :]
        objclass = int(each_anno[0])
        pts = np.asarray(each_anno[1:].reshape(4, 2), np.float32)
        rect = cv2.minAreaRect(pts)
        pts_4 = cv2.boxPoints(rect)
        bl = pts_4[0,:]
        tl = pts_4[1,:]
        tr = pts_4[2,:]
        br = pts_4[3,:]
        object_struct={}
        object_struct['name'] = f'{objclass}'
        object_struct['difficult'] = 0
        object_struct['bbox'] = [float(tl[0]),
                                 float(tl[1]),
                                 float(tr[0]),
                                 float(tr[1]),
                                 float(br[0]),
                                 float(br[1]),
                                 float(bl[0]),
                                 float(bl[1])]
        objects.append(object_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
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


def voc_eval(pre_result,
             annopath,
             images_path,
             eval_class,
             ovthresh=0.5,
             use_07_metric=False):

    imagenames = os.listdir(images_path)
    label_txts = os.listdir(annopath)
    for index, label_txt in enumerate(label_txts):
        path = os.path.join(annopath, label_txt)
        context = np.loadtxt(path).reshape(-1, 10)[:, :-1]
        # context = [list(map(float, each)) for each in context]
        # context = np.array(context)
        # context = np.insert(context, 0, values=list(map(int, context[:,-1])), axis=1)[:, :-1]
        fig_name = label_txt.replace('.txt', '.tif')
        fig_arr = np.array([fig_name] * context.shape[0]).reshape(-1, 1)
        context = np.concatenate([fig_arr, context], axis=1)
        if index == 0:
            total_lable = context
        total_lable = np.concatenate([total_lable, context], axis=0)

    anno_dict = {}
    for each in total_lable:
        each_obj = [float(i) for i in each[1:]]
        new_obj = [int(each_obj[-1])]
        new_obj.extend(each_obj[:-1])
        if each[0] not in anno_dict:
            anno_dict[each[0]] = new_obj
        else:
            anno_dict[each[0]].extend(new_obj)
    recs = {}
    for i, imagename in enumerate(imagenames):
        # recs[imagename] = parse_gt(os.path.join(annopath.format(imagename)))
        recs[imagename] = txt2gt(anno_dict[imagename])
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == eval_class]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    #deal with pre result
    image_ids = [x.strip('.tif') for x in imagenames]

    pic_index = 0
    splitlines = []
    for each_pic_result in pre_result:
        class_index = 0
        pic_name = [imagenames[pic_index]]
        for each_class_result in each_pic_result:
            class_index += 1
            if each_class_result.shape[0] == 0:
                continue
            obj_number = each_class_result.shape[0]
            each_class_result = np.insert(each_class_result, 0, values=each_class_result[:, -1], axis=1)[:, :-1]
            for each_obj in each_class_result:
                temp = []
                temp.extend(pic_name)
                temp.extend(str(class_index))
                temp.extend(list(each_obj))
                splitlines.append(temp)
        pic_index += 1

    splitlines = [line[0:1] + line[2:] for line in splitlines if line[1] == eval_class]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    #print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    if len(confidence)>1:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)

        #print('check sorted_scores: ', sorted_scores)
        #print('check sorted_ind: ', sorted_ind)

        ## note the usage only in numpy not for list
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
    #print('check imge_ids: ', image_ids)
    #print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    # print('check fp:', fp)
    # print('check tp', tp)


    # print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    # if npos == 0:
    #     npos = 1
    recall = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(recall, prec, use_07_metric)

    return recall, prec, ap

def main():
    detpath = r'PATH_TO_BE_CONFIGURED/Task1_{:s}.txt'
    annopath = r'PATH_TO_BE_CONFIGURED/{:s}.txt' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    imagesetfile = r'PATH_TO_BE_CONFIGURED/test.txt'
    classnames = ['ship']
    classaps = []
    map = 0
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
                                 annopath,
                                 imagesetfile,
                                 classname,
                                 ovthresh=0.5,
                                 use_07_metric=True)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)

        # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
    # plt.show()
    map = map/len(classnames)
    print('map:', map)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)

if __name__ == '__main__':
    main()
