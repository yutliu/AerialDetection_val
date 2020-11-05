import cv2
import os
import numpy as np
import argparse

# def parse_args():
#     parser = argparse.ArgumentParser(description='Result generate')
#     parser.add_argument('config', default='')
#     parser.add_argument('--result_dir', default="")
#     parser.add_argument('--save_dir', default="")
#     args = parser.parse_args()
#     for name in args.config.split('/'):
#         if '.py' in name:
#             dir_name = name.split('.')[0]
#     assert args.config != [], 'config is empty'
#     args.result_dir = os.path.join('work_dirs', dir_name, 'Task1_results')
#     args.save_dir = os.path.join('work_dirs', dir_name)
#
#     return args

if __name__ == '__main__':
    # args = parse_args()
    test_dir = "/media/adminer/data/Rocketforce/SummaryData_mmdet_add100/test1024_2/images/"
    result_dir = 'work_dirs/cascade_rcnn_RoITrans_x101_fpn_1x_dota/Task1_results_nms'
    save_dir = 'work_dirs/cascade_rcnn_RoITrans_x101_fpn_1x_dota'
    txt_name = '科目四_南信大+华航队.txt'
    if os.path.exists(os.path.join(save_dir, txt_name)):
        os.remove(os.path.join(save_dir, txt_name))
    class_names = {'1', '2', '3', '4', '5'}
    all_detdict = {}
    for class_name in class_names:
        txt_path = os.path.join(result_dir, class_name+'.txt')
        with open(txt_path, 'r') as f:
            txt_result = f.readlines()
            for each_det in txt_result:
                each_det = each_det.strip().split(' ')
                each_det.insert(1, class_name)
                each_result = [float(value) for i,value in enumerate(each_det) if i > 0]
                if each_det[0] not in all_detdict:
                    all_detdict[each_det[0]] = each_result
                else:
                    all_detdict[each_det[0]].extend(each_result)

    for img_name, each_result in all_detdict.items():
        image_path = os.path.join(test_dir, img_name)
        img = cv2.imread(image_path)
        cat_pts0 = np.asarray(each_result, np.float32).reshape(-1, 10)
        save_txtpath = os.path.join(save_dir, txt_name)

        for cat_pt in cat_pts0:
            cat = int(cat_pt[0])
            score = cat_pt[1]
            pt = cat_pt[2:]
            with open(save_txtpath, 'a+') as f:
                oldimg_name = img_name.split('_')[0] + '.tif'
                f.write('{} {} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                    oldimg_name, cat, score, pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]))
        #     tl = pt[0, :]
        #     tr = pt[1, :]
        #     br = pt[2, :]
        #     bl = pt[3, :]
        #     cv2.line(img, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 0, 255), 1, 1)
        #     cv2.line(img, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), (255, 0, 255), 1, 1)
        #     cv2.line(img, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), (0, 255, 255), 1, 1)
        #     cv2.line(img, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), (255, 0, 0), 1, 1)
        #     cv2.putText(img, '{} {:.2f}'.format(cat, score), (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
        #                 (0, 0, 255), 1, 1)
            # cv2.imwrite(f'gt/{self.img_ids[index]}.png', img)
            # cv2.namedWindow('img', 0);
            # cv2.resizeWindow('img', 800, 800);
            # cv2.imshow('img', np.uint8(img))
        # k = cv2.waitKey(0) & 0xFF
        # if k == ord('q'):
        #     cv2.destroyAllWindows()







