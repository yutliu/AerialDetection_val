import pickle

pkl_file = '../work_dirs/cascade_mask_rcnn_r50_fpn_1x_dota/result/result.pkl'
with open(pkl_file, 'rb') as f:
    result = pickle.load(f, encoding='bytes')

print(result)


