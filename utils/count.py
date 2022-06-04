import os
import numpy as np
import torch

VAL_SCAN_NAMES = [line.rstrip() for line in open('/home/dhanalaxmi.gaddam/3D_New/scannet/meta_data/scannetv2_val.txt')] 
OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
GT_PATH = '/home/dhanalaxmi.gaddam/3D_New/scannet/scannet_train_detection_data' # path of data dumped with scripts in scannet folder 
PRED_PATH = '/home/dhanalaxmi.gaddam/MRCH3D/11_May/result' # path of predictions 

def select_bbox(bboxes):
    choose_ids = []
    for i in range(bboxes.shape[0]):
        if bboxes[i,-1] in OBJ_CLASS_IDS:
            choose_ids.append(i)
    bboxes = bboxes[choose_ids]
    return bboxes


def mrmse(non_zero,count_pred, count_gt):
    ## compute mrmse
    nzero_mask=torch.ones(len(count_gt))
    if non_zero==1:
        nzero_mask=torch.zeros(len(count_gt))
        nzero_mask[count_gt!=0]=1
    mrmse=torch.pow(count_pred - count_gt, 2)
    mrmse = torch.mul(mrmse, nzero_mask)
    mrmse = torch.sum(mrmse, 0)
    nzero = torch.sum(nzero_mask, 0)
    mrmse = torch.div(mrmse, nzero)
    mrmse = torch.sqrt(mrmse)
#     print(mrmse.size())
    mrmse = torch.mean(mrmse)
    return mrmse

def rel_mrmse(non_zero,count_pred, count_gt):
    ## compute relative mrmse
    nzero_mask=torch.ones(len(count_gt))
    if non_zero==1:
        nzero_mask=torch.zeros(len(count_gt))
        nzero_mask[count_gt!=0]=1
    num = torch.pow(count_pred - count_gt, 2)
    denom = count_gt.clone()
    denom = denom+1
    rel_mrmse = torch.div(num, denom)
    rel_mrmse = torch.mul(rel_mrmse, nzero_mask)
    rel_mrmse = torch.sum(rel_mrmse, 0)
    nzero = torch.sum(nzero_mask, 0)
    rel_mrmse = torch.div(rel_mrmse, nzero)
    rel_mrmse = torch.sqrt(rel_mrmse)
    rel_mrmse = torch.mean(rel_mrmse)
    return rel_mrmse

def export_one_scan(scan_name):
    gt_bbox = np.load(os.path.join(GT_PATH, scan_name+'_all_noangle_40cls.npy'))
    gt_bbox = select_bbox(np.unique(gt_bbox,axis=0))
    pred_proposals = np.load(os.path.join(PRED_PATH, 'opt'+scan_name+'_nms.npy'))
    
    gt_count = []
    pred_count = []

    for i in range(gt_bbox.shape[0]):
        x = int(gt_bbox[i,-1])
        gt_count.append(x)
        

    for i in range(pred_proposals.shape[0]):
        x = int(OBJ_CLASS_IDS[int(pred_proposals[i,-1])-1])
        pred_count.append(x)
    
    return gt_count, pred_count


N = len(VAL_SCAN_NAMES)
cat = len(OBJ_CLASS_IDS)
c_ik = np.zeros((cat,N))
c_ik_hat_h3d = np.zeros((cat,N))
c = 0
p = 0
for ind, k in enumerate(OBJ_CLASS_IDS):

    for i, scan_name in enumerate(sorted(VAL_SCAN_NAMES)):
        gt_objects, h3d_pred_objects = export_one_scan(scan_name)
        c = gt_objects.count(k)
        p = h3d_pred_objects.count(k)
        c_ik[ind,i] = c
        c_ik_hat_h3d[ind,i] = p
print(c_ik)
print(c_ik_hat_h3d)

for i in range(c_ik.shape[0]):
    g_cat_1 = c_ik[:,i]
    p_cat_1 = c_ik_hat_h3d[:,i]
    rmse_cat0 = mrmse(0, torch.from_numpy(p_cat_1).float(), torch.from_numpy(g_cat_1).float())
    rmse_cat1 = mrmse(1, torch.from_numpy(p_cat_1).float(), torch.from_numpy(g_cat_1).float())
    rel_rmse_cat0 = rel_mrmse(0, torch.from_numpy(p_cat_1).float(), torch.from_numpy(g_cat_1).float())
    rel_rmse_cat1 = rel_mrmse(1, torch.from_numpy(p_cat_1).float(), torch.from_numpy(g_cat_1).float())
    print(rmse_cat0, rmse_cat1, rel_rmse_cat0, rel_rmse_cat1)