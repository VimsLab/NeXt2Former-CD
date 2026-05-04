# encoding: utf-8

import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def hist_info(n_cl, pred, gt):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))
    confusionMatrix = np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                        minlength=n_cl ** 2).reshape(n_cl, n_cl)
    return confusionMatrix, labeled, correct

def compute_score(hist, correct, labeled):
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:]) # useless for NYUDv2
    recall_1 = hist[1,1]/(hist[1].sum())
    precision_1 = hist[1,1]/(hist[:,1].sum())

    # Per-class dice
    tp = np.diag(hist)
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp
    denom = (2 * tp + fp + fn)
    dice_per_class = np.divide(2 * tp, denom, out=np.full_like(tp, np.nan, dtype=float), where=denom > 0)
    dice_1 = dice_per_class[1] if dice_per_class.size > 1 else np.nan

    freq = hist.sum(1) / hist.sum()
    freq_IoU = (iou[freq > 0] * freq[freq > 0]).sum()

    classAcc = np.diag(hist) / hist.sum(axis=1)
    mean_pixel_acc = np.nanmean(classAcc)

    pixel_acc = correct / labeled

    return iou, recall_1, precision_1, mean_IoU, mean_IoU_no_back, freq_IoU, mean_pixel_acc, pixel_acc, dice_1, dice_per_class