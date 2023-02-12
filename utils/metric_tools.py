import numpy as np
import torch.distributed as dist
import torch
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def clear(self):
        self.initialized = False


class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""

    def __init__(self, n_class, dataset_name):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class
        self.dataset_name = dataset_name

    def update_cm(self, pr, gt, weight=1):
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val, self.dataset_name)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum, self.dataset_name)
        return scores_dict


# gather hist from all ranks
def gather_hist(score):
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    score = torch.tensor(score).to(device, dtype=torch.float)
    dist.all_reduce(score, op=dist.ReduceOp.SUM)
    return score.cpu().numpy()


def cm2F1(confusion_matrix, dataset_name):
    hist = gather_hist(confusion_matrix)
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)

    return mean_F1


def cm2score(confusion_matrix, dataset_name):
    hist = gather_hist(confusion_matrix)
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
    pe = (sum_a1 * sum_a0).sum() / (hist.sum() + np.finfo(np.float32).eps) ** 2
    # kappa
    kappa = (acc - pe) / (1 - pe)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)

    # iou
    iu = (tp + np.finfo(np.float32).eps) / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    cls_iou = dict(zip(['iou_' + str(i) for i in range(n_class)], iu))
    cls_precision = dict(zip(['precision_' + str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_' + str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_' + str(i) for i in range(n_class)], F1))

    score_dict = {'OA': acc, 'miou': mean_iu, 'mf1': mean_F1, 'kappa': kappa}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """Compute confuse matrix"""

    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes) & (label_pred < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes ** 2).reshape(num_classes, num_classes)
        return hist

    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix
