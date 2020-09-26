import torch.nn as nn
import torch.nn.functional as F

from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss


# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred, target, gamma, alpha)
    # TODO: find a proper way to handle the shape of weight
    if weight is not None:
        weight = weight.view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        #assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        #self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.size_average = True

        print('Apply Focal Loss')

    #def forward(self,
    #            pred,
    #            target,
    #            weight=None,
    #            avg_factor=None,
    #            reduction_override=None):
    #    assert reduction_override in (None, 'none', 'mean', 'sum')
    #    reduction = (
    #        reduction_override if reduction_override else self.reduction)
    #    if self.use_sigmoid:
    #        loss_cls = self.loss_weight * sigmoid_focal_loss(
    #            pred,
    #            target,
    #            weight,
    #            gamma=self.gamma,
    #            alpha=self.alpha,
    #            reduction=reduction,
    #            avg_factor=avg_factor)
    #    else:
    #        raise NotImplementedError
    #    return loss_cls

    def forward(self, input, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        #reduction = (
        #    reduction_override if reduction_override else self.reduction)
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        #if self.alpha is not None:
        #    assert False

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: 
            return loss.mean() * self.loss_weight
        else: 
            return loss.sum() * self.loss_weight

