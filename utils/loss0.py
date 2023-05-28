# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        #self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 0.4, 0.4, 0.4])
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors # anchors: [3, 3, 2]  3个 feature map 每个 feature map 上有3个 anchor(w,h) 
                                 # 这里的 anchor 尺寸是相对 feature map 的
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # targets为标签信息 (gt)
        # targets.shape = torch.Size([nt, 6]), 意味着这一张图片中有nt个真实框，每个真实框有六列参数
        # 其中第1列为图片在当前 batch 的 id 号，第2列为类别 id，后面依次是归一化了的 gt 框的 x,y,w,h 坐标。
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # gain 是为了后面将 targets=[na,nt,7] 中的归一化了的 x y w h 映射到相对 feature map 尺度上
        # 7: image_index+class+xywh+anchor_index
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        # ai 从 torch.Size([1, 3]) => torch.Size([3, 1]) => torch.Size([3, nt])
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # targets 从 torch.Size([nt, 6]) => torch.Size([3, nt, 6]) => torch.Size([3, nt, 7])，最后一列加上了anchor id
        # ai[..., None].shape = torch.Size([3, nt, 1])
        # 先假设所有的 target 对三个 anchor 都是正样本(复制三份) 再进行筛选，并将 ai 加进去，标记当前是哪个 anchor 的 target
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        # off 是偏置矩阵:
        # tensor([[ 0.00000,  0.00000],
        #         [ 0.50000,  0.00000],
        #         [ 0.00000,  0.50000],
        #         [-0.50000,  0.00000],
        #         [ 0.00000, -0.50000]])
        
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            # anchors => (3, 3, 2), anchors[i] => (3, 2)
            # anchors 是在 feature map 中的尺寸
            anchors = self.anchors[i]
            # p[i].shape => (20, 20) or (40, 40) or (80, 80)
            # gain 从 (1, 1, 1, 1, 1, 1, 1) => (1, 1, 20, 20, 20, 20, 1)
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # 将 target 中 x, y, w, h 的归一化尺度放大到相对当前 feature map 的坐标尺度, 对应 t[..., 2:6]
            t = targets * gain  # t => (3,nt,7)
            if nt:
                # 筛选出和真实框 target 大小符合的 anchors
                # t[..., 4:6] => (3. nt, 2)
                # anchors[:, None] => (3, 1, 2)
                # r => (3. nt, 2)
                r = t[..., 4:6] / anchors[:, None]  # 真实框的宽高除以 anchors 的宽高
                # 观察宽高比，只选择宽高比在 0.25-4 之间的 anchors
                # 阈值参数保存在hyp文件中
                # j 是一个由 true 和 false 构成的 tensor， 形状为 (3, nt) 
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t'] 
                # yolov3 v4的筛选方法: wh_iou，GT与 anchor 的 wh_iou 超过一定的阈值就是正样本
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # 进行筛选
                # 得到筛选后的 targets， t 从 (3, nt, 7) => (nt_after, 7)
                # nt_after 是筛选后，满足宽高比条件的真实框个数
                # 第1列为图片在当前 batch 的 id，第2列为类别 id，后面依次是恢复后的 gt 框的 x,y,w,h 坐标
                # 最后一列是anchor id，一个真实框最多可以有三个 anchors 与之对应，id 为0，1，2

                # 筛选当前格子周围的格子，找到两个离 target 中心最近的格子。总共三个格子去计算一个 gt 框的 loss
                # feature map 上的原点在左上角，向右为 x 轴正坐标，向下为 y 轴正坐标
                gxy = t[:, 2:4]  # grid xy，取 target 中心的坐标 x，y ( feature map 左上角为坐标原点，向右向下为正坐标轴)
                gxi = gain[[2, 3]] - gxy  # 80-x, 80-y，得到 target 中心点相对于feature map右下角的坐标，向左向上为正坐标轴
                
                ###########################
                gwh = t[:, 4:6]
                gxy_grid = gxy.long()
                gxy_off = gxy - gxy_grid
                x_off = gxy_off[:,0]
                y_off = gxy_off[:,1]
                w = gwh[:,0]
                h = gwh[:,1]
                ###########################'''
                
                # 筛选中心坐标 距离当前 grid_cell 的左、上方偏移小于 g=0.5 且中心坐标必须大于1(坐标不能在边上，否则此时就没有4个格子了)
                # j: [nt_after] bool 如果是True表示当前target中心点所在的格子的左边格子也对该target进行回归(后续进行计算损失)
                # k: [nt_after] bool 如果是True表示当前target中心点所在的格子的上边格子也对该target进行回归(后续进行计算损失)
                j, k = ((gxy % 1 < g) & (gxy > 1)).T

                ###########################
                a = (w * 0.5 -  x_off) >= 0.25
                j = a & j
                #left = ((w * 0.5 -  x_off) >= 0.25  & (j))
                for i, n in enumerate(j):
                    if w[i] * 0.5 - x_off[i] <= 0.25 and n:
                        j[i] = False
                    elif n:
                        j[i] = True   
                for i, n in enumerate(k):
                    if h[i] * 0.5 - y_off[i] <= 0.25 and n:
                        k[i] = False
                    elif n:
                        k[i] = True  
                 ###########################'''
                
                # 筛选中心坐标 距离当前 grid_cell 的右、下方偏移小于 g=0.5 且中心坐标必须大于1(坐标不能在边上，否则此时就没有4个格子了)
                # l: [nt_after] bool 如果是True表示当前target中心点所在的格子的右边格子也对该target进行回归(后续进行计算损失)
                # m: [nt_after] bool 如果是True表示当前target中心点所在的格子的下边格子也对该target进行回归(后续进行计算损失)
                l, m = ((gxi % 1 < g) & (gxi > 1)).T

                ############################
                for i, n in enumerate(l):
                    if w[i] * 0.5 + x_off[i] <= 1.25 and n:
                        l[i] = False
                    elif n:
                        l[i] = True
                for i, n in enumerate(m):
                    if h[i] * 0.5 + y_off[i] <= 1.25 and n:
                        m[i] = False
                    elif n:
                        m[i] = True
                ############################'''
                
                # j: [5, nt_after]  torch.ones_like(j): 当前格子, 不需要筛选，全是True。  j, k, l, m: 左上右下格子的筛选结果
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # 在筛选出真实框格子周围格子后，找到2个离真实框中心最近的两个格子，并一同作为正样本
                # t (nt_after, 7) => (5, nt_after, 7) => (3 * nt_after, 7)
                # (5, nt_after, 7): 将原target信息复制四份，总共得到五份，循序分别为，本身，左，上，右，下
                # t.repeat((5, 1, 1))[j]能够依次在 本身，左，上，右，下 五个矩阵中筛选出符合的feature points
                # t (nt_after, 7) => (3 * nt_after, 7)
                t = t.repeat((5, 1, 1))[j]
                # t中每行的顺序为anchors[0]对应的feature map的GT本身坐标，anchors[1]对应的feature map的GT本身坐标，anchors[2]对应的feature map的GT本身坐标，依次以这样的顺序遍历左上右下
                # 得到所有筛选后的网格的中心相对于这个要预测的真实框所在网格边界（左右上下边框）的偏移量
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchor编号
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            # gij是所有被视为正样本的网格的左上角坐标
            gi, gj = gij.T  # grid indices，x and y

            # Append
            # indices列表加入一个tuple（image, anchor, grid y, grid x）
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch