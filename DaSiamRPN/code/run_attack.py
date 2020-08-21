
# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import cv2
from utils import rect_2_cxy_wh, cxy_wh_2_rect, get_subwindow_tracking

from os.path import realpath, dirname, join, isdir, exists

import matplotlib.pyplot as plt
import random


def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)
    total_stride = 8
    score_size = (instance_size-exemplar_size)/total_stride+1
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    # adaptive change search region #
    adaptive = True

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1

def rtaa_attack(net, x_init, x, gt, target_pos, target_sz, scale_z, p, eps=10, alpha=1, iteration=10, x_val_min=0, x_val_max=255):
    x = Variable(x.data)
    x_adv = Variable(x_init.data, requires_grad=True)

    alpha = eps * 1.0 / iteration

    for i in range(iteration):
        delta, score = net(x_adv)

        score_temp = score.permute(1, 2, 3, 0).contiguous().view(2, -1)
        score = torch.transpose(score_temp, 0, 1)
        delta1 = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()

        # calculate proposals
        gt_cen = rect_2_cxy_wh(gt)
        gt_cen = np.tile(gt_cen, (p.anchor.shape[0], 1))
        gt_cen[:, 0] = ((gt_cen[:, 0] - target_pos[0]) * scale_z - p.anchor[:, 0]) / p.anchor[:, 2]
        gt_cen[:, 1] = ((gt_cen[:, 1] - target_pos[1]) * scale_z - p.anchor[:, 1]) / p.anchor[:, 3]
        gt_cen[:, 2] = np.log(gt_cen[:, 2] * scale_z) / p.anchor[:, 2]
        gt_cen[:, 3] = np.log(gt_cen[:, 3] * scale_z) / p.anchor[:, 3]

        # create pseudo proposals randomly
        gt_cen_pseudo = rect_2_cxy_wh(gt)
        gt_cen_pseudo = np.tile(gt_cen_pseudo, (p.anchor.shape[0], 1))

        rate_xy1 = np.random.uniform(0.3, 0.5)
        rate_xy2 = np.random.uniform(0.3, 0.5)
        rate_wd = np.random.uniform(0.7, 0.9)

        gt_cen_pseudo[:, 0] = ((gt_cen_pseudo[:, 0] - target_pos[0] - rate_xy1 * gt_cen_pseudo[:, 2]) * scale_z - p.anchor[:, 0]) / p.anchor[:, 2]
        gt_cen_pseudo[:, 1] = ((gt_cen_pseudo[:, 1] - target_pos[1] - rate_xy2 * gt_cen_pseudo[:, 3]) * scale_z - p.anchor[:, 1]) / p.anchor[:, 3]
        gt_cen_pseudo[:, 2] = np.log(gt_cen_pseudo[:, 2] * rate_wd * scale_z) / p.anchor[:, 2]
        gt_cen_pseudo[:, 3] = np.log(gt_cen_pseudo[:, 3] * rate_wd * scale_z) / p.anchor[:, 3]


        delta[0, :] = (delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0])/ scale_z + target_pos[0]
        delta[1, :] = (delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1])/ scale_z + target_pos[1]
        delta[2, :] = (np.exp(delta[2, :]) * p.anchor[:, 2])/scale_z
        delta[3, :] = (np.exp(delta[3, :]) * p.anchor[:, 3])/scale_z
        location = np.array([delta[0] - delta[2] / 2, delta[1] - delta[3] / 2, delta[2], delta[3]])

        label = overlap_ratio(location, gt)

        # set thresold to define positive and negative samples, following the training step
        iou_hi = 0.6
        iou_low = 0.3

        # make labels
        y_pos = np.where(label > iou_hi, 1, 0)
        y_pos = torch.from_numpy(y_pos).cuda().long()
        y_neg = np.where(label < iou_low, 0, 1)
        y_neg = torch.from_numpy(y_neg).cuda().long()
        pos_index = np.where(y_pos.cpu() == 1)
        neg_index = np.where(y_neg.cpu() == 0)
        index = np.concatenate((pos_index, neg_index), axis=1)

        # make pseudo lables
        y_pos_pseudo = np.where(label > iou_hi, 0, 1)
        y_pos_pseudo = torch.from_numpy(y_pos_pseudo).cuda().long()
        y_neg_pseudo = np.where(label < iou_low, 1, 0)
        y_neg_pseudo = torch.from_numpy(y_neg_pseudo).cuda().long()

        y_truth = y_pos
        y_pseudo = y_pos_pseudo

        # calculate classification loss
        loss_truth_cls = -F.cross_entropy(score[index], y_truth[index])
        loss_pseudo_cls = -F.cross_entropy(score[index], y_pseudo[index])
        loss_cls = (loss_truth_cls - loss_pseudo_cls) * (1)

        # calculate regression loss
        loss_truth_reg = -rpn_smoothL1(delta1, gt_cen, y_pos)
        loss_pseudo_reg = -rpn_smoothL1(delta1, gt_cen_pseudo, y_pos)
        loss_reg = (loss_truth_reg - loss_pseudo_reg) * (5)

        # final adversarial loss
        loss = loss_cls + loss_reg

        # calculate the derivative
        net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        loss.backward(retain_graph=True)

        adv_grad = where((x_adv.grad > 0) | (x_adv.grad < 0), x_adv.grad, 0)
        adv_grad = torch.sign(adv_grad)
        x_adv = x_adv - alpha * adv_grad

        x_adv = where(x_adv > x + eps, x + eps, x_adv)
        x_adv = where(x_adv < x - eps, x - eps, x_adv)
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)

    return x_adv

def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p, f, gt, state):
    delta, score = net(x_crop)

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]



    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score

    # window float
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])


    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

    return target_pos, target_sz, score[best_pscore_id]

def SiamRPN_init(im, target_pos, target_sz, net):
    state = dict()
    p = TrackerConfig()
    p.update(net.cfg)
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    if p.adaptive:
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287  # small object big search region
        else:
            p.instance_size = 271

        p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1

    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))

    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    net.temple(z.cuda())

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state

def SiamRPN_track(state, im, f, last_result, att_per, def_per, image_save, iter=10):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    x_crop = x_crop.cuda()

    # adversarial attack
    if type(att_per) != type(0):
        att_per = att_per.cpu().detach().numpy()
        att_per = np.resize(att_per, (1, x_crop.shape[1], x_crop.shape[2], x_crop.shape[3]))
        att_per = torch.from_numpy(att_per).cuda()
    x_crop_init = x_crop + att_per * 1
    x_crop_init = torch.clamp(x_crop_init, 0, 255)
    x_adv1 = rtaa_attack(net, x_crop_init, x_crop, last_result, target_pos, target_sz, scale_z, p, iteration=iter)
    att_per = x_adv1 - x_crop


    target_pos, target_sz, score = tracker_eval(net, x_adv1, target_pos, target_sz * scale_z, window, scale_z, p, f, last_result, state)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state, att_per, def_per

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''
    rect1 = np.transpose(rect1)

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def rpn_smoothL1(input, target, label):
    r"""
    :param input: torch.Size([1, 1125, 4])
    :param target: torch.Size([1, 1125, 4])
            label: (torch.Size([1, 1125]) pos neg or ignore
    :return:
    """
    input = torch.transpose(input, 0, 1)
    pos_index = np.where(label.cpu() == 1)#changed
    target = torch.from_numpy(target).cuda().float()
    loss = F.smooth_l1_loss(input[pos_index], target[pos_index], reduction='sum')


    return loss

def rect_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2, rect[2], rect[3]])




